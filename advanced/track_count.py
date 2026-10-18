"""
Unique people and vehicle counting with tracking (advanced 1 and 2).

Reads a live stream (or a recorded clip) continuously at a few frames per
second and tracks every person and vehicle with YOLOv8 + ByteTrack. Each
track ID is counted once, so a person who stays in view for many frames
counts as a single individual. New IDs that continue a track lost a few
seconds earlier in the same place are not counted again (see
UniqueCounter). Results are aggregated in fixed time windows (one CSV row
per window) and plotted at the end.

Tracking needs consecutive frames close in time. That is why
this script does not use capture_frame (which reopens the stream
every call).

Run from the project root:
    python -m advanced.track_count --duration 600
    python -m advanced.track_count --source clip.mp4 --fps 1 --exclude-zone ""
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from common.capture import OPEN_TIMEOUT_MS, READ_TIMEOUT_MS, CaptureError, resolve_stream_url

from common.count_people import load_model
from common.zones import is_excluded, parse_zone

# COCO class ids. Bicycles are left out (the rider is already a person)
PERSON = 0
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASSES = [PERSON, *VEHICLES]

TRACKER_CONFIG = Path(__file__).with_name("bytetrack.yaml")

RESULTS_DIR = Path("results/tracking")
DEFAULT_STREAM_FPS = 25.0
MAX_RECONNECTS = 5

# A new ID that starts less than REASSOCIATE_SECONDS after another one was
# lost, closer than its own height (at least REASSOCIATE_MIN_PX), is the
# same object
REASSOCIATE_SECONDS = 10.0
REASSOCIATE_MIN_PX = 40

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

FIELDS = [
    "window_start",
    "camera_id",
    "frames",
    "people_unique",
    "vehicles_unique",
    "people_reassociated",
    "vehicles_reassociated",
    "people_detections",
    "vehicles_detections",
    "people_max_frame",
]


class VideoSource:
    """
    Consecutive frames from a live stream or a video file.

    A live stream is reopened when it fails, up to MAX_RECONNECTS times.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_file = Path(source).is_file()
        self.reconnects = 0
        self.fps = DEFAULT_STREAM_FPS
        self._cap = cv2.VideoCapture()
        self._open()

    def _open(self) -> None:
        url = self.source if self.is_file else resolve_stream_url(self.source)

        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS,
        ])

        if not self._cap.isOpened():
            raise CaptureError(f"Could not open {self.source!r}")

        # HLS streams sometimes report a nonsense frame rate
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 120 else DEFAULT_STREAM_FPS

    def read(self, skip: int) -> np.ndarray | None:
        """
        Discard 'skip' frames and return the next one.

        grab() only reads the frame. retrieve() converts it to an image,
        so the discarded frames are cheaper than a full read().
        """
        while True:
            if all(self._cap.grab() for _ in range(skip + 1)):
                ok, frame = self._cap.retrieve()
                if ok:
                    return frame

            if self.is_file or self.reconnects >= MAX_RECONNECTS:
                return None

            self.reconnects += 1
            print(f"Stream lost, reconnection ({self.reconnects}/{MAX_RECONNECTS})")
            self._cap.release()
            time.sleep(2)
            try:
                self._open()
            except CaptureError as exc:
                print(f"Reconnection failed: {exc}")

    def release(self) -> None:
        self._cap.release()


@dataclass
class Track:
    """Sightings of one track ID (classes seen, and first and last box)."""
    classes: Counter[int]
    first_time: float
    first_box: Box
    last_time: float
    last_box: Box
    confirmed: bool = False  # Reached min_hits (counted or reassociated)

    def main_class(self) -> int:
        return self.classes.most_common(1)[0][0]


def _feet(box: Box) -> tuple[float, float]:
    """Bottom-centre point of a box, the same point used by the zones."""
    x1, _, x2, y2 = box
    return (x1 + x2) / 2, y2


class UniqueCounter:
    """
    Counts each track ID once, after it has been seen in 'min_hits' frames.

    Requiring a few sightings filters out IDs that live for one frame
    (false positives). The class of a track is the most frequent one
    among its sightings.

    Static objects (people sitting or standing, parked vehicles) often lose
    their ID when they are briefly missed or occluded, and ByteTrack gives
    them a new one. So a new ID that starts where another one of the same
    kind (person/vehicle) was last seen, less than REASSOCIATE_SECONDS
    earlier, is treated as the same object and is not counted again.
    """

    def __init__(self, min_hits: int) -> None:
        self.min_hits = min_hits
        self._tracks: dict[int, Track] = {}

    def update(
        self, track_id: int, cls: int, box: Box, t: float
    ) -> tuple[int, bool] | None:
        """
        Register one sighting of a track at time 't' (seconds).

        Returns:
            ``(class, reassociated)`` the first time the track reaches
            'min_hits' sightings, or None otherwise. ``reassociated`` is
            True when the track continues a lost one and must not be
            counted.
        """
        track = self._tracks.get(track_id)
        if track is None:
            track = Track(Counter(), t, box, t, box)
            self._tracks[track_id] = track

        track.classes[cls] += 1
        track.last_time, track.last_box = t, box

        if track.confirmed or sum(track.classes.values()) < self.min_hits:
            return None

        track.confirmed = True
        main_class = track.main_class()
        return main_class, self._continues_lost_track(track, main_class)

    def _continues_lost_track(self, track: Track, main_class: int) -> bool:
        """True if 'track' starts where a confirmed track was lost shortly before."""
        is_person = main_class == PERSON
        x, y = _feet(track.first_box)
        height = track.first_box[3] - track.first_box[1]
        max_distance = max(height, REASSOCIATE_MIN_PX)

        for other in self._tracks.values():
            if other is track or not other.confirmed:
                continue
            if (other.main_class() == PERSON) != is_person:
                continue
            if not 0 < track.first_time - other.last_time <= REASSOCIATE_SECONDS:
                continue

            other_x, other_y = _feet(other.last_box)
            if math.hypot(x - other_x, y - other_y) < max_distance:
                return True

        return False


def new_window(start: datetime, camera_id: str) -> dict:
    row = {field: 0 for field in FIELDS}
    row["window_start"] = start.isoformat(timespec="seconds")
    row["camera_id"] = camera_id
    return row


def process_result(
    result, zone, counter: UniqueCounter, window: dict, t: float
) -> None:
    """
    Add the tracks of one frame (at time 't', seconds) to the current window.
    """
    window["frames"] += 1
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return

    people_in_frame = 0
    for (x1, y1, x2, y2), cls, track_id in zip(
        boxes.xyxy.tolist(), boxes.cls.int().tolist(), boxes.id.int().tolist()
    ):
        if is_excluded(x1, y1, x2, y2, zone):
            continue

        if cls == PERSON:
            people_in_frame += 1
            window["people_detections"] += 1
        else:
            window["vehicles_detections"] += 1

        confirmed = counter.update(track_id, cls, (x1, y1, x2, y2), t)
        if confirmed is None:
            continue

        main_class, reassociated = confirmed
        kind = "people" if main_class == PERSON else "vehicles"
        window[f"{kind}_reassociated" if reassociated else f"{kind}_unique"] += 1

    window["people_max_frame"] = max(window["people_max_frame"], people_in_frame)


def plot_windows(rows: list[dict], path: Path) -> None:
    """Bar chart of unique people and vehicles per window."""
    x = np.arange(len(rows))
    labels = [row["window_start"][11:16] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, [row["people_unique"] for row in rows], 0.4, label="People")
    ax.bar(x + 0.2, [row["vehicles_unique"] for row in rows], 0.4, label="Vehicles")
    ax.set_xticks(x, labels, rotation=45)
    ax.set_xlabel("Window start")
    ax.set_ylabel("Unique count")
    ax.set_title(f"Unique people and vehicles ({rows[0]['camera_id']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--source",
        default=os.getenv("CAMERA_URL"),
        help="Camera page URL, stream URL or video file"
    )
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "camera"))
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Frames processed per second of video"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Aggregation window in seconds"
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Sightings needed before an ID is counted"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds"
    )
    parser.add_argument(
        "--exclude-zone",
        default=os.getenv("EVAL_EXCLUDE_ZONE", ""),
        help="Polygon to ignore (empty: none)"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Also save the annotated frames as annotated.mp4"
    )

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.source:
        raise SystemExit("No source: pass --source or set CAMERA_URL")

    model = load_model()
    image_size = int(os.getenv("YOLO_IMGSZ", "960"))
    device = os.getenv("YOLO_DEVICE", "cpu")
    zone = parse_zone(args.exclude_zone)

    source = VideoSource(args.source)
    step = max(1, round(source.fps / args.fps))
    print(f"Stream at {source.fps:.1f} fps, processing 1 of every {step} frames "
          f"({source.fps / step:.2f} fps)")

    run_start = datetime.now()
    out_dir = RESULTS_DIR / f"{args.camera_id}_{run_start:%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    counter = UniqueCounter(args.min_hits)
    rows: list[dict] = []
    window: dict | None = None
    window_end = run_start
    frames_read = 0
    video: cv2.VideoWriter | None = None

    with (out_dir / "windows.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        try:
            while True:
                frame = source.read(step - 1)
                if frame is None:
                    break
                frames_read += step

                if source.is_file:
                    now = run_start + timedelta(seconds=frames_read / source.fps)
                else:
                    now = datetime.now()

                if args.duration and (now - run_start).total_seconds() >= args.duration:
                    break

                if window is None or now >= window_end:
                    if window is not None:
                        writer.writerow(window)
                        csv_file.flush()
                        rows.append(window)
                        print(window)
                    window = new_window(now, args.camera_id)
                    window_end = now + timedelta(seconds=args.window)

                # persist=True keeps the tracker state between calls
                result = model.track(
                    frame,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    classes=CLASSES,
                    imgsz=image_size,
                    device=device,
                    verbose=False,
                )[0]
                elapsed = (now - run_start).total_seconds()
                process_result(result, zone, counter, window, elapsed)

                if args.save_video:
                    annotated = result.plot()
                    if video is None:
                        height, width = annotated.shape[:2]
                        video = cv2.VideoWriter(
                            str(out_dir / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            source.fps / step,
                            (width, height),
                        )
                    video.write(annotated)

        except KeyboardInterrupt:
            print("Stopped by the user")
        finally:
            # The last window is written even if it is incomplete
            if window is not None:
                writer.writerow(window)
                rows.append(window)
            source.release()
            if video is not None:
                video.release()

    if rows:
        plot_windows(rows, out_dir / "windows.png")
        total = lambda field: sum(row[field] for row in rows)
        print(f"Unique people: {total('people_unique')} "
              f"(reassociated: {total('people_reassociated')}, "
              f"per-frame sum: {total('people_detections')})")
        print(f"Unique vehicles: {total('vehicles_unique')} "
              f"(reassociated: {total('vehicles_reassociated')}, "
              f"per-frame sum: {total('vehicles_detections')})")
    print(f"Reconnections: {source.reconnects}. Results in {out_dir}")


if __name__ == "__main__":
    main()
