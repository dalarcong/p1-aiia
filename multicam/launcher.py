"""
Multi-camera launcher (task 4.1)

Starts one thread per camera. Each thread captures one frame every
INTERNAL seconds from its own webcam and sends it to the VM server.
All cameras log to a single CSV with the client format
(results/multicam/<run>/requests.csv), so the same analysis applies.

Cameras are read from multicam/cameras.txt, one "<camera_id> <page_url>"
per line. --cameras N uses the first N.

Run from the project root:
    python -m multicam.launcher --cameras 4 --duration 300
"""
from __future__ import annotations

import argparse
import csv
import os
import itertools
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from common.capture import CaptureError, FrameSource
from vm.vm_client import CSV_FIELDS, send_frame

CAMERAS_FILE = Path("multicam/cameras.txt")
RESULTS_DIR = Path("results/multicam")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send frames from N cameras to the VM")
    parser.add_argument(
        "--cameras",
        type=int,
        required=True,
        help="Number of cameras (the first N of the list)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Seconds to run (default: until Ctrl+C)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between frames of each camera (default: CAPTURE_INTERVAL or 5)"
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Folder with recorded frames (<folder>/<camera_id>/*.jpg) to send instead of capturing",
    )

    return parser.parse_args()


def load_cameras(n: int | None = None) -> list[tuple[str, str]]:
    """Return the first n (camera_id, page_url) pairs of the cameras file (all if None)."""
    lines = [
        line.split()
        for line in CAMERAS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if n is None:
        n = len(lines)

    if not 1 <= n <= len(lines):
        raise SystemExit(f"--cameras must be between 1 and {len(lines)} ({CAMERAS_FILE})")

    return [(camera_id, url) for camera_id, url in lines[:n]]


class CsvLog:
    """CSV file shared by all camera threads. A lock keeps rows from mixing."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class ReplaySource:
    """
    Recorded frames of one camera, returned one after another in a loop.

    Same capture() interface as FrameSource.
    """
    def __init__(self, frames_dir: Path) -> None:
        paths = sorted(frames_dir.glob("*.jpg"))
        if not paths:
            raise SystemExit(f"No recorded frames in {frames_dir}")
        self._frames = itertools.cycle([path.read_bytes() for path in paths])

    def capture(self) -> bytes:
        return next(self._frames)


def _camera_loop(
    camera_id: str,
    source: FrameSource | ReplaySource,
    vm_url: str,
    token: str,
    interval: float,
    start_delay: float,
    stop: threading.Event,
    log: CsvLog,
) -> None:
    """Capture and send one frame every `interval` seconds until `stop` is set."""
    session = requests.Session()

    # Staggered start: cameras send at different moments within the interval,
    # as independent cameras would, instead of all at the same time
    if stop.wait(start_delay):
        return

    while not stop.is_set():
        start = time.monotonic()
        now = datetime.now().astimezone()
        row = {"timestamp": now.isoformat(timespec="seconds"), "camera_id": camera_id}

        try:
            jpeg = source.capture()
            row["frame_bytes"] = len(jpeg)
            row["capture_ms"] = round((time.monotonic() - start) * 1000, 1)

            sent = time.perf_counter()
            result = send_frame(session, vm_url, token, camera_id, jpeg, now)
            row["latency_ms"] = round((time.perf_counter() - sent) * 1000, 1)
            row["inference_ms"] = result["inference_ms"]
            row["n_people"] = result["n_people"]
            row["status"] = "ok"

            print(
                f"[{now:%H:%M:%S}] {camera_id:<26} people={row['n_people']:<3} "
                f"latency={row['latency_ms']:.0f} ms inference={row['inference_ms']:.0f} ms"
            )
        except CaptureError as e:
            row["status"] = "capture_error"
            print(f"[{now:%H:%M:%S}] {camera_id:<26} [CAPTURE ERROR] {e}")
        except (requests.RequestException, KeyError) as e:
            row["status"] = "request_error"
            print(f"[{now:%H:%M:%S}] {camera_id:<26} [REQUEST ERROR] {e}")

        log.write(row)
        stop.wait(max(0.0, interval - (time.monotonic() - start)))


def run_cameras(
    n_cameras: int,
    duration: float | None,
    interval: float,
    vm_url: str,
    token: str,
    replay_dir: Path | None = None
) -> tuple[Path, bool]:
    """
    Send frames from the first n_cameras cameras for `duration` seconds.

    Frames are captured live from each webcam or, with replay_dir, taken from
    the frames recorded by multicam/record.py.
    With duration None, run until Ctrl+C. Returns the run folder and whether
    the run was interrupted with Ctrl+C, so a caller such as the sweep can
    stop too.
    """
    cameras = load_cameras(n_cameras)

    # Built before starting the threads, so missing recordings stop the run at once
    sources = [
        ReplaySource(replay_dir / camera_id) if replay_dir else FrameSource(url)
        for camera_id, url in cameras
    ]

    run_dir = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{len(cameras)}cams"
    run_dir.mkdir(parents=True)
    log = CsvLog(run_dir / "requests.csv")
    stop = threading.Event()

    threads = [
        threading.Thread(
            target=_camera_loop,
            args=(camera_id, source, vm_url, token, interval, i * interval / len(cameras), stop, log),
            name=camera_id,
        )
        for i, ((camera_id, _), source) in enumerate(zip(cameras, sources))
    ]

    mode = f"replaying frames from {replay_dir}" if replay_dir else "live capture"
    print(f"{len(cameras)} cameras ({mode}), one frame every {interval}s each, to {vm_url}")
    print(f"Logging requests in {run_dir / 'requests.csv'} (Ctrl+C to stop)")
    for thread in threads:
        thread.start()

    interrupted = False
    try:
        if duration is None:
            while True:
                time.sleep(1)
        else:
            time.sleep(duration)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Each thread finishes its current request before stopping
        print("\nStopping...")
        stop.set()
        for thread in threads:
            thread.join()
        log.close()

    print(f"Results written to {run_dir / 'requests.csv'}")
    return run_dir, interrupted


def main() -> None:
    args = _arguments()
    load_dotenv()
    vm_url = os.environ["VM_URL"].rstrip("/")
    token = os.environ["VM_TOKEN"]
    interval = args.interval or float(os.getenv("CAPTURE_INTERVAL", "5"))
    run_cameras(args.cameras, args.duration, interval, vm_url, token, args.replay)


if __name__ == "__main__":
    main()
