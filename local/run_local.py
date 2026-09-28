"""
Task 1 full pipeline

Captures one frame at the configured interval, counts people with YOLOv8,
logs the common metrics CSV and writes the latest annotated frame.

Run from the project root:
    python -m local.run_local

For a short smoke test:
    python -m local.run_local --max-frames 3
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from common.annotate import annotate_people
from common.capture import CaptureError, capture_frame, resolve_stream_url
from common.count_people import DetectionError, count_people, load_model
from common.results import append_result

ARCHITERCTURE = "local"

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local YOLOv8 person counting")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N successfully processed frames"
    )
    return parser.parse_args()


def _atomic_write(path: Path, data: bytes) -> None:
    """
    Replace a generated file atomically so readers never see a partial JPEG.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def main() -> None:
    args = _arguments()
    load_dotenv()

    camera_url = os.getenv("CAMERA_URL")
    if not camera_url:
        raise SystemExit("CAMERA_URL is not configured in .env")

    camera_id = os.getenv("CAMERA_ID", "camera-1")
    interval = float(os.getenv("CAPTURE_INTERVAL", "5"))
    results_csv = Path(os.getenv("RESULTS_CSV", "results/detections.csv"))
    annotated_frame = Path(
        os.getenv("ANNOTATED_FRAME", "results/processed_frame_local.jpg")
    )

    if interval <= 0:
        raise SystemExit("CAPTURE_INTERVAL must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be positive")

    print("Loading YOLOv8 model...")
    load_model()
    print(f"Camera={camera_id} | interval={interval}s | CSV={results_csv}")

    stream_url: str | None = None
    processed = 0

    try:
        while args.max_frames is None or processed < args.max_frames:
            cycle_start = time.monotonic()

            try:
                if stream_url is None:
                    stream_url = resolve_stream_url(camera_url)
                    print("[INFO] Stream URL resolved")

                jpeg = capture_frame(stream_url)
                timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

                """
                For later architecture comparisons latency starts when the frame
                is already available to the processing architecture. Capture time
                is therefore excluded from this metric
                """
                inference_start = time.monotonic()
                n_people, boxes = count_people(jpeg)
                latency_ms = (time.monotonic() - inference_start) * 1000.0

                append_result(
                    results_csv,
                    timestamp=timestamp,
                    camera_id=camera_id,
                    architecture=ARCHITERCTURE,
                    n_people=n_people,
                    latency_ms=latency_ms,
                    size_bytes=len(jpeg)
                )

                annotated = annotate_people(jpeg, boxes)
                _atomic_write(annotated_frame, annotated)

                processed += 1
                total_s = time.monotonic() - cycle_start

                print(
                    f"[{timestamp}] people={n_people}"
                    f"latency={latency_ms:.0f} ms size={len(jpeg) / 1024:.0f} KB"
                    f"cycle={total_s:.2f} s"
                )

            except CaptureError as exc:
                print(f"[CAPTURE ERROR] {exc}")
                # Skyline tokens expire. Force re-resolution on the next cycle.
                stream_url = None
            except DetectionError as exc:
                print(f"[DETECTION ERROR] {exc}")

            elapsed = time.monotonic() - cycle_start
            if elapsed > interval:
                print(
                    f"[WARN] Processing took {elapsed:.2f}s, longer than the "
                    f"configured {interval:.2f}s interval"
                )
            time.sleep(max(0.0, interval - elapsed))

    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()