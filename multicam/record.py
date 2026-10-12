"""
Record frames for the replay mode (task 4.2).

Captures --frames frames from each camera of multicam/cameras.txt, one
camera at a time and one frame every --interval seconds, and saves them as
<out>/<camera_id>/NN.jpg. Recording one camera at a time avoids the capture
bottleneck seen when many live cameras are captured at once.

Run from the project root:
    python -m multicam.record --frames 12 --out replay_frames
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common.capture import CaptureError, FrameSource
from multicam.launcher import load_cameras

# Consecutive capture errors after which a camera is skipped
MAX_FAILURES = 3

def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record frames of each camera for the replay mode")

    parser.add_argument(
        "--frames", 
        type=int, 
        default=12, 
        help="Frames per camera"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=5, 
        help="Seconds between frames"
    )
    parser.add_argument(
        "--out", 
        type=Path, 
        default=Path("replay_frames"), 
        help="Output folder"
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=None,
        help="Record only the first N cameras (default: all)",
    )

    return parser.parse_args()


def _record_camera(camera_id: str, url: str, out_dir: Path, frames: int, interval: float) -> int:
    """Save `frames` frames of one camera; return how many were saved."""
    out_dir.mkdir(parents=True, exist_ok=True)
    source = FrameSource(url)
    saved = failures = 0

    while saved < frames and failures < MAX_FAILURES:
        start = time.monotonic()
        try:
            jpeg = source.capture()
            (out_dir / f"{saved:02d}.jpg").write_bytes(jpeg)
            saved += 1
            failures = 0
            print(f"  {camera_id}: frame {saved}/{frames} ({len(jpeg) / 1024:.0f} KB)")
        except CaptureError as e:
            failures += 1
            print(f"  {camera_id}: [CAPTURE ERROR] {e}")

        if saved < frames:
            time.sleep(max(0.0, interval - (time.monotonic() - start)))

    return saved


def main() -> None:
    args = _arguments()
    cameras = load_cameras(args.cameras)
    print(f"Recording {args.frames} frames of {len(cameras)} cameras into {args.out}")

    incomplete = []
    for camera_id, url in cameras:
        saved = _record_camera(camera_id, url, args.out / camera_id, args.frames, args.interval)
        if saved < args.frames:
            incomplete.append(f"{camera_id} ({saved}/{args.frames})")

    if incomplete:
        print(f"\nIncomplete cameras: {', '.join(incomplete)}")
    print(f"Frames written to {args.out}")


if __name__ == "__main__":
    main()