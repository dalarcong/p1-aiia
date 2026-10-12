"""Local capture loop

Captures one frame every CAPTURE_INTERVAL seconds from CAMERA_URL and
saves it as JPEG in frames/.

Run from the project root:
    python -m local.capture_loop
"""
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from common.capture import CaptureError, FrameSource

FRAMES_DIR = Path("frames")


def main() -> None:
    load_dotenv()
    camera_url = os.environ["CAMERA_URL"]
    interval = float(os.getenv("CAPTURE_INTERVAL", "5"))
    FRAMES_DIR.mkdir(exist_ok=True)

    source = FrameSource(camera_url)

    print(f"Capturing every {interval}s (Ctrl+C to stop)")
    try:
        while True:
            start = time.monotonic()
            now = datetime.now()
            resolutions_before = source.url_resolutions
            try:
                jpeg = source.capture()
                if source.url_resolutions > resolutions_before:
                    print(f"[{now:%H:%M:%S}] [INFO] Stream URL resolved")

                path = FRAMES_DIR / f"{now:%Y%m%d_%H%M%S}.jpg"
                path.write_bytes(jpeg)

                elapsed = time.monotonic() - start
                print(f"[{now:%H:%M:%S}] {path.name} "
                      f"{len(jpeg) / 1024:.0f} KB {elapsed:.2f} s")
            except CaptureError as e:
                print(f"[{now:%H:%M:%S}] [ERROR]: {e}")

            # Sleep only the remaining time, so frames are taken every `interval`
            # seconds regardless of how long the capture took.
            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()