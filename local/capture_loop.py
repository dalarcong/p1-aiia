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

from common.capture import CaptureError, capture_frame, resolve_stream_url

FRAMES_DIR = Path("frames")


def main() -> None:
    load_dotenv()
    camera_url = os.environ["CAMERA_URL"]
    interval = float(os.getenv("CAPTURE_INTERVAL", "5"))
    FRAMES_DIR.mkdir(exist_ok=True)

    # Resolved lazily and reset on failure, so an expired token is renewed
    # on the next iteration without reloading the page on every capture.
    stream_url = None

    print(f"Capturing every {interval}s (Ctrl+C to stop)")
    try:
        while True:
            start = time.monotonic()
            now = datetime.now()
            try:
                if stream_url is None:
                    stream_url = resolve_stream_url(camera_url)
                    print(f"[{now:%H:%M:%S}] [INFO] Stream URL resolved")
                try:
                    jpeg = capture_frame(stream_url)
                except CaptureError:
                    # The stream token may have expired: renew it and retry once
                    stream_url = resolve_stream_url(camera_url)
                    print(f"[{now:%H:%M:%S}] [INFO] Stream URL renewed")
                    jpeg = capture_frame(stream_url)
                
                path = FRAMES_DIR / f"{now:%Y%m%d_%H%M%S}.jpg"
                path.write_bytes(jpeg)

                elapsed = time.monotonic() - start
                print(f"[{now:%H:%M:%S}] {path.name} "
                      f"{len(jpeg) / 1024:.0f} KB {elapsed:.2f} s")
            except CaptureError as e:
                print(f"[{now:%H:%M:%S}] [ERROR]: {e}")
                stream_url = None

            # Sleep only the remaining time, so frames are taken every `interval`
            # seconds regardless of how long the capture took.
            time.sleep(max(0.0, interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()