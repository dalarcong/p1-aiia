"""
Loop for uploading cotinuously images to the container.

    python -m serverless.upload_loop
"""

import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from common.capture import FrameSource
from serverless.blob_uploader import upload_frame


def main():
    load_dotenv()

    camera_url = os.environ["CAMERA_URL"]
    camera_id = os.getenv("CAMERA_ID", "camera-1")
    interval = float(
        os.getenv("SERVERLESS_CAPTURE_INTERVAL", "0.5")
    )

    source = FrameSource(camera_url)

    print(
        f"Uploading one frame every {interval}s "
        f"from camera '{camera_id}'"
    )

    while True:
        cycle_start = time.monotonic()

        try:
            jpeg = source.capture()
            captured_at = datetime.now(timezone.utc)

            blob_name = upload_frame(
                jpeg,
                camera_id=camera_id,
                captured_at=captured_at,
            )

            print(
                f"Uploaded {blob_name} "
                f"({len(jpeg) / 1024:.0f} KB)"
            )

        except Exception as exc:
            print(f"[ERROR] {exc}")

        elapsed = time.monotonic() - cycle_start
        time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    main()