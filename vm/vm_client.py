"""
VM client (task 2.2)

Captures one frame every CAPTURE_INTERVAL seconds from CAMERA_URL, sends it
to the VM server (POST /frame) and logs the result and the timings of every
request in a CSV.

Run from the project root:
    python -m vm.vm_client
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from common.capture import CaptureError, FrameSource

RESULTS_DIR = Path("results/vm_client")

CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = 30

CSV_FIELDS = [
    "timestamp",
    "camera_id",
    "frame_bytes",
    "capture_ms",
    "latency_ms",
    "inference_ms",
    "n_people",
    "status"
]


def send_frame(
    session: requests.Session,
    vm_url: str,
    token: str,
    camera_id: str,
    jpeg: bytes,
    captured_at: datetime
) -> dict:
    """
    Send one frame to the VM server and return its JSON answer.

    Raises:
        requests.RequestException: On network errors, timeouts or a non-2xx
            answer.
    """
    response = session.post(
        f"{vm_url}/frame",
        data=jpeg,
        headers={
            "Content-Type": "image/jpeg",
            "Authorization": f"Bearer {token}",
            "X-Camera-Id": camera_id,
            "X-Captured-At": captured_at.isoformat(timespec="seconds")
        },
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
    )

    response.raise_for_status()
    return response.json()


def main() -> None:
    load_dotenv()
    camera_url = os.environ["CAMERA_URL"]
    interval = float(os.getenv("CAPTURE_INTERVAL", "5"))
    vm_url = os.environ["VM_URL"].rstrip("/")
    token = os.environ["VM_TOKEN"]
    camera_id = os.getenv("CAMERA_ID", "camera")

    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    csv_path = run_dir / "requests.csv"

    source = FrameSource(camera_url)

    # One session for the whole run. The connection to the VM is reused,
    # so latency_ms does not include a new TCP handshake on every frame.
    session = requests.Session()

    print(f"Sending one frame every {interval}s to {vm_url} (Ctrl+C to stop)")
    print(f"Logging requests in {csv_path}")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while True:
                start = time.monotonic()
                now = datetime.now().astimezone()
                row = {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "camera_id": camera_id,
                }
                resolutions_before = source.url_resolutions

                try:
                    jpeg = source.capture()
                    row["frame_bytes"] = len(jpeg)
                    row["capture_ms"] = round((time.monotonic() - start) * 1000, 1)
                    if source.url_resolutions > resolutions_before:
                        print(f"[{now:%H:%M:%S}] [INFO] Stream URL resolved")

                    # Round trip only (upload + inference + answer), without
                    # the capture, which depends on the webcams site
                    sent = time.perf_counter()
                    result = send_frame(session, vm_url, token, camera_id, jpeg, now)
                    row["latency_ms"] = round((time.perf_counter() - sent) * 1000, 1)
                    row["inference_ms"] = result["inference_ms"]
                    row["n_people"] = result["n_people"]
                    row["status"] = "ok"

                except CaptureError as e:
                    row["status"] = "capture_error"
                    print(f"[{now:%H:%M:%S}] [CAPTURE ERROR] {e}")
                except (requests.RequestException, KeyError) as e:
                    row["status"] = "request_error"
                    print(f"[{now:%H:%M:%S}] [REQUEST ERROR] {e}")
                else:
                    print(
                        f"[{now:%H:%M:%S}] people={row['n_people']} "
                        f"{row['frame_bytes'] / 1024:.0f} KB "
                        f"latency={row['latency_ms']:.0f} ms "
                        f"inference={row['inference_ms']:.0f} ms"
                    )

                writer.writerow(row)
                handle.flush()

                # Sleep only the remaining time, so frames are sent every
                # `interval` seconds regardless of how long each one took
                time.sleep(max(0.0, interval - (time.monotonic() - start)))
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    main()
