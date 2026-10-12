"""Real Azure VM inference server for task 2.3.

The public client talks to Apache on port 80. Apache reverse-proxies
``POST /frame`` to this FastAPI application, which listens only on
``127.0.0.1:8000``.

If VM_WEB_DATA_DIR is set, every processed frame is also published for the
results website: the latest annotated frame and a copy of the current run's
CSV, in a folder served by Apache.

Run from the project root (one worker so the 1 GB VM keeps one YOLO copy):

    python3 -m uvicorn vm.server:app --host 127.0.0.1 --port 8000 --workers 1
"""
from __future__ import annotations

import hmac
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Annotated

import numpy as np
from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, HTTPException

from common.annotate import annotate_people
from common.count_people import DetectionError, count_people, load_model
from common.image import ImageDecodeError, ImageEncodeError, encode_jpeg
from vm.results import append_vm_result


load_dotenv()

_TOKEN = os.getenv("VM_TOKEN", "")
if not _TOKEN:
    raise RuntimeError("VM_TOKEN must be configured and must not be empty")

_RESULTS_ROOT = Path(os.getenv("VM_RESULTS_ROOT", "results/vm"))
_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_RESULTS_CSV = _RESULTS_ROOT / _RUN_ID / "detections.csv"

# Optional folder served by Apache for the website (task 2.4).
_web_data_dir = os.getenv("VM_WEB_DATA_DIR")
_WEB_DATA_DIR = Path(_web_data_dir) if _web_data_dir else None

# FastAPI executes normal ``def`` endpoints in a thread pool.  Serialising
# access to the shared YOLO instance prevents concurrent inferences from
# competing for the very limited RAM/CPU of the VM.  Queueing time remains
# visible in the client's round-trip latency, while inference_ms measures
# only the actual YOLO call.
_inference_lock = threading.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm up YOLO before accepting the first real frame."""
    print("Loading YOLO model...")
    load_model()
    # The first inferences are much slower (one-off setup inside PyTorch), so
    # one is run on a blank image to keep that cost out of the measurements.
    count_people(encode_jpeg(np.zeros((640, 640, 3), dtype=np.uint8)))
    print(f"YOLO ready. VM results will be stored in {_RESULTS_CSV}")

    if _WEB_DATA_DIR is not None:
        _WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # The website shows the current run only, like the run's own CSV
        (_WEB_DATA_DIR / "detections.csv").unlink(missing_ok=True)
        print(f"Publishing website data in {_WEB_DATA_DIR}")

    yield

app = FastAPI(title="Smart Lighting VM server", lifespan=lifespan)


def _write_atomically(path: Path, data: bytes) -> None:
    """
    Write a file so that readers never see it half-written.

    The data goes to a temporary file in the same folder, which then replaces
    the target in one step. The temporary name includes the thread id, so two
    requests publishing at the same time do not share it.
    """
    tmp_path = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def _publish_for_web(jpeg: bytes, boxes: list, result: dict) -> None:
    """
    Publish the latest annotated frame and append the result to the web CSV.

    Errors are logged and swallowed: the website is secondary, so a failure
    here must not turn a correctly counted frame into an error for the client.
    """
    if _WEB_DATA_DIR is None:
        return

    try:
        _write_atomically(_WEB_DATA_DIR / "latest.jpg", annotate_people(jpeg, boxes))
        append_vm_result(_WEB_DATA_DIR / "detections.csv", **result)
    except (OSError, ImageDecodeError, ImageEncodeError) as exc:
        print(f"[WEB ERROR] {exc}")


@app.get("/health")
def health() -> dict[str, str]:
    """Small local health endpoint useful when checking Apache/FastAPI."""
    return {"status": "ok"}


@app.post("/frame")
def receive_frame(
    jpeg: Annotated[bytes, Body(media_type="image/jpeg")] = b"",
    authorization: Annotated[str | None, Header()] = None,
    x_camera_id: Annotated[str, Header()] = "unknown",
    x_captured_at: Annotated[str | None, Header()] = None,
) -> dict:
    """Validate, process and persist one JPEG frame."""
    expected_authorization = f"Bearer {_TOKEN}"
    if authorization is None or not hmac.compare_digest(
        authorization,
        expected_authorization,
    ):
        raise HTTPException(status_code=401, detail="invalid token")

    if not jpeg:
        raise HTTPException(status_code=400, detail="empty body")

    if x_captured_at is None:
        raise HTTPException(status_code=400, detail="missing X-Captured-At header")

    
    # The VM does not use this timestamp to calculate latency. The client does
    # that with its own monotonic clock.  We still validate and persist it so
    # each VM result can be matched to its source frame.
    try:
        datetime.fromisoformat(x_captured_at)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="invalid X-Captured-At header",
        ) from exc

    received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    try:
        with _inference_lock:
            inference_started = time.perf_counter()
            n_people, boxes = count_people(jpeg)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid JPEG") from exc
    except DetectionError as exc:
        # Do not leak a traceback to the public client; the server log keeps
        # the underlying exception for diagnosis.
        print(f"[DETECTION ERROR] {exc}")
        raise HTTPException(status_code=500, detail="inference failed") from exc

    result = {
        "captured_at": x_captured_at,
        "received_at": received_at,
        "camera_id": x_camera_id,
        "n_people": n_people,
        "inference_ms": inference_ms,
        "frame_bytes": len(jpeg),
    }
    append_vm_result(_RESULTS_CSV, **result)
    _publish_for_web(jpeg, boxes, result)

    print(
        f"[{received_at}] camera={x_camera_id} people={n_people} "
        f"inference={inference_ms:.1f} ms size={len(jpeg) / 1024:.0f} KB"
    )

    return {
        "n_people": n_people,
        "inference_ms": round(inference_ms, 1),
        "boxes": boxes,
    }
