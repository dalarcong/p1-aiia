"""Real Azure VM inference server for task 2.3.

The public client talks to Apache on port 80. Apache reverse-proxies
``POST /frame`` to this FastAPI application, which listens only on
``127.0.0.1:8000``.

Run from the project root (one worker so the 2 GB VM keeps one YOLO copy):

    python3 -m uvicorn vm.server:app --host 127.0.0.1 --port 8000 --workers 1
"""
from __future__ import annotations

import hmac
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, HTTPException

from common.count_people import DetectionError, count_people, load_model
from common.image import ImageDecodeError
from vm.results import append_vm_result


load_dotenv()

_TOKEN = os.getenv("VM_TOKEN", "")
if not _TOKEN:
    raise RuntimeError("VM_TOKEN must be configured and must not be empty")

_RESULTS_ROOT = Path(os.getenv("VM_RESULTS_ROOT", "results/vm"))
_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
_RESULTS_CSV = _RESULTS_ROOT / _RUN_ID / "detections.csv"

# FastAPI executes normal ``def`` endpoints in a thread pool.  Serialising
# access to the shared YOLO instance prevents concurrent inferences from
# competing for the very limited RAM/CPU of the VM.  Queueing time remains
# visible in the client's round-trip latency, while inference_ms measures
# only the actual YOLO call.
_inference_lock = Lock()

app = FastAPI(title="Smart Lighting VM server")


@app.on_event("startup")
def startup() -> None:
    """Load YOLO before accepting the first real frame."""
    print("Loading YOLO model...")
    load_model()
    print(f"YOLO ready. VM results will be stored in {_RESULTS_CSV}")


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

    append_vm_result(
        _RESULTS_CSV,
        captured_at=x_captured_at,
        received_at=received_at,
        camera_id=x_camera_id,
        n_people=n_people,
        inference_ms=inference_ms,
        frame_bytes=len(jpeg),
    )

    print(
        f"[{received_at}] camera={x_camera_id} people={n_people} "
        f"inference={inference_ms:.1f} ms size={len(jpeg) / 1024:.0f} KB"
    )

    return {
        "n_people": n_people,
        "inference_ms": round(inference_ms, 1),
        "boxes": boxes,
    }
