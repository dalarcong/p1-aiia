"""
Fake VM server for testing the client.

Implements the client <-> VM contract without running YOLO: checks the
token, accepts the JPEG and answers with a fixed result. It is also the
starting point for the real server.

Run from the project root:
    uvicorn vm.fake_server:app --host 0.0.0.0 --port 8000
"""
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, FastAPI, Header, HTTPException

load_dotenv()
TOKEN = os.environ["VM_TOKEN"]

app = FastAPI(title="VM server (fake)")

@app.post("/frame")
def receive_frame(
    jpeg: Annotated[bytes, Body(media_type="image/jpeg")] = b"",
    authorization: Annotated[str | None, Header()] = None,
    x_camera_id: Annotated[str, Header()] = "unknown",
):
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="invalid token")
    if not jpeg:
        raise HTTPException(status_code=400, detail="empty body")

    print(f"Received {len(jpeg) / 1024:.0f} KB from {x_camera_id}")
    return {"n_people": 0, "inference_ms": 0.0, "boxes": []}
