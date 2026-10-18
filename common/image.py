"""
JPEG/OpenCV conversion helpers.
"""
from __future__ import annotations

import cv2
import numpy as np


class ImageDecodeError(Exception):
    """Raised when JPEG bytes cannot be decoded into an image."""


class ImageEncodeError(Exception):
    """Raised when an OpenCV image cannot be encoded as JPEG."""


def decode_jpeg(img: bytes) -> np.ndarray:
    """Decode JPEG bytes into an OpenCV BGR image."""
    array = np.frombuffer(img, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if frame is None:
        raise ImageDecodeError("Could not decode JPEG image")

    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    """Encode an OpenCV BGR image as JPEG bytes."""
    if not 0 <= quality <= 100:
        raise ValueError("JPEG quality must be between 0 and 100")

    ok, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not ok:
        raise ImageEncodeError("Could not encode JPEG image")

    return buffer.tobytes()


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """
    Shrink a frame to the given width, keeping its aspect ratio.
    """
    height, original_width = frame.shape[:2]
    if width >= original_width:
        return frame

    new_height = round(height * width / original_width)
    return cv2.resize(frame, (width, new_height), interpolation=cv2.INTER_AREA)