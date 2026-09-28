"""
Utilities to render detection boxes on JPEG images.
"""
from __future__ import annotations

import cv2

from common.image import decode_jpeg, encode_jpeg
from common.types import BoundingBox

def annotate_people(
        img: bytes, 
        boxes: list[BoundingBox], 
        quality: int = 90
) -> bytes:
    """
    Draw boxes and confidence scores and return JPEG bytes
    """
    frame = decode_jpeg(img)
    height, width = frame.shape[:2]

    for box in boxes:
        x1 = max(0, min( width - 1, box["x1"]))
        y1 = max(0, min(height - 1, box["y1"]))
        x2 = max(0, min( width - 1, box["x2"]))
        y2 = max(0, min(height - 1, box["y2"]))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"Person {box['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return encode_jpeg(frame, quality=quality)