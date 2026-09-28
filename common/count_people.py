"""
Person detection with YOLOv8 (task 1.3)
"""
from __future__ import annotations

import os

from ultralytics import YOLO
from common.image import decode_jpeg
from common.types import BoundingBox

PERSON_CLASS_ID = 0  # COCO (Common Objects in Context) class id for "person"

class DetectionError(Exception):
    """Raised when YOLO configuration, loading or inference fails."""

_model: YOLO | None = None

def load_model() -> YOLO:
    """
    Load the YOLO model once and reuse it across frames.

    This way paying model initialisation cost on every frame is avoided.
    The model selected is the YOLOv8 nano model (6MB).
    """
    global _model

    if _model is None:
        model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")

        try:
            _model = YOLO(model_name)
        except Exception as exc:
            raise DetectionError(
                f"Could not load YOLO model {model_name!r}: {exc}"
            ) from exc
        
    return _model


def _get_prediction_settings() -> tuple[float, int, str]:
    """
    Read and validate inference settings from environment variables.
    """
    try:
        confidence = float(os.getenv("YOLO_CONF", "0.5"))
        image_size = int(os.getenv("YOLO_IMGSZ", "640"))
    except ValueError as exc:
        raise DetectionError(
            "YOLO_CONF and YOLO_IMGSZ must be numeric"
        ) from exc

    device = os.getenv("YOLO_DEVICE", "cpu")

    if not 0.0 <= confidence <= 1.0:
        raise DetectionError("Confidence (YOLO_CONF) must be between 0 and 1")
    if image_size <= 0:
        raise DetectionError("Image size (YOLO_IMGSZ) must be greater than 0")

    return confidence, image_size, device

def count_people(
        img: bytes
) -> tuple[int, list[BoundingBox]]:
    """
    Count people in a JPEG image using YOLOv8

    Args:
        img: JPEG-encoded image bytes. ``capture_frame`` output.
    
    Returns:
        ``(n, boxes)`` where ``n`` is the number of detected people and
        ``boxes`` contains JSON-serialisable ``xyxy`` coordinates plus 
        confidence for each person.

    Raises:
        DetectionError: If the JPEG is invalid, configuration is invalid or
        YOLO inference fails.
    """

    frame = decode_jpeg(img)
    confidence, image_size, device = _get_prediction_settings()

    try:
        result = load_model().predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
    except DetectionError:
        raise
    except Exception as exc:
        raise DetectionError(f"YOLO inference failed: {exc}") from exc

    boxes: list[BoundingBox] = []

    if result.boxes is None:
        return 0, boxes

    coordinates = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    for (x1, y1, x2, y2), score in zip(coordinates, confidences):
        boxes.append(
            {
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2)),
                "confidence": float(score),
            }
        )

    return len(boxes), boxes