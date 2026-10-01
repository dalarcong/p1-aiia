"""
Exclusion zones for person counting.

A person is excluded when the bottom-centre point of its bounding box
(its feet) falls inside the zone.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def parse_zone(text: str) -> np.ndarray | None:
    """
    Parse a polygon written as 'x1,y1;x2,y2' in pixel coordinates.

    Returns:
        The polygon as an OpenCV contour, or None if 'text' is empty.

    Raises:
        ValueError: If the text is not a list of at least 3 points
    """

    text = text.strip()
    if not text:
        return None

    try:
        points =  [
            tuple(int(value) for value in point.split(","))
            for point in text.split(";")
        ]
    except ValueError as exc:
        raise ValueError(f"Invalid zone {text!r}: expected 'x,y;x,y;...'") from exc

    if len(points) < 3 or any(len(point) != 2 for point in points):
        raise ValueError(f"Invalid zone {text!r}: expected at least 3 x,y points")

    return np.array(points, dtype=np.int32)


def load_exclude_zone() -> np.ndarray | None:
    """Read the exclusion polygon from the EVAL_EXCLUDE_ZONE env variable."""
    return parse_zone(os.getenv("EVAL_EXCLUDE_ZONE", ""))


def is_excluded(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    zone: np.ndarray | None
) -> bool:
    """Return True if the feet of the box are inside the zone."""
    if zone is None:
        return False

    feet = ((x1 + x2) / 2, float(y2))
    return cv2.pointPolygonTest(zone, feet, False) >= 0