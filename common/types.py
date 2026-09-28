"""
Shared data types used by the detection pipeline.
"""
from typing import TypedDict


class BoundingBox(TypedDict):
    """
    Bounding box of one detectection, in original-image coordinates.
    """
    
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float