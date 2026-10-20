"""
src/detector/base_detector.py
Abstract base class for vehicle and license plate detectors.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
import numpy as np
from src.pipeline.types import BoundingBox


def crop_plate_with_padding(
    frame: np.ndarray,
    bbox: BoundingBox,
    padding_ratio: float = 0.06
) -> np.ndarray:
    """
    Extracts a license plate crop with boundary-clamped margin padding
    to ensure edge characters, borders, and bevels are not clipped.
    """
    h, w = frame.shape[:2]
    bw = bbox.width
    bh = bbox.height

    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))

    x1 = max(0, bbox.xmin - pad_x)
    y1 = max(0, bbox.ymin - pad_y)
    x2 = min(w, bbox.xmax + pad_x)
    y2 = min(h, bbox.ymax + pad_y)

    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else np.zeros((32, 32, 3), dtype=np.uint8)


class BaseDetector(ABC):
    """Abstract interface for all plate detectors in the pipeline."""
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[BoundingBox, np.ndarray]]:
        """
        Pure function:
        Takes an input RGB/BGR frame, returns a list of (BoundingBox, cropped_plate_image).
        """
        pass
