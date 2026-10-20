"""
src/detector/yolo_detector.py
Ultralytics YOLO-World and YOLOv8 plate detector.
Pure input frame -> returns list of (BoundingBox, cropped_plate).
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from src.detector.base_detector import BaseDetector, crop_plate_with_padding
from src.pipeline.types import BoundingBox

logger = logging.getLogger("YOLODetector")


class YOLODetector(BaseDetector):
    """
    High-throughput license plate detector supporting YOLOv8 and zero-shot YOLO-World.
    """
    def __init__(
        self,
        model_path: str = "models/yolov8s-worldv2.pt",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        padding_ratio: float = 0.06,
        target_classes: Optional[List[str]] = None,
        device: str = "auto",
        half_precision: bool = True,
    ):
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.padding_ratio = padding_ratio
        self.target_classes = target_classes or ["license plate", "vehicle registration plate", "number plate"]
        self.device = device
        self.half_precision = half_precision
        self.model = None

        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            import torch

            # Resolve device
            if self.device == "auto":
                resolved_dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                resolved_dev = self.device

            self.resolved_device = resolved_dev
            logger.info(f"Loading YOLO detector: '{self.model_path}' onto {self.resolved_device}...")

            self.model = YOLO(self.model_path)

            # Set custom vocabulary if YOLO-World model
            if "world" in str(self.model_path).lower() and hasattr(self.model, "set_classes"):
                try:
                    self.model.set_classes(self.target_classes)
                    logger.info(f"Zero-shot classes initialized: {self.target_classes}")
                except Exception as e:
                    logger.warning(f"Could not set YOLO-World classes: {e}")

            logger.info("YOLO detector ready.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{self.model_path}': {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Tuple[BoundingBox, np.ndarray]]:
        """
        Pure detection function:
        Takes an input frame, runs neural inference, and returns list of (BoundingBox, plate_crop).
        """
        if self.model is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        results = []

        try:
            preds = self.model.predict(
                source=frame,
                device=self.resolved_device,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                verbose=False,
            )

            if not preds or len(preds) == 0 or preds[0].boxes is None:
                return []

            for box in preds[0].boxes:
                coords = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item()) if hasattr(box, "cls") else 0
                label = self.model.names.get(cls_id, "plate") if hasattr(self.model, "names") else "plate"

                x1 = max(0, int(round(coords[0])))
                y1 = max(0, int(round(coords[1])))
                x2 = min(w, int(round(coords[2])))
                y2 = min(h, int(round(coords[3])))

                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2, confidence=conf, label=label)
                plate_crop = crop_plate_with_padding(frame, bbox, self.padding_ratio)

                results.append((bbox, plate_crop))

        except Exception as e:
            logger.error(f"Error during YOLO plate inference: {e}")

        return results
