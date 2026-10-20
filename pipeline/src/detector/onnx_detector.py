"""
src/detector/onnx_detector.py
Lightweight CPU-optimized ONNX Runtime plate detector.
Designed for low-power edge gateways and CPU appliances.
"""
import logging
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from src.detector.base_detector import BaseDetector, crop_plate_with_padding
from src.pipeline.types import BoundingBox

logger = logging.getLogger("ONNXDetector")


class ONNXDetector(BaseDetector):
    """
    ONNX Runtime plate detector with zero heavy deep-learning framework dependencies.
    """
    def __init__(
        self,
        model_path: str = "models/thai_plate_detector.onnx",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        padding_ratio: float = 0.06,
        input_size: Tuple[int, int] = (640, 640),
    ):
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.padding_ratio = padding_ratio
        self.input_size = input_size
        self.session = None

        self._load_session()

    def _load_session(self):
        try:
            import onnxruntime as ort
            path = Path(self.model_path)
            if not path.exists():
                logger.info(f"ONNX model file '{self.model_path}' not present. Standing by for export.")
                return

            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(str(path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ONNX detector session active ({providers[0]}).")
        except Exception as e:
            logger.warning(f"Could not initialize ONNX detector: {e}")
            self.session = None

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        tw, th = self.input_size
        scale = min(tw / w, th / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        pad_x = (tw - nw) // 2
        pad_y = (th - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (tw, th), swapRB=True, crop=False)
        return blob, scale, (pad_x, pad_y)

    def detect(self, frame: np.ndarray) -> List[Tuple[BoundingBox, np.ndarray]]:
        if self.session is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        blob, scale, (pad_x, pad_y) = self._preprocess(frame)

        try:
            outputs = self.session.run(None, {self.input_name: blob})
            preds = outputs[0]  # shape: (1, 5+classes, num_anchors) or similar
            if preds.ndim == 3 and preds.shape[1] < preds.shape[2]:
                preds = np.transpose(preds[0], (1, 0))
            else:
                preds = preds[0]

            boxes = []
            scores = []

            for row in preds:
                cx, cy, bw, bh = row[:4]
                conf = float(np.max(row[4:])) if len(row) > 4 else float(row[4])
                if conf >= self.conf_threshold:
                    x1 = int(round((cx - bw / 2 - pad_x) / scale))
                    y1 = int(round((cy - bh / 2 - pad_y) / scale))
                    bw_px = int(round(bw / scale))
                    bh_px = int(round(bh / scale))

                    boxes.append([x1, y1, bw_px, bh_px])
                    scores.append(conf)

            idxs = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
            results = []

            if len(idxs) > 0:
                for idx in idxs.flatten():
                    bx, by, bw_px, bh_px = boxes[idx]
                    x1 = max(0, bx)
                    y1 = max(0, by)
                    x2 = min(w, bx + bw_px)
                    y2 = min(h, by + bh_px)

                    if x2 > x1 and y2 > y1:
                        bbox = BoundingBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2, confidence=scores[idx], label="plate")
                        plate_crop = crop_plate_with_padding(frame, bbox, self.padding_ratio)
                        results.append((bbox, plate_crop))

            return results
        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            return []
