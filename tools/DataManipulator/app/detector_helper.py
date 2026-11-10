"""
Detector Helper Model Management for YOLO Object Detection (Vehicles & Plates).
Supports listing, switching, and running inference with multiple YOLO detector models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from PIL import Image

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class DetectorHelper:
    _instance: Optional["DetectorHelper"] = None

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model = None
        self.model_path: Optional[Path] = None
        self.class_names: Dict[int, str] = {}
        self.error_msg: Optional[str] = None
        
        resolved = self._resolve_model_path(model_path)
        if resolved and resolved.exists():
            self.switch_model(resolved)
        else:
            self.error_msg = f"No detector weights found."

    @classmethod
    def get_instance(cls, model_path: Optional[Union[str, Path]] = None) -> "DetectorHelper":
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """Discovers available YOLO detector models in models/detector/."""
        app_dir = Path(__file__).resolve().parent
        root = app_dir.parents[2]
        models: Dict[str, str] = {}

        det_root = root / "models" / "detector"
        if det_root.is_dir():
            for pt in det_root.rglob("*.pt"):
                rel = pt.relative_to(det_root)
                label = f"Detector: {str(rel).replace(chr(92), '/')}"
                models[label] = str(pt.resolve())

        # Also check runs/train if any exists
        runs_train = root / "runs" / "train"
        if runs_train.is_dir():
            for pt in runs_train.rglob("best.pt"):
                rel = pt.relative_to(runs_train)
                label = f"Detector: Trained ({str(rel).replace(chr(92), '/')})"
                models[label] = str(pt.resolve())

        return models

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        if explicit_path:
            p = Path(explicit_path).resolve()
            if p.is_file():
                return p

        app_dir = Path(__file__).resolve().parent
        root = app_dir.parents[2]
        candidates = [
            root / "models" / "detector" / "front_rear_vehicle_lpr_best.pt",
            root / "models" / "detector" / "front_rear_lpr_yolov11n.pt",
            root / "models" / "detector" / "vehicle_plate_detector.pt",
            root / "models" / "detector" / "rear_front_best.pt",
            root / "models" / "detector" / "yolov8s.pt",
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        return None

    def switch_model(self, model_path: Union[str, Path]) -> bool:
        """Switches active detector model to given .pt weights."""
        if not HAS_YOLO:
            self.error_msg = "Ultralytics YOLO not installed in environment."
            return False

        p = Path(model_path).resolve()
        if not p.is_file():
            self.error_msg = f"Detector checkpoint not found: {p}"
            return False

        try:
            self.model = YOLO(str(p))
            self.model_path = p
            self.class_names = getattr(self.model, "names", {})
            if isinstance(self.class_names, list):
                self.class_names = {i: n for i, n in enumerate(self.class_names)}
            self.error_msg = None
            return True
        except Exception as e:
            self.model = None
            self.error_msg = f"Failed to load detector {p.name}: {e}"
            return False

    def predict(self, image_input: Union[str, Path, Image.Image], conf: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs YOLO detection on input image.
        Returns list of bounding boxes:
        [
            {
                "cls_id": int,
                "cls_name": str,
                "bbox_norm": (cx, cy, w, h),
                "conf": float
            }, ...
        ]
        """
        if self.model is None:
            return []

        try:
            res = self.model(image_input, conf=conf, verbose=False)[0]
            boxes = []
            for b in res.boxes:
                cls_id = int(b.cls[0])
                cname = self.class_names.get(cls_id, str(cls_id))
                confidence = float(b.conf[0])
                xywhn = b.xywhn[0].tolist()  # [cx, cy, w, h] normalized
                boxes.append({
                    "cls_id": cls_id,
                    "cls_name": cname,
                    "bbox_norm": (round(xywhn[0], 4), round(xywhn[1], 4), round(xywhn[2], 4), round(xywhn[3], 4)),
                    "conf": round(confidence, 4)
                })
            return boxes
        except Exception as e:
            self.error_msg = str(e)
            return []
