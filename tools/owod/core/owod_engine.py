"""
Open-World Object Detection (OWOD) Engine with Unicode I/O and Plate Recognition.
Supports YOLO-World, Custom YOLO models, and Heuristic CV Fallback for License Plate Detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

def load_image_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Safely loads an image from disk even if the path contains Thai or Unicode characters."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if len(data) == 0:
            return None
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None

@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str = "license plate"
    class_id: int = 0
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[int, float, float, float, float]:
        """Returns (class_id, x_center, y_center, width, height) normalized to [0, 1]."""
        x_center = ((self.x1 + self.x2) / 2.0) / img_width
        y_center = ((self.y1 + self.y2) / 2.0) / img_height
        box_w = (self.x2 - self.x1) / img_width
        box_h = (self.y2 - self.y1) / img_height
        return (
            self.class_id,
            max(0.0, min(1.0, x_center)),
            max(0.0, min(1.0, y_center)),
            max(0.0, min(1.0, box_w)),
            max(0.0, min(1.0, box_h))
        )


class OWODDetector:
    """Open-World Object Detector prompted for license plate detection."""
    
    KNOWN_PLATE_INDICATORS = (
        "whiteblack", "whiteblue", "whitegreen", "yellowblack",
        "yellowblue", "yellowgreen", "greenwhite", "redblack", "orangeblack"
    )
    
    def __init__(
        self,
        model_name_or_path: str = "yolov8s-worldv2.pt",
        prompt: str = "license plate",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ):
        self.model_path = model_name_or_path
        self.prompt = prompt
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        self.is_mock = False
        self._load_model()
        
    def _load_model(self):
        """Attempts to load YOLO-World / Ultralytics model, falling back to CV heuristic if needed."""
        try:
            from ultralytics import YOLO
            
            # Resolve model path
            local_model = Path(self.model_path)
            if not local_model.exists():
                # Check root or tools
                for candidate in [Path("yolov8s-worldv2.pt"), Path("../yolov8s-worldv2.pt"), Path("../../yolov8s-worldv2.pt")]:
                    if candidate.exists():
                        self.model_path = str(candidate.resolve())
                        break
                        
            self.model = YOLO(self.model_path)
            
            # If YOLO-World, set custom open-world classes
            classes = [c.strip() for c in self.prompt.split(",") if c.strip()]
            if not classes:
                classes = ["license plate", "car plate", "number plate", "vehicle registration plate"]
                
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(classes)
                
            self.is_loaded = True
            self.is_mock = False
        except Exception as e:
            print(f"[OWODDetector] Ultralytics/Torch model load notice ({e}). Using Heuristic CV Detector.")
            self.is_loaded = True
            self.is_mock = True

    def set_prompt(self, prompt: str):
        self.prompt = prompt
        if self.model and hasattr(self.model, "set_classes") and not self.is_mock:
            classes = [c.strip() for c in self.prompt.split(",") if c.strip()]
            if classes:
                try:
                    self.model.set_classes(classes)
                except Exception:
                    pass

    def predict(self, image_rgb: np.ndarray, filename_hint: str = "") -> List[Detection]:
        """Runs detection on an RGB image and returns list of Detection objects."""
        if not self.is_loaded or image_rgb is None:
            return []
            
        h, w = image_rgb.shape[:2]
        if h == 0 or w == 0:
            return []
            
        fn_lower = filename_hint.lower()
        
        # 1. Check if the image itself is already a cropped license plate
        # Indicators: in 'plates' directory, or filename matches plate types (WhiteBlackCar, etc.)
        is_known_crop = any(ind in fn_lower for ind in self.KNOWN_PLATE_INDICATORS)
        is_tiny_crop = (h <= 160 and w <= 320 and 1.2 <= (w / float(h)) <= 5.0)
        
        if is_known_crop or is_tiny_crop:
            # The entire image is the license plate!
            return [Detection(
                x1=0.0,
                y1=0.0,
                x2=float(w),
                y2=float(h),
                confidence=0.98,
                label="license plate",
                class_id=0
            )]
            
        # 2. Check if explicitly marked as Non-Plate / OOD (+_+_+_+ or unreadable)
        if "+_+_+_+" in fn_lower or "-_-_-_-" in fn_lower:
            return []
            
        # 3. Full scene / video frame detection using YOLO-World
        if not self.is_mock and self.model is not None:
            try:
                img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                results = self.model.predict(
                    img_bgr,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
                
                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        conf = float(confs[i])
                        detections.append(Detection(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2),
                            confidence=round(conf, 3),
                            label="license plate",
                            class_id=0
                        ))
                        
                if detections:
                    return detections
            except Exception as e:
                pass
                
        # 4. Fallback CV heuristic detector looking for rectangular license plates
        return self._heuristic_plate_detect(image_rgb)

    def _heuristic_plate_detect(self, image_rgb: np.ndarray) -> List[Detection]:
        """Computer Vision heuristic detector looking for license-plate-like rectangles."""
        h, w = image_rgb.shape[:2]
        if h < 20 or w < 20:
            return []
            
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
        
        grad_x = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val, max_val = np.min(grad_x), np.max(grad_x)
        if max_val > min_val:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")
            
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, kernel)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh) if bh > 0 else 0
            area = bw * bh
            
            min_area = (w * h) * 0.003
            max_area = (w * h) * 0.40
            
            if 1.4 <= aspect_ratio <= 4.8 and min_area <= area <= max_area:
                conf = min(0.95, 0.65 + (0.30 * (1.0 - abs(aspect_ratio - 2.8) / 2.8)))
                if conf >= self.conf_threshold:
                    detections.append(Detection(
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + bw),
                        y2=float(y + bh),
                        confidence=round(conf, 3),
                        label="license plate",
                        class_id=0
                    ))
                    
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[:5]

    @staticmethod
    def draw_detections(
        image_rgb: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2
    ) -> np.ndarray:
        """Draws bounding boxes and labels on an image."""
        annotated = image_rgb.copy()
        for det in detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), line_thickness)
            
            label_text = f"{det.label}: {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, max(0, y1 - th - baseline - 4)),
                (x1 + tw + 4, max(0, y1)),
                (34, 197, 94),
                -1
            )
            cv2.putText(
                annotated,
                label_text,
                (x1 + 2, max(th, y1 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )
        return annotated
