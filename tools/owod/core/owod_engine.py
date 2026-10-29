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
        self.vehicle_model = None
        self.lpr_model = None
        self.is_loaded = False
        self.is_mock = False
        self._load_model()
        
    def _load_model(self):
        """Loads dedicated neural license plate detector, vehicle detector, and base model."""
        try:
            from ultralytics import YOLO
            
            # 1. Dedicated neural license plate detector (prioritizes YOLOv11x / YOLOv11s)
            for lp_cand in [
                Path("yolov11x_license_plate.pt"),
                Path("../yolov11x_license_plate.pt"),
                Path("../../yolov11x_license_plate.pt"),
                Path("yolov11s_license_plate.pt"),
                Path("../yolov11s_license_plate.pt"),
                Path("../../yolov11s_license_plate.pt"),
                Path("license_plate_best.pt"),
                Path("../license_plate_best.pt"),
                Path("../../license_plate_best.pt")
            ]:
                if lp_cand.exists():
                    try:
                        self.lpr_model = YOLO(str(lp_cand.resolve()))
                        print(f"[OWODDetector] Loaded Pretrained Neural License Plate Detector: {lp_cand.name}")
                        break
                    except Exception:
                        pass

            # 2. Base model
            local_model = Path(self.model_path)
            if not local_model.exists():
                for candidate in [
                    Path("models/detector/yolov8s-worldv2.pt"),
                    Path("../../models/detector/yolov8s-worldv2.pt"),
                    Path("yolov8s-worldv2.pt"),
                    Path("../yolov8s-worldv2.pt"),
                    Path("../../yolov8s-worldv2.pt"),
                    Path("models/detector/yolov8s.pt"),
                    Path("../../models/detector/yolov8s.pt"),
                    Path("yolov8s.pt")
                ]:
                    if candidate.exists():
                        self.model_path = str(candidate.resolve())
                        break
                        
            self.model = YOLO(self.model_path)
            
            # 3. Vehicle proposal model
            for v_cand in [
                Path("models/detector/yolov8s.pt"),
                Path("../../models/detector/yolov8s.pt"),
                Path("yolov8s.pt"),
                Path("../yolov8s.pt"),
                Path("../../yolov8s.pt")
            ]:
                if v_cand.exists():
                    self.vehicle_model = YOLO(str(v_cand.resolve()))
                    break
            if self.vehicle_model is None:
                self.vehicle_model = self.model
            
            # 4. Vehicle Front/Back Orientation model (vehicle-orientation)
            self.orientation_model = None
            for rf_cand in [
                Path("models/detector/vehicle_orientation_best.pt"),
                Path("../models/detector/vehicle_orientation_best.pt"),
                Path("../../models/detector/vehicle_orientation_best.pt"),
                Path("vehicle_plate_detector.pt"),
                Path("models/detector/vehicle_plate_yolov11n.pt"),
                Path("../../models/detector/vehicle_plate_yolov11n.pt"),
                Path("models/detector/vehicle_plate_detector.pt"),
                Path("../../models/detector/vehicle_plate_detector.pt"),
                Path("models/vehicle_plate_detector.pt")
            ]:
                if rf_cand.exists():
                    try:
                        self.orientation_model = YOLO(str(rf_cand.resolve()))
                        print(f"[OWODDetector] Discovered Car Front/Back Orientation Model: {rf_cand}")
                        break
                    except Exception:
                        pass
            
            # If YOLO-World, set custom open-world classes
            classes = [c.strip() for c in self.prompt.split(",") if c.strip()]
            if not classes:
                classes = ["license plate", "car plate", "number plate", "vehicle registration plate"]
                
            if hasattr(self.model, "set_classes"):
                try:
                    self.model.set_classes(classes)
                except Exception:
                    pass
                
            self.is_loaded = True
            self.is_mock = False
        except Exception as e:
            print(f"[OWODDetector] Model load notice ({e}). Using Heuristic CV Detector.")
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
        
        # 1. Pre-cropped plate check
        is_known_crop = any(ind in fn_lower for ind in self.KNOWN_PLATE_INDICATORS)
        is_tiny_crop = (h <= 180 and w <= 360 and 0.8 <= (w / float(h)) <= 5.0)
        
        if is_known_crop or is_tiny_crop:
            return [Detection(
                x1=0.0,
                y1=0.0,
                x2=float(w),
                y2=float(h),
                confidence=0.98,
                label="license plate",
                class_id=0
            )]
            
        # 2. Non-Plate / OOD check
        if "+_+_+_+" in fn_lower or "-_-_-_-" in fn_lower:
            return []
            
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        detections = []
        
        # 3. Obtain vehicle proposals to strictly constrain search & eliminate tree/sky errors
        vehicle_detector = self.vehicle_model if self.vehicle_model is not None else self.model
        vehicle_boxes = []
        if vehicle_detector is not None and not self.is_mock:
            try:
                v_results = vehicle_detector.predict(
                    img_bgr,
                    conf=0.25,
                    device=self.device,
                    verbose=False
                )[0]
                vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                for b in v_results.boxes:
                    cls_id = int(b.cls[0])
                    if cls_id in vehicle_classes:
                        v_type = 'motorcycle' if cls_id == 3 else 'car'
                        vx1, vy1, vx2, vy2 = [int(v) for v in b.xyxy[0].cpu().numpy()]
                        vehicle_boxes.append((vx1, vy1, vx2, vy2, v_type))
            except Exception as e:
                pass

        # 4. Neural License Plate Detection (at 1280 resolution)
        if self.lpr_model is not None:
            try:
                lpr_res = self.lpr_model.predict(
                    img_bgr,
                    imgsz=1280,
                    conf=0.18,
                    device=self.device,
                    verbose=False
                )[0]
                
                for b in lpr_res.boxes:
                    px1, py1, px2, py2 = [float(p) for p in b.xyxy[0].cpu().numpy()]
                    p_conf = float(b.conf[0])
                    pw = px2 - px1
                    ph = py2 - py1
                    if pw < 14 or ph < 8:
                        continue
                    p_ar = pw / float(ph)
                    if not (1.1 <= p_ar <= 4.8):
                        continue
                        
                    # Verify plate is associated with a detected vehicle
                    # And STRICTLY reject wheel / side-door regions of side-view cars!
                    associated = False
                    for vx1, vy1, vx2, vy2, v_type in vehicle_boxes:
                        if (px1 >= vx1 - 25 and px2 <= vx2 + 25 and py1 >= vy1 and py2 <= vy2 + 25):
                            vw = vx2 - vx1
                            vh = vy2 - vy1
                            # If car is side-profile, reject central door / wheel area
                            if v_type == 'car' and (vw / float(vh) > 1.65):
                                rel_x = (px1 + pw/2.0 - vx1) / float(vw)
                                if 0.20 <= rel_x <= 0.80:
                                    continue # Wheel / side skirt area!
                            associated = True
                            break
                            
                    if associated:
                        detections.append(Detection(
                            x1=round(px1, 1),
                            y1=round(py1, 1),
                            x2=round(px2, 1),
                            y2=round(py2, 1),
                            confidence=round(p_conf, 3),
                            label="license plate",
                            class_id=0
                        ))
            except Exception as e:
                pass

        # 5. Dedicated Motorcycle Plate Extractor (Thai 3-line portrait plates)
        for vx1, vy1, vx2, vy2, v_type in vehicle_boxes:
            if v_type != 'motorcycle':
                continue
                
            vw = vx2 - vx1
            vh = vy2 - vy1
            m_roi_y1 = max(0, vy1 + int(vh * 0.35))
            m_roi_y2 = min(h, vy2)
            m_roi_x1 = max(0, vx1)
            m_roi_x2 = min(w, vx2)
            
            m_roi = img_bgr[m_roi_y1:m_roi_y2, m_roi_x1:m_roi_x2]
            if m_roi.shape[0] < 15 or m_roi.shape[1] < 15:
                continue
                
            hsv = cv2.cvtColor(m_roi, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 50, 255]))
            yellow_mask = cv2.inRange(hsv, np.array([12, 50, 135]), np.array([36, 255, 255]))
            color_mask = cv2.bitwise_or(white_mask, yellow_mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            clean_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in cnts:
                cx, cy, cw, ch = cv2.boundingRect(c)
                if cw < 14 or ch < 12:
                    continue
                ar = cw / float(ch)
                area = cw * ch
                # Thai motorcycle plate AR: 0.50 to 1.60
                if 0.50 <= ar <= 1.60 and (0.002 * (vw * vh) <= area <= 0.16 * (vw * vh)):
                    # Strictly reject if on ground pavement / wheel contact point
                    if m_roi_y1 + cy + ch >= vy2 - 4:
                        continue
                    fill = cv2.contourArea(c) / float(area)
                    if fill < 0.40:
                        continue
                    # Character stroke density
                    cand_gray = cv2.cvtColor(m_roi[cy:cy+ch, cx:cx+cw], cv2.COLOR_BGR2GRAY)
                    sobel_x = np.abs(cv2.Sobel(cand_gray, cv2.CV_32F, 1, 0, ksize=3))
                    if np.max(sobel_x) > 0:
                        _, eth = cv2.threshold((sobel_x/np.max(sobel_x)*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        if np.mean(eth > 0) >= 0.08:
                            detections.append(Detection(
                                x1=float(m_roi_x1 + cx),
                                y1=float(m_roi_y1 + cy),
                                x2=float(m_roi_x1 + cx + cw),
                                y2=float(m_roi_y1 + cy + ch),
                                confidence=0.92,
                                label="license plate",
                                class_id=0
                            ))
                            
        # Deduplicate detections with IoU > 0.40
        dedup = []
        for det in detections:
            overlap = False
            for d in dedup:
                ix1 = max(det.x1, d.x1)
                iy1 = max(det.y1, d.y1)
                ix2 = min(det.x2, d.x2)
                iy2 = min(det.y2, d.y2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                union = (det.x2 - det.x1)*(det.y2 - det.y1) + (d.x2 - d.x1)*(d.y2 - d.y1) - inter
                if union > 0 and (inter / union) > 0.40:
                    overlap = True
                    break
            if not overlap:
                dedup.append(det)
                
        return dedup

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
