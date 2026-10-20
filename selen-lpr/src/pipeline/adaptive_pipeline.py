"""
Master Selen-LPR Adaptive Pipeline (Pure PyTorch & OpenCV).
Zero EasyOCR, Zero Ultralytics dependencies.
Combines:
1. Adaptive Vehicle Foreground ROI Extractor
2. Pure PyTorch Plate Detector
3. Global Scale Normalization (s_x, s_y, s_w, s_h in [0.0, 1.0])
4. CIE LAB Color & Layout Classifier
5. Pure PyTorch Vision-Attention OCR Engine (64x600 RCNN + BiLSTM + Attention)
6. DLT Legal Fusion & Cross-Border Classification Engine
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import torch

from PIL import Image

from ..models.plate_detector import PureTorchPlateDetector
from ..models.vision_attention_ocr import VisionAttentionOCR
from ..models.vocab import decode_raw_prediction
from ..utils.color_extractor import extract_plate_colors
from ..utils.dlt_rules import map_plate_to_vehicle_type
from ..schema.types import NormalizedCoordinates, PlateDesign, VehicleType, PlateReading, DetectionResult


class SelenLPRPipeline:
    """
    High-performance, pure PyTorch commercial LPR pipeline.
    """

    def __init__(
        self,
        ocr_weights_path: Optional[str] = None,
        detector_weights_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        conf_threshold: float = 0.25
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        print(f"[INFO] Initializing SelenLPRPipeline on device: {self.device}")

        # 1. Initialize Pure PyTorch Detector
        self.detector = PureTorchPlateDetector(
            weights_path=detector_weights_path,
            conf_threshold=conf_threshold,
            device=str(self.device)
        )

        # 2. Adaptive Foreground ROI Extractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

        # 3. Initialize Pure PyTorch Vision-Attention OCR Model
        self.ocr_model = VisionAttentionOCR(num_classes=187, batch_max_length=25)
        if ocr_weights_path and Path(ocr_weights_path).exists():
            print(f"[INFO] Loading standalone PyTorch OCR weights from: {ocr_weights_path}")
            state = torch.load(ocr_weights_path, map_location=self.device)
            self.ocr_model.load_state_dict(state)
        self.ocr_model.to(self.device)
        self.ocr_model.eval()

    def extract_adaptive_rois(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Extracts vehicle motion bounding box as an adaptive ROI.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, 270))
        fg_mask = self.bg_subtractor.apply(small)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = w / 480.0
        scale_y = h / 270.0

        rois = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 1200:
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                pad_x = int(sw * 0.15)
                pad_y = int(sh * 0.15)
                fx = max(0, int((sx - pad_x) * scale_x))
                fy = max(0, int((sy - pad_y) * scale_y))
                fw = min(w - fx, int((sw + 2 * pad_x) * scale_x))
                fh = min(h - fy, int((sh + 2 * pad_y) * scale_y))
                rois.append((fx, fy, fw, fh))

        return rois if rois else [(0, 0, w, h)]

    def _preprocess_ocr_crop(self, crop_bgr: np.ndarray, target_h: int = 64, target_w: int = 600) -> torch.Tensor:
        """
        Prepares plate crop for pure PyTorch Vision-Attention OCR matching authentic NormalizePAD.
        Converts to grayscale, PIL BICUBIC resize with aspect ratio preservation,
        edge border replication padding, and normalized to [-1, 1].
        """
        if len(crop_bgr.shape) == 3:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_bgr
        h, w = gray.shape
        ratio = w / float(max(h, 1))
        resized_w = min(int(np.ceil(target_h * ratio)), target_w)

        pil_img = Image.fromarray(gray, 'L')
        resized = pil_img.resize((resized_w, target_h), Image.BICUBIC)

        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # [-1, 1]

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, resized_w]
        pad_tensor = torch.zeros(1, 1, target_h, target_w, dtype=torch.float32)
        pad_tensor[:, :, :, :resized_w] = tensor
        if target_w != resized_w:
            pad_tensor[:, :, :, resized_w:] = tensor[:, :, :, resized_w - 1:resized_w].expand(1, 1, target_h, target_w - resized_w)
        return pad_tensor.to(self.device)

    def process_frame(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Executes complete inference pipeline on an input frame.
        """
        frame_h, frame_w = frame.shape[:2]
        rois = self.extract_adaptive_rois(frame)

        results: List[DetectionResult] = []

        for roi in rois:
            plates = self.detector.detect_in_roi(frame, roi)

            for crop_img, (gx, gy, gw, gh), conf in plates:
                if gw < 20 or gh < 10:
                    continue

                # Normalize plate aspect ratio: Thai vehicle plates are ~2.2 - 2.5
                # Trim extra vertical margin / bumper shadow if detector box is too tall
                if gw / max(gh, 1) < 1.8 and gw > 35:
                    target_h = max(15, int(gw / 2.25))
                    diff = gh - target_h
                    if diff > 0:
                        gy = min(frame_h - target_h, gy + int(diff * 0.25))
                        gh = target_h
                        crop_img = frame[max(0, gy):min(frame_h, gy + gh), max(0, gx):min(frame_w, gx + gw)]

                aspect_ratio = gw / max(gh, 1)

                # 1. Global Normalized Scale Coordinates [0.0, 1.0]
                s_coords = NormalizedCoordinates(
                    s_x=gx / frame_w,
                    s_y=gy / frame_h,
                    s_w=gw / frame_w,
                    s_h=gh / frame_h
                )

                # 2. CIE LAB Color & Layout Classification (Categorize Vehicle First)
                bg_color, text_color = extract_plate_colors(crop_img)
                layout = "motorcycle_double_row" if aspect_ratio < 1.45 else "standard_single_row"
                design = PlateDesign(background_color=bg_color, text_color=text_color, layout=layout)

                # Preliminary vehicle categorization based on design & aspect ratio
                cat_id, desc = map_plate_to_vehicle_type(
                    bg_color=design.background_color,
                    text_color=design.text_color,
                    d3_prefix="",
                    aspect_ratio=aspect_ratio
                )

                # 3. Pure PyTorch Vision-Attention OCR Forward Pass
                ocr_input = self._preprocess_ocr_crop(crop_img)
                with torch.no_grad():
                    probs = self.ocr_model(ocr_input)  # [1, 26, 187]
                    pred_indices = probs.squeeze(0).argmax(dim=-1).cpu().numpy().tolist()

                # Category-First Decoding: cuts out plate IDs incompatible with this vehicle category
                d3, d4, province, full_plate = decode_raw_prediction(pred_indices, vehicle_category=desc)
                reading = PlateReading(d3=d3, d4=d4, province=province, full_plate=full_plate)

                # Refine vehicle categorization with decoded prefix
                cat_id, desc = map_plate_to_vehicle_type(
                    bg_color=design.background_color,
                    text_color=design.text_color,
                    d3_prefix=reading.d3,
                    aspect_ratio=aspect_ratio
                )
                vehicle = VehicleType(category_id=cat_id, description=desc)

                result = DetectionResult(
                    normalized_coords=s_coords,
                    confidence=conf,
                    plate_design=design,
                    vehicle_type=vehicle,
                    reading=reading
                )
                results.append(result)

        return results
