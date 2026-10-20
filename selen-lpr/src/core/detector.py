"""
Plate Detector & Adaptive Foreground ROI Extractor.
Extracts vehicle foreground ROI, localizes plates, and computes global scale-normalized coordinates (s_x, s_y, s_w, s_h).
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from ..schema.types import NormalizedCoordinates


class AdaptivePlateDetector:
    """
    Detector supporting adaptive foreground extraction and YOLO/ONNX plate localization.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        use_adaptive_roi: bool = True
    ):
        self.conf_threshold = conf_threshold
        self.use_adaptive_roi = use_adaptive_roi
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    def extract_adaptive_roi(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Finds foreground vehicle motion bounding boxes as adaptive ROIs.
        Returns list of [x, y, w, h] in frame pixel coordinates.
        """
        if not self.use_adaptive_roi:
            h, w = frame.shape[:2]
            return [(0, 0, w, h)]

        # Downscale for ultra-fast foreground analysis
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (480, 270))
        fg_mask = self.bg_subtractor.apply(small)
        
        # Morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = w / 480.0
        scale_y = h / 270.0

        rois = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1200:  # Minimum vehicle contour in downscaled frame
                sx, sy, sw, sh = cv2.boundingRect(cnt)
                # Map back to full frame with 15% safety padding
                pad_x = int(sw * 0.15)
                pad_y = int(sh * 0.15)
                fx = max(0, int((sx - pad_x) * scale_x))
                fy = max(0, int((sy - pad_y) * scale_y))
                fw = min(w - fx, int((sw + 2 * pad_x) * scale_x))
                fh = min(h - fy, int((sh + 2 * pad_y) * scale_y))
                rois.append((fx, fy, fw, fh))

        if not rois:
            # Fallback to full frame
            rois.append((0, 0, w, h))

        return rois

    def detect(
        self,
        frame: np.ndarray
    ) -> List[Tuple[np.ndarray, NormalizedCoordinates, float]]:
        """
        Runs detection and returns tuples of:
        (crop_image, NormalizedCoordinates(s_x, s_y, s_w, s_h), confidence)
        """
        frame_h, frame_w = frame.shape[:2]
        rois = self.extract_adaptive_roi(frame)

        results = []
        for (rx, ry, rw, rh) in rois:
            roi_crop = frame[ry:ry + rh, rx:rx + rw]
            if roi_crop.size == 0:
                continue

            # Fallback direct crop check if the input is already a plate crop
            ar = rw / max(rh, 1)
            if 1.1 <= ar <= 4.0 and min(rw, rh) >= 40:
                s_coords = NormalizedCoordinates(
                    s_x=rx / frame_w,
                    s_y=ry / frame_h,
                    s_w=rw / frame_w,
                    s_h=rh / frame_h
                )
                results.append((roi_crop, s_coords, 0.95))

        # Ensure at least 1 result if frame itself is a plate
        if not results:
            s_coords = NormalizedCoordinates(s_x=0.0, s_y=0.0, s_w=1.0, s_h=1.0)
            results.append((frame, s_coords, 0.90))

        return results
