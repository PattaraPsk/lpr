"""
src/ocr/segmenter.py
Morphological character and band segmenter for Thai license plates.
Performs Hough deskewing, CLAHE equalization, horizontal projection split,
and Connected Component Analysis (CCA) glyph isolation.
"""
from typing import List, Optional, Tuple
import cv2
import numpy as np


class PlateSegmenter:
    """
    Splits Thai license plate into top registration band (consonants + digits)
    and bottom province band, filtering out screws and borders.
    """
    def __init__(
        self,
        target_char_height: int = 128,
        filter_screws: bool = True,
        max_deskew_angle: float = 25.0,
    ):
        self.target_char_height = target_char_height
        self.filter_screws = filter_screws
        self.max_deskew_angle = max_deskew_angle

    def deskew(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detects plate inclination via Hough lines and rotates to horizontal."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)

        angle = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines:
                coords = np.asarray(line).flatten()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    if x2 - x1 != 0:
                        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                        if -self.max_deskew_angle < deg < self.max_deskew_angle:
                            angles.append(deg)
            if len(angles) > 0:
                angle = float(np.median(angles))

        if abs(angle) > 0.8:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed, angle

        return img, 0.0

    def segment(
        self, plate_crop: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], float]:
        """
        Segments plate crop into:
        - List of individual normalized character image crops (top band)
        - Cropped province band image (bottom band)
        - Deskew rotation angle
        """
        if plate_crop is None or plate_crop.size == 0:
            return [], None, 0.0

        deskewed, angle = self.deskew(plate_crop)
        h, w = deskewed.shape[:2]

        gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY) if deskewed.ndim == 3 else deskewed
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Clear border noise
        margin_y = max(4, int(h * 0.05))
        margin_x = max(6, int(w * 0.035))
        clean_bin = binary.copy()
        clean_bin[:margin_y, :] = 0
        clean_bin[-margin_y:, :] = 0
        clean_bin[:, :margin_x] = 0
        clean_bin[:, -margin_x:] = 0

        # Horizontal projection profiling to separate top band from province text
        proj_y = np.sum(clean_bin > 0, axis=1)
        split_start = int(h * 0.48)
        split_end = int(h * 0.76)
        if split_end > split_start:
            split_y = split_start + int(np.argmin(proj_y[split_start:split_end]))
        else:
            split_y = int(h * 0.62)

        top_bin = clean_bin[:split_y, :]
        province_crop = deskewed[split_y:h, :]

        # Connected component analysis for individual characters in top band
        contours, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_boxes = []

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            if bw > 0.65 * w or bh > 0.95 * split_y:
                continue
            if bh >= 0.30 * split_y and area >= 250 and bw >= 10:
                is_screw = (by < split_y * 0.35) and (bx < w * 0.18 or bx > w * 0.82) and (bh < split_y * 0.38)
                if not (self.filter_screws and is_screw):
                    valid_boxes.append((bx, by, bw, bh))

        # Sort left to right
        valid_boxes.sort(key=lambda b: b[0])

        char_crops = []
        for bx, by, bw, bh in valid_boxes:
            char_patch = deskewed[by:by + bh, bx:bx + bw]
            if char_patch.size > 0:
                scale = self.target_char_height / float(bh)
                target_w = max(4, int(round(bw * scale)))
                resized = cv2.resize(char_patch, (target_w, self.target_char_height), interpolation=cv2.INTER_LANCZOS4)
                char_crops.append(resized)

        return char_crops, province_crop, angle
