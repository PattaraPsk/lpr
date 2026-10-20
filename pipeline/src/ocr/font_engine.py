"""
src/ocr/font_engine.py
Thai License Plate Typography Engine, Stroke Analyzer & Glyph Renderer.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class ThaiPlateMetrics:
    mean_aspect_ratio_digits: float = 0.58
    mean_aspect_ratio_thai: float = 0.72
    mean_stroke_width_ratio: float = 0.14
    mean_fill_ratio: float = 0.55
    char_height_to_plate_ratio: float = 0.38
    province_height_to_char_ratio: float = 0.35
    letter_to_number_gap_ratio: float = 0.85
    intra_char_gap_ratio: float = 0.28


class FontEngine:
    def __init__(self, font_path: Optional[str] = None):
        self.metrics = ThaiPlateMetrics()
        default_font = Path(__file__).parent / "fonts" / "NotoSansThai-Bold.ttf"
        if font_path and Path(font_path).exists():
            self.font_path = str(font_path)
        elif default_font.exists():
            self.font_path = str(default_font)
        else:
            self.font_path = "arial.ttf"

    @staticmethod
    def estimate_stroke_width(binary_mask: np.ndarray) -> float:
        """Computes average stroke width using Euclidean distance transform."""
        if np.sum(binary_mask > 0) == 0:
            return 0.0
        dist = cv2.distanceTransform((binary_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        non_zero = dist[dist > 0]
        return float(np.median(non_zero) * 2.0) if len(non_zero) > 0 else 0.0

    def render_plate(
        self,
        plate_w: int,
        plate_h: int,
        consonants: str,
        digits: str,
        province: str,
        bg_color: Tuple[int, int, int] = (248, 248, 248),
        text_color: Tuple[int, int, int] = (20, 20, 20),
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Renders synthetic Thai plate according to official DLT proportions."""
        img = Image.new("RGB", (plate_w, plate_h), bg_color)
        draw = ImageDraw.Draw(img)

        # Draw outer rounded border
        margin = int(plate_h * 0.04)
        border_w = max(2, int(plate_h * 0.025))
        draw.rounded_rectangle(
            [(margin, margin), (plate_w - margin, plate_h - margin)],
            radius=int(plate_h * 0.06),
            outline=text_color,
            width=border_w,
        )

        char_h = int(round(plate_h * self.metrics.char_height_to_plate_ratio))
        prov_h = int(round(char_h * self.metrics.province_height_to_char_ratio))

        try:
            font_top = ImageFont.truetype(self.font_path, char_h)
            font_prov = ImageFont.truetype(self.font_path, prov_h)
        except Exception:
            font_top = ImageFont.load_default()
            font_prov = ImageFont.load_default()

        # Render top text (consonants + digits)
        top_text = f"{consonants} {digits}"
        bbox_top = draw.textbbox((0, 0), top_text, font=font_top)
        top_w = bbox_top[2] - bbox_top[0]
        top_x = (plate_w - top_w) // 2
        top_y = int(plate_h * 0.12)
        draw.text((top_x, top_y), top_text, fill=text_color, font=font_top)

        # Render province text
        bbox_prov = draw.textbbox((0, 0), province, font=font_prov)
        prov_w = bbox_prov[2] - bbox_prov[0]
        prov_x = (plate_w - prov_w) // 2
        prov_y = int(plate_h * 0.64)
        draw.text((prov_x, prov_y), province, fill=text_color, font=font_prov)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), []
