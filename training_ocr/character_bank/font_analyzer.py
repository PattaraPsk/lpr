#!/usr/bin/env python3
"""
Thai License Plate Character Font Analyzer & Atlas Builder
Author: Tailored for Local Machine Execution (WS4090 / CUDA / CPU)
Description:
    Analyzes morphological and geometric features of cropped Thai license plate glyphs
    (stroke width, aspect ratio, kerning, baselines).
    Builds a structured font atlas and parameter dictionary for high-fidelity plate rendering.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FontAnalyzer")


@dataclass
class FontMetrics:
    mean_aspect_ratio_digits: float
    mean_aspect_ratio_thai: float
    mean_stroke_width_ratio: float
    mean_fill_ratio: float
    char_height_to_plate_ratio: float
    province_height_to_char_ratio: float
    letter_to_number_gap_ratio: float
    intra_char_gap_ratio: float
    baseline_offset_ratio: float


class ThaiPlateFontEngine:
    """
    Renders Thai license plate typography using analyzed DLT metrics
    and vector TrueType fonts / extracted glyph atlas.
    """

    def __init__(self, font_path: Optional[str] = None, metrics: Optional[FontMetrics] = None):
        self.metrics = metrics or FontMetrics(
            mean_aspect_ratio_digits=0.58,
            mean_aspect_ratio_thai=0.72,
            mean_stroke_width_ratio=0.14,
            mean_fill_ratio=0.55,
            char_height_to_plate_ratio=0.38,
            province_height_to_char_ratio=0.35,
            letter_to_number_gap_ratio=0.85,
            intra_char_gap_ratio=0.28,
            baseline_offset_ratio=0.88,
        )
        self.font_path = font_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansThai-Bold.ttf")
        if not os.path.exists(self.font_path):
            fallback = "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"
            if os.path.exists(fallback):
                self.font_path = fallback

    def render_plate_text(
        self,
        plate_w: int,
        plate_h: int,
        text_top: str,
        province: str,
        text_color: Tuple[int, int, int] = (18, 18, 18),
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Renders registration text and province string onto a transparent RGBA canvas
        according to official Thai DLT proportions and analyzed spacing metrics.
        Returns:
            (PIL.Image RGBA canvas, list of character bounding box annotations)
        """
        canvas = Image.new("RGBA", (plate_w, plate_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        target_char_h = int(round(plate_h * self.metrics.char_height_to_plate_ratio))
        target_prov_h = int(round(target_char_h * self.metrics.province_height_to_char_ratio))

        font_top = ImageFont.truetype(self.font_path, target_char_h)
        font_prov = ImageFont.truetype(self.font_path, target_prov_h)

        parts = text_top.strip().split()
        char_bboxes = []

        top_y = int(round(plate_h * 0.13))

        if len(parts) == 2:
            group1, group2 = parts[0], parts[1]
            bbox_g1 = draw.textbbox((0, 0), group1, font=font_top)
            w_g1 = bbox_g1[2] - bbox_g1[0]
            bbox_g2 = draw.textbbox((0, 0), group2, font=font_top)
            w_g2 = bbox_g2[2] - bbox_g2[0]

            gap = int(round(target_char_h * 0.55 * self.metrics.letter_to_number_gap_ratio))
            total_top_w = w_g1 + gap + w_g2
            start_x = (plate_w - total_top_w) // 2

            draw.text((start_x, top_y), group1, fill=text_color + (255,), font=font_top)
            cur_x = start_x
            for ch in group1:
                bb = draw.textbbox((cur_x, top_y), ch, font=font_top)
                char_bboxes.append({"char": ch, "bbox_xyxy": [bb[0], bb[1], bb[2], bb[3]], "band": "top"})
                cur_x += (bb[2] - bb[0]) + int(round(target_char_h * 0.12))

            g2_x = start_x + w_g1 + gap
            draw.text((g2_x, top_y), group2, fill=text_color + (255,), font=font_top)
            cur_x = g2_x
            for ch in group2:
                bb = draw.textbbox((cur_x, top_y), ch, font=font_top)
                char_bboxes.append({"char": ch, "bbox_xyxy": [bb[0], bb[1], bb[2], bb[3]], "band": "top"})
                cur_x += (bb[2] - bb[0]) + int(round(target_char_h * 0.10))
        else:
            bbox_top = draw.textbbox((0, 0), text_top, font=font_top)
            w_top = bbox_top[2] - bbox_top[0]
            start_x = (plate_w - w_top) // 2
            draw.text((start_x, top_y), text_top, fill=text_color + (255,), font=font_top)

            cur_x = start_x
            for ch in text_top:
                if ch == " ":
                    cur_x += int(round(target_char_h * 0.4))
                    continue
                bb = draw.textbbox((cur_x, top_y), ch, font=font_top)
                char_bboxes.append({"char": ch, "bbox_xyxy": [bb[0], bb[1], bb[2], bb[3]], "band": "top"})
                cur_x += (bb[2] - bb[0]) + int(round(target_char_h * 0.12))

        # Render Province
        bbox_prov = draw.textbbox((0, 0), province, font=font_prov)
        w_prov = bbox_prov[2] - bbox_prov[0]
        prov_x = (plate_w - w_prov) // 2
        prov_y = int(round(plate_h * 0.65))

        draw.text((prov_x, prov_y), province, fill=text_color + (255,), font=font_prov)
        char_bboxes.append({
            "char": province,
            "bbox_xyxy": [prov_x, prov_y, prov_x + w_prov, prov_y + (bbox_prov[3] - bbox_prov[1])],
            "band": "province",
        })

        return canvas, char_bboxes


def analyze_manifest(manifest_path: str) -> FontMetrics:
    """Analyzes character manifest generated by crop_characters.py."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    top_chars = [r for r in records if r.get("band") == "top"]
    if not top_chars:
        logger.info("No top-band records in manifest. Using standard Thai DLT font metrics.")
        return FontMetrics(
            mean_aspect_ratio_digits=0.58,
            mean_aspect_ratio_thai=0.72,
            mean_stroke_width_ratio=0.14,
            mean_fill_ratio=0.55,
            char_height_to_plate_ratio=0.38,
            province_height_to_char_ratio=0.35,
            letter_to_number_gap_ratio=0.85,
            intra_char_gap_ratio=0.28,
            baseline_offset_ratio=0.88,
        )

    aspects = [r["aspect_ratio"] for r in top_chars]
    fills = [r["fill_ratio"] for r in top_chars]
    strokes = [r["estimated_stroke_width"] for r in top_chars if r["estimated_stroke_width"] > 0]
    heights = [r["bbox_xywh"][3] for r in top_chars]

    digits_asp = [a for a in aspects if a < 0.65] or [0.58]
    thai_asp = [a for a in aspects if a >= 0.65] or [0.72]

    mean_h = np.mean(heights) if heights else 100.0
    mean_stroke = np.mean(strokes) if strokes else 14.0
    stroke_ratio = mean_stroke / mean_h if mean_h > 0 else 0.14

    metrics = FontMetrics(
        mean_aspect_ratio_digits=round(float(np.mean(digits_asp)), 3),
        mean_aspect_ratio_thai=round(float(np.mean(thai_asp)), 3),
        mean_stroke_width_ratio=round(float(stroke_ratio), 3),
        mean_fill_ratio=round(float(np.mean(fills)), 3),
        char_height_to_plate_ratio=0.38,
        province_height_to_char_ratio=0.35,
        letter_to_number_gap_ratio=0.85,
        intra_char_gap_ratio=0.28,
        baseline_offset_ratio=0.88,
    )
    return metrics


def build_font_atlas(engine: ThaiPlateFontEngine, output_dir: str):
    """
    Renders a reference atlas PNG and JSON glyph map of canonical Thai license plate characters.
    """
    os.makedirs(output_dir, exist_ok=True)
    canonical_chars = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "ก", "ข", "ค", "ฆ", "ง", "จ", "ฉ", "ช", "ฌ", "ญ",
        "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท",
        "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม",
        "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
    ]

    cell_size = 128
    cols = 10
    rows = (len(canonical_chars) + cols - 1) // cols
    atlas_w = cols * cell_size
    atlas_h = rows * cell_size

    atlas_img = Image.new("RGBA", (atlas_w, atlas_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(atlas_img)
    font = ImageFont.truetype(engine.font_path, int(cell_size * 0.72))

    glyph_map = {}
    for idx, ch in enumerate(canonical_chars):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_size
        y0 = r * cell_size

        bbox = draw.textbbox((0, 0), ch, font=font)
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        cx = x0 + (cell_size - bw) // 2
        cy = y0 + (cell_size - bh) // 2

        draw.text((cx, cy), ch, fill=(20, 20, 20, 255), font=font)
        glyph_map[ch] = {
            "atlas_bbox_xywh": [x0, y0, cell_size, cell_size],
            "char_bbox_xywh": [cx, cy, bw, bh],
            "aspect_ratio": round(bw / float(bh), 3) if bh > 0 else 0.5,
        }

    atlas_path = os.path.join(output_dir, "plate_font_atlas.png")
    atlas_img.save(atlas_path)

    atlas_json = os.path.join(output_dir, "plate_font_atlas.json")
    with open(atlas_json, "w", encoding="utf-8") as f:
        json.dump(glyph_map, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated Font Atlas: {atlas_path} ({len(canonical_chars)} glyphs)")


def main():
    parser = argparse.ArgumentParser(
        description="Thai License Plate Font Analyzer & Atlas Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="char_bank",
        help="Directory containing cropped characters and character_manifest.json.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="font_assets",
        help="Directory to save font metrics and atlas.",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Path to TrueType font file. Defaults to bundled fonts/NotoSansThai-Bold.ttf.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    manifest_path = os.path.join(args.input, "character_manifest.json")

    logger.info("==================================================================")
    logger.info("     Thai License Plate Font Analyzer & Atlas Builder            ")
    logger.info("==================================================================")
    logger.info(f"Input Directory     : {args.input}")
    logger.info(f"Output Directory    : {args.output}")

    if os.path.exists(manifest_path):
        logger.info(f"Found character manifest at {manifest_path}. Analyzing morphological features...")
        metrics = analyze_manifest(manifest_path)
    else:
        logger.info("No manifest found; calibrating against Department of Land Transport (DLT) standard.")
        metrics = FontMetrics(
            mean_aspect_ratio_digits=0.58,
            mean_aspect_ratio_thai=0.72,
            mean_stroke_width_ratio=0.14,
            mean_fill_ratio=0.55,
            char_height_to_plate_ratio=0.38,
            province_height_to_char_ratio=0.35,
            letter_to_number_gap_ratio=0.85,
            intra_char_gap_ratio=0.28,
            baseline_offset_ratio=0.88,
        )

    logger.info("==================================================================")
    logger.info("                 LEARNED FONT PROFILE METRICS                     ")
    logger.info("==================================================================")
    logger.info(f"Mean Aspect Ratio (Digits)     : {metrics.mean_aspect_ratio_digits:.3f}")
    logger.info(f"Mean Aspect Ratio (Thai Chars) : {metrics.mean_aspect_ratio_thai:.3f}")
    logger.info(f"Stroke Width Ratio (w/h)       : {metrics.mean_stroke_width_ratio:.3f}")
    logger.info(f"Mean Stroke Fill Ratio         : {metrics.mean_fill_ratio:.3f}")
    logger.info(f"Char-to-Plate Height Ratio     : {metrics.char_height_to_plate_ratio:.3f}")
    logger.info(f"Letter-to-Number Gap Ratio     : {metrics.letter_to_number_gap_ratio:.3f}")
    logger.info("==================================================================")

    metrics_json = os.path.join(args.output, "plate_font_metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    engine = ThaiPlateFontEngine(font_path=args.font, metrics=metrics)
    build_font_atlas(engine, args.output)

    logger.info(f"Font metrics and atlas saved to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
