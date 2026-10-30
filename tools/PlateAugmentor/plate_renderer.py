#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic Thai License Plate Renderer
======================================
Generates photorealistic Thai license plate images matching DLT specifications.
Supports all plate types: WhiteBlackCar, WhiteBlackCar2Line, WhiteBlackMotor,
RedBlack, YellowBlack.

Returns plate images with per-character bounding boxes for YOLO annotation.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Thai Domain Constants (DLT Specifications)
# ---------------------------------------------------------------------------

VALID_CONSONANTS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
DIGITS = list("0123456789")

# OCR remapping for excluded characters
OCR_REMAP = {"ฃ": "ข", "ฅ": "ค", "ฏ": "ฎ"}

# Excluded negative word combos (DLT regulation)
EXCLUDED_COMBOS = {"ศพ", "สพ", "ศบ", "ษพ", "ชน", "จน", "ตด", "งง"}

# Font paths (prefer project font, fallback chain)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FONT_PATHS = [
    ROOT_DIR / "dataset" / "fonts" / "rbz-licenseplate.ttf",
    ROOT_DIR / "model" / "fonts" / "rbz-licenseplate.ttf",
    ROOT_DIR / "models" / "fonts" / "rbz-licenseplate.ttf",
]

# Province list (inline subset for standalone usage)
try:
    sys.path.insert(0, str(ROOT_DIR))
    from tools.DataManipulator.app.provinces import PROVINCES_ISO_DATA
    PROVINCES = [th for th, en, iso in PROVINCES_ISO_DATA]
except ImportError:
    PROVINCES = [
        "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี",
        "ชลบุรี", "เชียงใหม่", "ขอนแก่น", "นครราชสีมา", "ภูเก็ต",
        "สงขลา", "สุราษฎร์ธานี", "เชียงราย", "อุดรธานี", "นครสวรรค์",
    ]

# ---------------------------------------------------------------------------
# Plate Type Configurations
# ---------------------------------------------------------------------------

PLATE_CONFIGS = {
    "WhiteBlackCar": {
        "width": 520, "height": 110,
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "border_color": (0, 0, 0),
        "province_color": (80, 80, 80),
        "layout": "single_line",
        "d3_len": (2, 2),  # min, max consonants
        "d4_len": (4, 4),  # exactly 4 digits
        "display_name": "White/Black Car (Single Line)",
    },
    "WhiteBlackCar2Line": {
        "width": 340, "height": 170,
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "border_color": (0, 0, 0),
        "province_color": (80, 80, 80),
        "layout": "two_line",
        "d3_len": (2, 3),
        "d4_len": (4, 4),
        "display_name": "White/Black Car (Two Line)",
    },
    "WhiteBlackMotor": {
        "width": 220, "height": 170,
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "border_color": (0, 0, 0),
        "province_color": (80, 80, 80),
        "layout": "motorcycle",
        "d3_len": (1, 2),
        "d4_len": (3, 4),
        "display_name": "White/Black Motorcycle",
    },
    "RedBlack": {
        "width": 340, "height": 170,
        "bg_color": (220, 38, 38),
        "text_color": (255, 255, 255),
        "border_color": (0, 0, 0),
        "province_color": (240, 240, 240),
        "layout": "two_line",
        "d3_len": (2, 2),
        "d4_len": (4, 4),
        "display_name": "Red Temporary Plate",
    },
    "YellowBlack": {
        "width": 340, "height": 170,
        "bg_color": (250, 204, 21),
        "text_color": (0, 0, 0),
        "border_color": (0, 0, 0),
        "province_color": (60, 60, 60),
        "layout": "two_line",
        "d3_len": (2, 2),
        "d4_len": (4, 4),
        "display_name": "Yellow/Black Taxi",
    },
}


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    """Load Thai license plate font with fallback chain."""
    for fp in FONT_PATHS:
        if fp.exists():
            try:
                return ImageFont.truetype(str(fp), size)
            except Exception:
                continue

    # System Thai font fallbacks (Windows)
    system_fallbacks = [
        "C:/Windows/Fonts/THSarabunNew.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in system_fallbacks:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue

    # Ultimate fallback
    return ImageFont.load_default()


def _draw_rounded_rect(draw: ImageDraw.ImageDraw, xy: Tuple, fill, outline, width=2, radius=4):
    """Draw a rounded rectangle."""
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _generate_random_d3(length: int) -> str:
    """Generate random Thai consonant prefix, avoiding excluded combos."""
    for _ in range(100):
        chars = "".join(random.choices(VALID_CONSONANTS, k=length))
        if chars not in EXCLUDED_COMBOS:
            return chars
    return "กข"  # safe fallback


def _generate_random_d4(length: int) -> str:
    """Generate random digit string."""
    return "".join(random.choices(DIGITS, k=length))


# ---------------------------------------------------------------------------
# Rendering Functions
# ---------------------------------------------------------------------------

def render_single_line(
    d3: str, d4: str, province: str, config: dict
) -> Tuple[Image.Image, List[Dict]]:
    """Render single-line plate: [d3 d4] with province below."""
    w, h = config["width"], config["height"]
    img = Image.new("RGB", (w, h), config["bg_color"])
    draw = ImageDraw.Draw(img)

    # Border
    _draw_rounded_rect(draw, (0, 0, w - 1, h - 1),
                       fill=config["bg_color"], outline=config["border_color"],
                       width=2, radius=4)

    # Fonts
    main_font = _find_font(48)
    prov_font = _find_font(18)

    plate_text = f"{d3} {d4}"
    bboxes = []

    # Measure total text width to center
    text_bbox = draw.textbbox((0, 0), plate_text, font=main_font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    start_x = (w - text_w) // 2
    y_main = 12

    # Draw characters and track bboxes
    cursor_x = start_x
    for ch in plate_text:
        if ch == " ":
            ch_bbox = draw.textbbox((cursor_x, y_main), " ", font=main_font)
            cursor_x = ch_bbox[2]
            continue
        ch_bbox = draw.textbbox((cursor_x, y_main), ch, font=main_font)
        draw.text((cursor_x, y_main), ch, fill=config["text_color"], font=main_font)
        bboxes.append({
            "char": ch,
            "bbox": (ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3])
        })
        cursor_x = ch_bbox[2] + 2

    # Province text
    prov_bbox = draw.textbbox((0, 0), province, font=prov_font)
    prov_w = prov_bbox[2] - prov_bbox[0]
    prov_x = (w - prov_w) // 2
    prov_y = h - 30
    draw.text((prov_x, prov_y), province, fill=config["province_color"], font=prov_font)

    return img, bboxes


def render_two_line(
    d3: str, d4: str, province: str, config: dict
) -> Tuple[Image.Image, List[Dict]]:
    """Render two-line plate: top=consonants, bottom=digits, province below."""
    w, h = config["width"], config["height"]
    img = Image.new("RGB", (w, h), config["bg_color"])
    draw = ImageDraw.Draw(img)

    # Border
    _draw_rounded_rect(draw, (0, 0, w - 1, h - 1),
                       fill=config["bg_color"], outline=config["border_color"],
                       width=2, radius=4)

    # Fonts
    main_font = _find_font(46)
    prov_font = _find_font(16)

    bboxes = []

    # Row 1: Thai consonants (d3)
    d3_bbox = draw.textbbox((0, 0), d3, font=main_font)
    d3_w = d3_bbox[2] - d3_bbox[0]
    d3_x = (w - d3_w) // 2
    d3_y = 8

    cursor_x = d3_x
    for ch in d3:
        ch_bbox = draw.textbbox((cursor_x, d3_y), ch, font=main_font)
        draw.text((cursor_x, d3_y), ch, fill=config["text_color"], font=main_font)
        bboxes.append({"char": ch, "bbox": (ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3])})
        cursor_x = ch_bbox[2] + 3

    # Horizontal divider
    divider_y = 65
    draw.line([(20, divider_y), (w - 20, divider_y)], fill=config["border_color"], width=1)

    # Row 2: Digits (d4)
    d4_bbox = draw.textbbox((0, 0), d4, font=main_font)
    d4_w = d4_bbox[2] - d4_bbox[0]
    d4_x = (w - d4_w) // 2
    d4_y = divider_y + 5

    cursor_x = d4_x
    for ch in d4:
        ch_bbox = draw.textbbox((cursor_x, d4_y), ch, font=main_font)
        draw.text((cursor_x, d4_y), ch, fill=config["text_color"], font=main_font)
        bboxes.append({"char": ch, "bbox": (ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3])})
        cursor_x = ch_bbox[2] + 3

    # Province text
    prov_bbox_m = draw.textbbox((0, 0), province, font=prov_font)
    prov_w = prov_bbox_m[2] - prov_bbox_m[0]
    prov_x = (w - prov_w) // 2
    prov_y = h - 26
    draw.text((prov_x, prov_y), province, fill=config["province_color"], font=prov_font)

    return img, bboxes


def render_motorcycle(
    d3: str, d4: str, province: str, config: dict
) -> Tuple[Image.Image, List[Dict]]:
    """Render motorcycle plate: square format, consonants top, digits bottom."""
    w, h = config["width"], config["height"]
    img = Image.new("RGB", (w, h), config["bg_color"])
    draw = ImageDraw.Draw(img)

    # Border
    _draw_rounded_rect(draw, (0, 0, w - 1, h - 1),
                       fill=config["bg_color"], outline=config["border_color"],
                       width=2, radius=4)

    # Fonts
    main_font = _find_font(42)
    prov_font = _find_font(14)

    bboxes = []

    # Row 1: Thai consonants
    d3_bbox = draw.textbbox((0, 0), d3, font=main_font)
    d3_w = d3_bbox[2] - d3_bbox[0]
    d3_x = (w - d3_w) // 2
    d3_y = 10

    cursor_x = d3_x
    for ch in d3:
        ch_bbox = draw.textbbox((cursor_x, d3_y), ch, font=main_font)
        draw.text((cursor_x, d3_y), ch, fill=config["text_color"], font=main_font)
        bboxes.append({"char": ch, "bbox": (ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3])})
        cursor_x = ch_bbox[2] + 2

    # Divider
    divider_y = 60
    draw.line([(15, divider_y), (w - 15, divider_y)], fill=config["border_color"], width=1)

    # Row 2: Digits
    d4_bbox = draw.textbbox((0, 0), d4, font=main_font)
    d4_w = d4_bbox[2] - d4_bbox[0]
    d4_x = (w - d4_w) // 2
    d4_y = divider_y + 5

    cursor_x = d4_x
    for ch in d4:
        ch_bbox = draw.textbbox((cursor_x, d4_y), ch, font=main_font)
        draw.text((cursor_x, d4_y), ch, fill=config["text_color"], font=main_font)
        bboxes.append({"char": ch, "bbox": (ch_bbox[0], ch_bbox[1], ch_bbox[2], ch_bbox[3])})
        cursor_x = ch_bbox[2] + 2

    # Province text
    prov_bbox_m = draw.textbbox((0, 0), province, font=prov_font)
    prov_w = prov_bbox_m[2] - prov_bbox_m[0]
    prov_x = (w - prov_w) // 2
    prov_y = h - 24
    draw.text((prov_x, prov_y), province, fill=config["province_color"], font=prov_font)

    return img, bboxes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LAYOUT_RENDERERS = {
    "single_line": render_single_line,
    "two_line": render_two_line,
    "motorcycle": render_motorcycle,
}


def render_plate(
    plate_type: str,
    d3: str,
    d4: str,
    province: str,
) -> Tuple[Image.Image, List[Dict], dict]:
    """
    Render a Thai license plate image.

    Args:
        plate_type: One of PLATE_CONFIGS keys
        d3: Thai consonant prefix (e.g. "กข")
        d4: Digit string (e.g. "1234")
        province: Thai province name

    Returns:
        (plate_image, char_bboxes, metadata)
    """
    if plate_type not in PLATE_CONFIGS:
        raise ValueError(f"Unknown plate type: {plate_type}. "
                         f"Valid types: {list(PLATE_CONFIGS.keys())}")

    config = PLATE_CONFIGS[plate_type]
    renderer = LAYOUT_RENDERERS[config["layout"]]
    img, bboxes = renderer(d3, d4, province, config)

    metadata = {
        "d3": d3,
        "d4": d4,
        "province": province,
        "plate_type": plate_type,
        "width": config["width"],
        "height": config["height"],
        "full_text": f"{d3} {d4} {province}",
    }

    return img, bboxes, metadata


def generate_random_plate(
    plate_type: Optional[str] = None,
) -> Tuple[Image.Image, List[Dict], dict]:
    """
    Generate a random valid Thai license plate.

    Args:
        plate_type: Specific type or None for random selection.

    Returns:
        (plate_image, char_bboxes, metadata)
    """
    if plate_type is None:
        plate_type = random.choice(list(PLATE_CONFIGS.keys()))

    config = PLATE_CONFIGS[plate_type]
    d3_min, d3_max = config["d3_len"]
    d4_min, d4_max = config["d4_len"]

    d3_len = random.randint(d3_min, d3_max)
    d4_len = random.randint(d4_min, d4_max)

    d3 = _generate_random_d3(d3_len)
    d4 = _generate_random_d4(d4_len)
    province = random.choice(PROVINCES)

    return render_plate(plate_type, d3, d4, province)


def get_plate_types() -> List[Tuple[str, str]]:
    """Returns list of (type_id, display_name) tuples."""
    return [(k, v["display_name"]) for k, v in PLATE_CONFIGS.items()]
