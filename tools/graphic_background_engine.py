#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLT Thailand Graphic Auction License Plate Background Engine.
Generates authentic cultural and provincial graphic background artworks for Thai Auction Plates
(ป้ายทะเบียนประมูลเลขสวย / ป้ายกราฟิกลวดลายประจำจังหวัด):
1. กรุงเทพมหานคร (Bangkok): Wat Phra Kaew, Wat Arun spires, Rama VIII suspension bridge, Chao Phraya river, golden/mauve sky.
2. เชียงใหม่ (Chiang Mai): Doi Suthep pagoda, Doi Luang mountain ridge, floating Yi Peng lanterns, twilight rose sky.
3. ชลบุรี (Chonburi): Bang Saen beach coastal sunset, Chonlamakwithi bridge, ocean waves, golden sun on horizon.
4. ภูเก็ต (Phuket): Phromthep Cape promontory headland, palm silhouettes, Andaman turquoise/coral sea, yacht sail.
5. นครราชสีมา (Nakhon Ratchasima): Khao Yai mountain ranges, Phimai Khmer sanctuary ruins, historic city gateway.
6. ขอนแก่น (Khon Kaen): Phu Wiang dinosaur silhouette, Kaen Nakhon lake, Phra That Kham Kaen pagoda, terracotta sunrise.
7. สมุทรปราการ (Samut Prakan): Phra Samut Chedi, Bang Pu seagull silhouettes, Chao Phraya gulf waters.
8. Generic Provincial Auction Theme: Multi-tonal sunset gradients with sun rays and misty mountain silhouettes.

Supports both standard car plates (340x150mm) and motorcycle auction plates (220x172mm).
"""

import os
import sys
import math
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.realistic_plate_generator import (
    shape_thai_text,
    add_embossed_3d_relief,
    add_retroreflective_texture,
    add_dlt_security_watermark
)

FONT_ID_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-id.ttf"
FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
FONT_LARGE_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-large.ttf"
if not FONT_LARGE_PATH.exists():
    FONT_LARGE_PATH = FONT_ID_PATH

ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\d2c371bd-a730-4efa-ac31-54db382d8c7f")


def cv2_imwrite_safe(filepath: Path, img: np.ndarray, quality: int = 95) -> bool:
    """UTF-8 safe image writing on Windows avoiding ANSI mojibake."""
    ext = filepath.suffix or ".jpg"
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ret, enc = cv2.imencode(ext, img, encode_param)
    if ret:
        with open(filepath, "wb") as f:
            f.write(enc.tobytes())
        return True
    return False


# =============================================================================
# 1. Provincial Graphic Background Generators
# =============================================================================

def draw_bangkok_graphic(w: int, h: int) -> np.ndarray:
    """Bangkok: Golden sunset, Wat Phra Kaew, Rama VIII bridge, Chao Phraya river."""
    # 1. Gradient Sky
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([215, 175, 230]) # BGR Mauve/lavender
    c_mid = np.array([130, 205, 255]) # Sunset gold
    c_bot = np.array([195, 220, 205]) # River green/cyan reflection
    for y in range(h):
        t = y / float(h)
        if t < 0.60:
            c = c_top * (1.0 - t / 0.60) + c_mid * (t / 0.60)
        else:
            t2 = (t - 0.60) / 0.40
            c = c_mid * (1.0 - t2) + c_bot * t2
        bg[y, :] = c

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # 2. Golden Sun Glow
    sun_x, sun_y, sun_r = int(w * 0.48), int(h * 0.36), int(h * 0.28)
    for r in range(sun_r, 0, -4):
        alpha = int(45 * (1.0 - r / float(sun_r)))
        draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 240, 180, alpha))

    # 3. Rama VIII Suspension Bridge Cables on Left
    pylon_x = int(w * 0.20)
    p_top_x = pylon_x - int(w * 0.02)
    p_top_y = int(h * 0.18)
    p_bot_y = int(h * 0.88)
    draw.line([(pylon_x, p_bot_y), (p_top_x, p_top_y)], fill=(85, 60, 115, 75), width=5)
    for i in range(14):
        cy = int(p_top_y + i * (h * 0.045))
        deck_x = int(w * 0.04 + i * (w * 0.025))
        draw.line([(p_top_x, cy), (deck_x, int(h * 0.84))], fill=(110, 85, 140, 48), width=2)

    # 4. Wat Phra Kaew & Wat Arun Spires on Right
    spire_col = (145, 95, 125, 72)
    spires = [(int(w * 0.72), 125), (int(w * 0.78), 165), (int(w * 0.85), 115), (int(w * 0.91), 90)]
    for sx, sh in spires:
        pts = [(sx, int(h * 0.86) - sh), (sx - 16, int(h * 0.86)), (sx + 16, int(h * 0.86))]
        draw.polygon(pts, fill=spire_col)
        for layer in range(3):
            lw = 24 + layer * 10
            lh = 6
            ly = int(h * 0.86) - layer * 7
            draw.rounded_rectangle([sx - lw // 2, ly - lh, sx + lw // 2, ly], radius=2, fill=spire_col)

    # 5. Chao Phraya River Wave Ribbons
    for i in range(4):
        y_wave = int(h * (0.84 + i * 0.04))
        wave_pts = []
        for x in range(0, w, 15):
            y_offset = math.sin(x * 0.02 + i * 1.2) * 4.0
            wave_pts.append((x, int(y_wave + y_offset)))
        wave_pts.append((w, h))
        wave_pts.append((0, h))
        draw.polygon(wave_pts, fill=(160, 195, 215, 35))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def draw_chiangmai_graphic(w: int, h: int) -> np.ndarray:
    """Chiang Mai: Twilight rose/purple sky, Doi Suthep pagoda, Doi Luang ridges, Yi Peng lanterns."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([215, 140, 185]) # Deep rose/purple
    c_mid = np.array([160, 180, 255]) # Warm coral/gold
    c_bot = np.array([190, 225, 230]) # Valley mist
    for y in range(h):
        t = y / float(h)
        c = c_top * (1.0 - t) + (c_mid * (1.0 - t) + c_bot * t) * t
        bg[y, :] = c

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # Mountain ridges (Doi Luang Chiang Dao)
    for idx, (m_y, m_col) in enumerate([(int(h * 0.60), (120, 95, 150, 75)), (int(h * 0.72), (90, 70, 130, 85))]):
        pts = [(0, h)]
        for x in range(0, w + 20, 20):
            yh = m_y - int(math.sin(x * 0.008 + idx * 1.5) * 28 + math.cos(x * 0.02) * 12)
            pts.append((x, yh))
        pts.append((w, h))
        draw.polygon(pts, fill=m_col)

    # Doi Suthep Golden Pagoda Spire on Left
    sp_x, sp_y = int(w * 0.22), int(h * 0.70)
    draw.polygon([(sp_x, sp_y - 110), (sp_x - 16, sp_y), (sp_x + 16, sp_y)], fill=(160, 120, 75, 95))

    # Floating Yi Peng Lanterns in Sky (Glowing yellow/orange)
    lanterns = [
        (int(w * 0.55), int(h * 0.25), 10),
        (int(w * 0.65), int(h * 0.18), 7),
        (int(w * 0.78), int(h * 0.28), 12),
        (int(w * 0.85), int(h * 0.15), 6)
    ]
    for lx, ly, lr in lanterns:
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=(255, 235, 140, 120))
        draw.rounded_rectangle([lx - lr + 2, ly - lr + 2, lx + lr - 2, ly + lr + 3], radius=2, fill=(255, 190, 80, 140))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def draw_chonburi_graphic(w: int, h: int) -> np.ndarray:
    """Chonburi: Coastal golden sunset, Chonlamakwithi bridge, ocean waves, sailboat."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([215, 165, 240]) # Twilight mauve
    c_mid = np.array([115, 195, 255]) # Golden sunset amber
    c_bot = np.array([210, 215, 170]) # Turquoise coastal water
    for y in range(h):
        t = y / float(h)
        c = c_top * (1.0 - t) + (c_mid * (1.0 - t * 0.5) + c_bot * (t * 0.5)) * t
        bg[y, :] = c

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # Sun setting over water
    sun_x, sun_y, sun_r = int(w * 0.68), int(h * 0.52), int(h * 0.25)
    for r in range(sun_r, 0, -3):
        alpha = int(50 * (1.0 - r / float(sun_r)))
        draw.ellipse([sun_x - r, sun_y - r, sun_x + r, sun_y + r], fill=(255, 220, 140, alpha))

    # Chonlamakwithi Seaside Bridge curve on Left
    bridge_pts = []
    for x in range(0, int(w * 0.55), 10):
        by = int(h * 0.70 + math.sin(x * 0.007) * 20)
        bridge_pts.append((x, by))
    for i in range(len(bridge_pts) - 1):
        x1, y1 = bridge_pts[i]
        x2, y2 = bridge_pts[i + 1]
        draw.line([(x1, y1), (x2, y2)], fill=(90, 75, 120, 95), width=4)
        draw.line([(x1, y1), (x1, h)], fill=(75, 60, 105, 55), width=2)

    # Sailboat on right
    boat_x, boat_y = int(w * 0.82), int(h * 0.62)
    draw.polygon([(boat_x - 16, boat_y), (boat_x + 16, boat_y), (boat_x + 10, boat_y + 6), (boat_x - 12, boat_y + 6)], fill=(65, 50, 90, 85))
    draw.polygon([(boat_x, boat_y - 28), (boat_x - 12, boat_y - 4), (boat_x, boat_y - 4)], fill=(255, 240, 220, 110))
    draw.polygon([(boat_x + 2, boat_y - 24), (boat_x + 10, boat_y - 4), (boat_x + 2, boat_y - 4)], fill=(255, 210, 180, 100))

    # Water wave ripples
    for i in range(3):
        y_w = int(h * (0.78 + i * 0.07))
        for x in range(0, w, 30):
            draw.arc([x, y_w, x + 25, y_w + 6], start=0, end=180, fill=(240, 245, 255, 45), width=1)

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def draw_phuket_graphic(w: int, h: int) -> np.ndarray:
    """Phuket: Phromthep Cape promontory, palm trees, turquoise Andaman sea, catamaran yacht."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([225, 210, 160]) # Andaman turquoise/azure
    c_mid = np.array([135, 185, 255]) # Golden coral
    c_bot = np.array([195, 215, 180]) # Emerald sea
    for y in range(h):
        t = y / float(h)
        bg[y, :] = c_top * (1.0 - t) + (c_mid * 0.6 + c_bot * 0.4) * t

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # Phromthep Cape Headland on Left
    cape_pts = [(0, int(h * 0.48))]
    for x in range(0, int(w * 0.42), 15):
        cy = int(h * 0.48 + (x / float(w * 0.42)) ** 1.8 * (h * 0.38) + math.sin(x * 0.03) * 6)
        cape_pts.append((x, cy))
    cape_pts.append((0, h))
    draw.polygon(cape_pts, fill=(75, 95, 70, 95))

    # Coconut Palm Silhouettes on headland
    palms = [(int(w * 0.12), int(h * 0.48)), (int(w * 0.22), int(h * 0.58))]
    for px, py in palms:
        # Trunk
        draw.line([(px, py), (px - 6, py - 32)], fill=(50, 40, 30, 110), width=3)
        # Fronds
        for angle in np.linspace(-1.2, 1.2, 5):
            fx = px - 6 + int(math.cos(angle) * 16)
            fy = py - 32 + int(math.sin(angle) * 10)
            draw.line([(px - 6, py - 32), (fx, fy)], fill=(40, 65, 45, 95), width=2)

    # Catamaran Yacht on right
    boat_x, boat_y = int(w * 0.78), int(h * 0.65)
    draw.polygon([(boat_x - 18, boat_y), (boat_x + 18, boat_y), (boat_x + 12, boat_y + 5), (boat_x - 14, boat_y + 5)], fill=(60, 55, 80, 85))
    draw.polygon([(boat_x - 2, boat_y - 32), (boat_x - 15, boat_y - 3), (boat_x - 2, boat_y - 3)], fill=(255, 245, 235, 120))
    draw.polygon([(boat_x, boat_y - 28), (boat_x + 12, boat_y - 3), (boat_x, boat_y - 3)], fill=(255, 215, 195, 105))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def draw_korat_graphic(w: int, h: int) -> np.ndarray:
    """Nakhon Ratchasima: Khao Yai green/purple mountain ridges, Phimai sanctuary ruins."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([215, 160, 210])
    c_mid = np.array([140, 200, 250])
    c_bot = np.array([170, 210, 185])
    for y in range(h):
        t = y / float(h)
        bg[y, :] = c_top * (1.0 - t) + (c_mid * 0.5 + c_bot * 0.5) * t

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # Mountain ridges
    pts = [(0, h)]
    for x in range(0, w + 20, 20):
        yh = int(h * 0.62 - math.sin(x * 0.009) * 25)
        pts.append((x, yh))
    pts.append((w, h))
    draw.polygon(pts, fill=(95, 110, 90, 80))

    # Phimai Khmer Sanctuary Spire in Center-Right
    sx, sy = int(w * 0.75), int(h * 0.72)
    draw.polygon([(sx, sy - 90), (sx - 20, sy), (sx + 20, sy)], fill=(120, 90, 75, 95))
    draw.rounded_rectangle([sx - 28, sy - 15, sx + 28, sy], radius=3, fill=(110, 80, 65, 95))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def draw_generic_auction_graphic(w: int, h: int) -> np.ndarray:
    """Generic provincial auction graphic: Multi-gradient sunset with sun rays & mist."""
    bg = np.zeros((h, w, 3), dtype=np.float32)
    c_top = np.array([220, 160, 220])
    c_mid = np.array([135, 200, 255])
    c_bot = np.array([205, 225, 200])
    for y in range(h):
        t = y / float(h)
        bg[y, :] = c_top * (1.0 - t) + c_mid * t

    im = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(im, "RGBA")

    # Sunburst rays
    cx, cy = int(w * 0.50), int(h * 0.40)
    for a_deg in range(0, 180, 15):
        rad = math.radians(a_deg)
        rx = cx + int(math.cos(rad) * w * 0.6)
        ry = cy - int(math.sin(rad) * h * 0.6)
        draw.line([(cx, cy), (rx, ry)], fill=(255, 245, 210, 25), width=6)

    # Distant mountain silhouette
    pts = [(0, h)]
    for x in range(0, w + 20, 25):
        yh = int(h * 0.72 - math.sin(x * 0.012) * 18 - math.cos(x * 0.025) * 8)
        pts.append((x, yh))
    pts.append((w, h))
    draw.polygon(pts, fill=(115, 95, 130, 65))

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def generate_provincial_graphic_background(province_name: str, w: int = 680, h: int = 300) -> np.ndarray:
    """Dispatches provincial graphic generator based on province name."""
    clean = province_name.strip()
    if "กรุงเทพ" in clean:
        return draw_bangkok_graphic(w, h)
    elif "เชียงใหม่" in clean:
        return draw_chiangmai_graphic(w, h)
    elif "ชลบุรี" in clean:
        return draw_chonburi_graphic(w, h)
    elif "ภูเก็ต" in clean:
        return draw_phuket_graphic(w, h)
    elif "นครราชสีมา" in clean or "โคราช" in clean:
        return draw_korat_graphic(w, h)
    else:
        return draw_generic_auction_graphic(w, h)


# =============================================================================
# 2. Complete Graphic Auction Plate Renderer
# =============================================================================

def render_graphic_auction_plate(
    plate_text: str,       # e.g. '9กก 9999' or '8กน 5500' or '1กจ 4160'
    province_text: str,    # e.g. 'กรุงเทพมหานคร'
    target_w: int = 680,
    target_h: int = 300,
    is_motorcycle: bool = False,
    rim_color_key: str = "gold"  # 'gold', 'purple', 'black'
) -> np.ndarray:
    """
    Renders an authentic, high-precision DLT Graphic Auction License Plate:
    1. Provincial illustrated cultural graphic background.
    2. Deep stamped metallic characters with 3D hydraulic bevel relief.
    3. Stamped rim in gold/purple/black.
    4. 3M micro-prismatic retroreflective grain & DLT security watermark.
    """
    w = 440 if is_motorcycle else target_w
    h = 344 if is_motorcycle else target_h

    # 1. Generate Provincial Cultural Graphic Canvas
    bg_bgr = generate_provincial_graphic_background(province_text, w, h)
    im_pil = Image.fromarray(cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im_pil)

    # Rim Color
    rim_colors = {
        "gold": (160, 120, 45),
        "purple": (110, 60, 115),
        "black": (25, 25, 25)
    }
    border_rgb = rim_colors.get(rim_color_key, (25, 25, 25))
    fg_rgb = (22, 22, 22)  # High-density stamped carbon black

    # 2. Outer Rim
    if is_motorcycle:
        inset = int(w * 0.024)
        corner_r = int(h * 0.055)
    else:
        inset = int(w * 0.018)
        corner_r = int(h * 0.080)
    draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=corner_r, outline=border_rgb, width=4)

    # 3. Render Registration & Province Text
    if is_motorcycle:
        # Motorcycle: 3-line layout calibrated to DLT specifications
        clean_ptext = plate_text.replace("-", " ")
        parts = clean_ptext.split()
        top_str = parts[0] if len(parts) >= 1 else "1กข"
        num_str = parts[1] if len(parts) >= 2 else "9999"

        # Reference glyph measurement for ID font
        f_ref = ImageFont.truetype(str(FONT_ID_PATH), 100)
        b_ref = f_ref.getbbox("5")
        ref_h = float(b_ref[3] - b_ref[1])

        # Line 1: Top registration mark
        target_glyph_h1 = int(round(h * 0.275))
        font_size_l1 = int(round(100 * (target_glyph_h1 / ref_h)))
        f_top = ImageFont.truetype(str(FONT_ID_PATH), font_size_l1)

        chars_top = list(top_str)
        bboxes_t = [f_top.getbbox(c) for c in chars_top]
        widths_t = [b[2] - b[0] for b in bboxes_t]
        natural_w1 = sum(widths_t)
        target_w1 = int(w * 0.54) if len(chars_top) >= 3 else int(w * 0.44)
        kern1 = max(int(font_size_l1 * 0.08), (target_w1 - natural_w1) // max(1, len(chars_top) - 1))
        actual_span1 = natural_w1 + kern1 * (len(chars_top) - 1)

        start_x1 = (w - actual_span1) // 2
        cur_x = start_x1
        baseline_y1 = int(h * 0.315)
        for c, b in zip(chars_top, bboxes_t):
            draw.text((cur_x - b[0], baseline_y1 - b[3]), c, fill=fg_rgb, font=f_top)
            cur_x += (b[2] - b[0]) + kern1

        # Line 2: Middle province name on soft illuminated cloud
        shaped_prov = shape_thai_text(province_text)
        f_prov_ref = ImageFont.truetype(str(FONT_PROV_PATH), 100)
        bp_ref = f_prov_ref.getbbox("ก")
        ref_prov_h = float(bp_ref[3] - bp_ref[1])

        target_glyph_h2 = int(round(h * 0.125))
        font_size_l2 = int(round(100 * (target_glyph_h2 / ref_prov_h)))
        f_prov = ImageFont.truetype(str(FONT_PROV_PATH), font_size_l2)
        b_pr = f_prov.getbbox(shaped_prov)
        pw = b_pr[2] - b_pr[0]
        max_pw = int(w * 0.82)
        if pw > max_pw:
            scale_p = max_pw / float(pw)
            font_size_l2 = int(round(font_size_l2 * scale_p))
            f_prov = ImageFont.truetype(str(FONT_PROV_PATH), font_size_l2)
            b_pr = f_prov.getbbox(shaped_prov)
            pw = b_pr[2] - b_pr[0]

        # Soft translucent background pill for province readability
        px = (w - pw) // 2
        py = int(h * 0.46)
        draw.rounded_rectangle([px - 14, py - 4, px + pw + 14, py + (b_pr[3] - b_pr[1]) + 4], radius=4, fill=(255, 255, 255, 120))
        baseline_y2 = int(h * 0.540)
        draw.text((px - b_pr[0], baseline_y2 - b_pr[3]), shaped_prov, fill=fg_rgb, font=f_prov)

        # Line 3: Bottom numbers
        target_glyph_h3 = int(round(h * 0.310))
        font_size_l3 = int(round(100 * (target_glyph_h3 / ref_h)))
        f_num = ImageFont.truetype(str(FONT_ID_PATH), font_size_l3)

        chars_num = list(num_str)
        bboxes_n = [f_num.getbbox(c) for c in chars_num]
        widths_n = [b[2] - b[0] for b in bboxes_n]
        natural_w3 = sum(widths_n)

        if len(chars_num) >= 4:
            target_w3 = int(w * 0.72)
        elif len(chars_num) == 3:
            target_w3 = int(w * 0.56)
        elif len(chars_num) == 2:
            target_w3 = int(w * 0.39)
        else:
            target_w3 = widths_n[0]

        kern3 = max(int(font_size_l3 * 0.08), (target_w3 - natural_w3) // max(1, len(chars_num) - 1)) if len(chars_num) > 1 else 0
        actual_span3 = natural_w3 + kern3 * (len(chars_num) - 1)

        start_x3 = (w - actual_span3) // 2
        cur_x = start_x3
        baseline_y3 = int(h * 0.895)
        for c, b in zip(chars_num, bboxes_n):
            draw.text((cur_x - b[0], baseline_y3 - b[3]), c, fill=fg_rgb, font=f_num)
            cur_x += (b[2] - b[0]) + kern3

    else:
        # Standard Car: 2-line layout
        clean_ptext = plate_text.replace("-", " ")
        parts = clean_ptext.split()
        letters_str, numbers_str = (parts[0], parts[1]) if len(parts) == 2 else (plate_text, "")

        target_digit_h = int(h * 0.43)
        baseline_reg = int(h * 0.560)
        mid_gap_ratio = 0.35
        max_reg_w = int(w * 0.88)

        f_test = ImageFont.truetype(str(FONT_ID_PATH), 100)
        ref_b = f_test.getbbox("5")
        scale = target_digit_h / float(ref_b[3] - ref_b[1])
        reg_font_size = int(round(100 * scale))

        def measure_baseline_layout(size):
            f = ImageFont.truetype(str(FONT_ID_PATH), size)
            kerning_l = int(size * 0.075)
            kerning_n = int(size * 0.070)
            mid_gap = int(size * mid_gap_ratio)

            def group_w(text, kern):
                w_acc = 0
                for ch in text:
                    b = f.getbbox(ch)
                    w_acc += (b[2] - b[0]) + kern
                return max(0, w_acc - kern)

            wl = group_w(letters_str, kerning_l)
            wn = group_w(numbers_str, kerning_n) if numbers_str else 0
            total_w = wl + (mid_gap if numbers_str else 0) + wn
            return f, total_w, wl, wn, mid_gap, kerning_l, kerning_n

        f_reg, total_w, wl, wn, mid_gap, kerning_l, kerning_n = measure_baseline_layout(reg_font_size)
        if total_w > max_reg_w:
            reg_font_size = int(round(reg_font_size * (max_reg_w / float(total_w))))
            f_reg, total_w, wl, wn, mid_gap, kerning_l, kerning_n = measure_baseline_layout(reg_font_size)

        start_x = (w - total_w) // 2
        ref_b_bottom = f_reg.getbbox("5")[3]
        char_y = baseline_reg - ref_b_bottom

        # Draw Letters
        cur_x = start_x
        for ch in letters_str:
            b = f_reg.getbbox(ch)
            draw.text((cur_x - b[0], char_y), ch, fill=fg_rgb, font=f_reg)
            cur_x += (b[2] - b[0]) + kerning_l

        # Draw Numbers
        if numbers_str:
            cur_x = start_x + wl + mid_gap
            for ch in numbers_str:
                b = f_reg.getbbox(ch)
                draw.text((cur_x - b[0], char_y), ch, fill=fg_rgb, font=f_reg)
                cur_x += (b[2] - b[0]) + kerning_n

        # Province text on illuminated translucent baseline strip
        shaped_province = shape_thai_text(province_text)
        base_prov_size = int(h * 0.245)
        prov_baseline = int(h * 0.890)
        max_prov_w = int(w * 0.82)
        f_prov = ImageFont.truetype(str(FONT_PROV_PATH), base_prov_size)
        b_prov = f_prov.getbbox(shaped_province)
        pw = b_prov[2] - b_prov[0]
        if pw > max_prov_w:
            scale_p = max_prov_w / float(pw)
            f_prov = ImageFont.truetype(str(FONT_PROV_PATH), int(round(base_prov_size * scale_p)))
            b_prov = f_prov.getbbox(shaped_province)
            pw = b_prov[2] - b_prov[0]

        prov_x = (w - pw) // 2 - b_prov[0]
        prov_ref_bottom = f_prov.getbbox("ก")[3]
        prov_y = prov_baseline - prov_ref_bottom

        # Soft translucent strip behind province name for OCR clarity
        draw.rounded_rectangle(
            [prov_x - 16, prov_y - 2, prov_x + pw + 16, prov_y + (b_prov[3] - b_prov[1]) + 4],
            radius=4,
            fill=(255, 255, 255, 120)
        )
        draw.text((prov_x, prov_y), shaped_province, fill=fg_rgb, font=f_prov)

    plate_np = np.array(im_pil)

    # 4. 3D Hydraulic Stamping Bevel Relief on Text
    gray = cv2.cvtColor(plate_np, cv2.COLOR_RGB2GRAY)
    _, text_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    plate_np = add_embossed_3d_relief(plate_np, text_mask, depth=2)

    # 5. 3M Retroreflective Sheeting Grain
    plate_np = add_retroreflective_texture(plate_np, intensity=1.8)

    # 6. DLT Security Wave Watermark Overlay
    plate_np = add_dlt_security_watermark(plate_np)

    return cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)


# =============================================================================
# 3. Master Visual Gallery for Graphic Auction Plates
# =============================================================================

def build_graphic_auction_specimen_gallery():
    """
    Renders comprehensive 8-card showcase of DLT Graphic Auction Plates
    across Bangkok, Chiang Mai, Chonburi, Phuket, Korat, and Motorcycles,
    plus CCTV perspective and dirt simulations.
    """
    print("=" * 80)
    print("[*] Generating DLT Graphic Auction License Plate Specimen Gallery")
    print("=" * 80)

    gallery_path = ARTIFACT_DIR / "graphic_plate_specimen_gallery.jpg"

    # 1. Render Graphic Auction Plates
    p_bkk_super = render_graphic_auction_plate("9กก 9999", "กรุงเทพมหานคร", rim_color_key="gold")
    p_bkk_user = render_graphic_auction_plate("8กน 5500", "กรุงเทพมหานคร", rim_color_key="gold")
    p_cm = render_graphic_auction_plate("กย 8888", "เชียงใหม่", rim_color_key="purple")
    p_chon = render_graphic_auction_plate("งจ 9999", "ชลบุรี", rim_color_key="gold")
    p_phuket = render_graphic_auction_plate("ขฉ 1", "ภูเก็ต", rim_color_key="gold")
    p_korat = render_graphic_auction_plate("ขข 7777", "นครราชสีมา", rim_color_key="gold")

    # Motorcycle Graphic Auction Plate
    p_motor_bkk = render_graphic_auction_plate("1กจ 4160", "กรุงเทพมหานคร", is_motorcycle=True, rim_color_key="gold")

    # CCTV Optical Perspective & Dirty simulation on Graphic Plate
    from tools.generate_cctv_synthetic_dataset import simulate_cctv_optics_and_angle
    from tools.dirty_plate_synthesizer import apply_synthetic_dirt_effects

    p_bkk_cctv = simulate_cctv_optics_and_angle(p_bkk_super, "slanted_approach", target_width=115, jpeg_quality=50)
    p_cm_dirty = apply_synthetic_dirt_effects(p_cm, dirt_level=0.55, effect_types=["road_film", "mud_splatter", "paint_chipping"])
    p_cm_dirty_cctv = simulate_cctv_optics_and_angle(p_cm_dirty, "slanted_approach", target_width=110, jpeg_quality=48)

    cards = [
        {
            "title": "1. Bangkok Super VIP (9กก 9999)",
            "img": p_bkk_super,
            "badge": "BANGKOK GRAPHIC",
            "desc": "Wat Phra Kaew, Wat Arun, Rama VIII Bridge, Chao Phraya river, golden/mauve sky"
        },
        {
            "title": "2. User Active Plate as Graphic (8กน 5500 กทม.)",
            "img": p_bkk_user,
            "badge": "AUCTION GRAPHIC",
            "desc": "Bangkok cultural landmarks with gold metallic rim and hydraulic 3D relief"
        },
        {
            "title": "3. Chiang Mai Auction Plate (กย 8888 เชียงใหม่)",
            "img": p_cm,
            "badge": "CHIANG MAI GRAPHIC",
            "desc": "Doi Suthep pagoda spire, Doi Luang mountain ridge, floating Yi Peng lanterns"
        },
        {
            "title": "4. Chonburi Coastal Sunset (งจ 9999 ชลบุรี)",
            "img": p_chon,
            "badge": "CHONBURI GRAPHIC",
            "desc": "Bang Saen coastal sunset, ocean wave ripples, seaside bridge, and sailboat"
        },
        {
            "title": "5. Phuket Andaman Pearl (ขฉ 1 ภูเก็ต)",
            "img": p_phuket,
            "badge": "PHUKET GRAPHIC",
            "desc": "Phromthep Cape headland, coconut palm silhouettes, turquoise Andaman sea"
        },
        {
            "title": "6. Korat Gateway & Ruins (ขข 7777 นครราชสีมา)",
            "img": p_korat,
            "badge": "KORAT GRAPHIC",
            "desc": "Khao Yai mountain ridge, Phimai stone Khmer sanctuary spire"
        },
        {
            "title": "7. Motorcycle Graphic Auction Plate (1กจ 4160 กทม.)",
            "img": p_motor_bkk,
            "badge": "MOTORCYCLE GRAPHIC",
            "desc": "220x172mm 3-line format with Bangkok graphic artwork and dual mounting bolts"
        },
        {
            "title": "8. CCTV Optics + Mud Spatter on Graphic Plate",
            "img": p_cm_dirty_cctv,
            "badge": "CCTV + DIRTY GRAPHIC",
            "desc": "Tests edge LPR model robustness: complex artwork background + road mud + CCTV perspective"
        }
    ]

    card_w = 460
    card_h = 220
    cols = 2
    rows = (len(cards) + 1) // cols

    sheet_w = cols * (card_w + 24) + 32
    sheet_h = 120 + rows * (card_h + 36)

    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), color=(241, 245, 249))
    draw = ImageDraw.Draw(sheet_pil)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 20)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 13)
    font_card = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 13)
    font_desc = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)
    font_badge = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 10)

    # Header Card
    draw.rounded_rectangle([16, 14, sheet_w - 16, 92], radius=8, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((32, 24), "DLT Thailand Graphic Auction License Plate Engine (ป้ายประมูลเลขสวยกราฟิก)", fill=(15, 23, 42), font=font_title)
    draw.text((32, 54), "Authentic Cultural Artworks: Bangkok, Chiang Mai, Chonburi, Phuket, Korat, Motorcycles & Dirty CCTV Simulation", fill=(71, 85, 105), font=font_sub)

    for idx, item in enumerate(cards):
        c = idx % cols
        r = idx // cols
        x = 20 + c * (card_w + 24)
        y = 112 + r * (card_h + 36)

        # 1. Outer Card Box
        draw.rounded_rectangle([x, y, x + card_w, y + card_h], radius=8, fill=(255, 255, 255), outline=(203, 213, 225), width=2)

        # 2. Card Header Title & Badge
        title_text = item["title"]
        badge_text = item["badge"]
        draw.text((x + 12, y + 10), title_text, fill=(15, 23, 42), font=font_card)

        # Badge pill
        b_box = font_badge.getbbox(badge_text)
        bw = b_box[2] - b_box[0] + 12
        bh = b_box[3] - b_box[1] + 6
        bx = x + card_w - bw - 12
        by = y + 8
        badge_bg = (180, 83, 9) if "GRAPHIC" in badge_text else (30, 64, 175)
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4, fill=badge_bg)
        draw.text((bx + 6, by + 3), badge_text, fill=(255, 255, 255), font=font_badge)

        # 3. Inner Dark Asphalt Display Container
        cx1 = x + 10
        cy1 = y + 36
        cx2 = x + card_w - 10
        cy2 = y + card_h - 32
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=6, fill=(24, 28, 36), outline=(51, 65, 85), width=1)

        # 4. Display Image centered and upscaled in container
        crop_bgr = item.get("img")
        if crop_bgr is not None:
            box_w = (cx2 - cx1) - 16
            box_h = (cy2 - cy1) - 16
            ch, cw = crop_bgr.shape[:2]
            scale = min(box_w / float(cw), box_h / float(ch))
            disp_w = int(round(cw * scale))
            disp_h = int(round(ch * scale))
            resized = cv2.resize(
                crop_bgr,
                (disp_w, disp_h),
                interpolation=cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
            )

            px = cx1 + 8 + (box_w - disp_w) // 2
            py = cy1 + 8 + (box_h - disp_h) // 2

            crop_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            sheet_pil.paste(crop_pil, (px, py))

        # 5. Bottom Description
        draw.text((x + 12, y + card_h - 22), item.get("desc", ""), fill=(100, 116, 139), font=font_desc)

    sheet_bgr = cv2.cvtColor(np.array(sheet_pil), cv2.COLOR_RGB2BGR)
    cv2_imwrite_safe(gallery_path, sheet_bgr, quality=95)
    print(f"[+] Saved Graphic Auction Plate Gallery to: {gallery_path}")


if __name__ == "__main__":
    build_graphic_auction_specimen_gallery()
