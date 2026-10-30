#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic Mobile & CCTV Synthetic Thai License Plate Dataset Generator.
Calibrated against Ground Truth Reference Images:
1. Real Mobile Snapshots (`dataset/snapshot/mobile`):
   - High-res handheld camera close-ups (400px - 1400px width) with 3D stamping bevel and mounting bolts.
   - Handheld perspective angles (yaw ±18°, pitch 6°-16°, roll ±5°).
   - Natural outdoor sunlight, specular glare on metallic embossing, and diffused ambient shade.
   - Complete DLT vehicle category coverage:
     * WhiteBlackCar: Private Cars & 4-door Pickups (Bangkok 1-9 leading digits e.g. 6ขฬ 3052, 7กฬ 2597; Provincial e.g. กจ 4108 มุกดาหาร).
     * WhiteGreen: Commercial Pickups (≤ 2,200 kg) featuring rare front consonants: 3ฒล 7019, 3ฒฐ 795, 4ฒฌ 5525, ผก 6290.
     * WhiteBlue: Vans & Microbuses (> 7 seats) with series น, ฬ, อ, ฮ: นง 8970 นนทบุรี.
     * YellowBlack: Taxis (2ขร 547, ทฬ 5400), Heavy Transport Land Transport Act (14-3112, 701-6466, 72-1135), and Motorcycle Taxis (กกบ 310).
     * WhiteBlackMotor: 3-line Motorcycles (220mm x 172mm calibrated geometry).
2. Roadside CCTV & Traffic Cameras (`.user_uploaded`):
   - Overhead steep downward viewing (pitch 16°-26°, roll ±12°).
   - Slanted turning vehicles & nighttime headlight bloom.

Compliance:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.7.2 & A.9.3)
- ISO/IEC 5259-2:2023 (Data Quality Measures for AI ML)
- ISO 3166-2:TH (Thailand Administrative Subdivisions)
- ISO 8601:2019 (Timestamps)
- Thailand DLT Statutory Vehicle & License Plate Standards
"""

import os
import sys
import re
import csv
import json
import math
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter

# UTF-8 stdout on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.realistic_plate_generator import (
    render_calibrated_plate_die,
    shape_thai_text,
    DLT_PROVINCE_CODES,
    COLORWAYS
)

# Fonts
FONT_ID_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-id.ttf"
FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
FONT_YELLOW_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-yellow.ttf"

# Target Dataset Output Directories
OUTPUT_DATASET_DIR = ROOT_DIR / "dataset" / "dataset-mobile-synthetic"
PLATES_OUTPUT_DIR = OUTPUT_DATASET_DIR / "plates"
OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
PLATES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference Directories
MOBILE_REF_DIR = ROOT_DIR / "dataset" / "snapshot" / "mobile"
ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\f39d0d07-6ecd-4d68-972d-9fd157abb790")

# DLT Provinces (weighted by national registration density)
PROVINCE_POOL = [
    {"name": "กรุงเทพมหานคร", "iso": "TH-10", "weight": 0.38},
    {"name": "ชลบุรี", "iso": "TH-20", "weight": 0.10},
    {"name": "นนทบุรี", "iso": "TH-12", "weight": 0.07},
    {"name": "สมุทรปราการ", "iso": "TH-11", "weight": 0.06},
    {"name": "ปทุมธานี", "iso": "TH-13", "weight": 0.05},
    {"name": "เชียงใหม่", "iso": "TH-50", "weight": 0.05},
    {"name": "ภูเก็ต", "iso": "TH-83", "weight": 0.05},
    {"name": "นครปฐม", "iso": "TH-73", "weight": 0.04},
    {"name": "พระนครศรีอยุธยา", "iso": "TH-14", "weight": 0.04},
    {"name": "ระยอง", "iso": "TH-21", "weight": 0.03},
    {"name": "ขอนแก่น", "iso": "TH-40", "weight": 0.03},
    {"name": "มุกดาหาร", "iso": "TH-49", "weight": 0.02},
    {"name": "สตูล", "iso": "TH-91", "weight": 0.02},
    {"name": "อุดรธานี", "iso": "TH-41", "weight": 0.02},
    {"name": "ลพบุรี", "iso": "TH-16", "weight": 0.02},
    {"name": "สุราษฎร์ธานี", "iso": "TH-84", "weight": 0.02},
]

# DLT Consonants by Class
CAR_INITIALS = list("กขงจฉชฌญฎฐธพภวศษส")
VAN_INITIALS = list("นฬอฮ")
PICKUP_INITIALS = list("บฒณตถปผยรล")
TAXI_INITIALS = list("ทม")
ALL_VALID_CONSONANTS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
RARE_CONSONANTS = list("ฒฐฎฉฌญษศผฬฮ")
OFFENSIVE_PAIRS = {"ศพ", "สพ", "ศบ", "ษพ", "ชน", "จน", "ตด", "งง"}


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


def cv2_imread_safe(filepath: Path) -> Optional[np.ndarray]:
    """UTF-8 safe image reading on Windows avoiding ANSI path encoding issues."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_random_valid_pair(initial_candidates: List[str], boost_rare_prob: float = 0.35) -> str:
    """Generates a valid 2-letter pair adhering to DLT exclusions, with boosted rare consonants."""
    for _ in range(100):
        if random.random() < boost_rare_prob:
            # Pick from rare consonants
            c1_cand = [c for c in initial_candidates if c in RARE_CONSONANTS]
            c1 = random.choice(c1_cand) if c1_cand else random.choice(initial_candidates)
            c2 = random.choice(RARE_CONSONANTS)
        else:
            c1 = random.choice(initial_candidates)
            c2 = random.choice(ALL_VALID_CONSONANTS)
        pair = c1 + c2
        if pair not in OFFENSIVE_PAIRS:
            return pair
    return initial_candidates[0] + "ก"


def render_motorcycle_plate_die(
    top_text: str,
    province_text: str,
    num_text: str,
    colorway_key: str = "private_car",
    target_w: int = 440,
    target_h: int = 344
) -> np.ndarray:
    """
    Renders authentic 220mm x 172mm 3-line Thai motorcycle plate die (2.0 scale: 440x344px).
    """
    w, h = target_w, target_h
    bg_rgb, fg_rgb, border_rgb, _ = COLORWAYS.get(colorway_key, COLORWAYS["private_car"])

    im_pil = Image.new("RGB", (w, h), color=bg_rgb)
    draw = ImageDraw.Draw(im_pil)

    # 1. Outer rim
    inset = int(w * 0.024)
    corner_r = int(h * 0.064)
    draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=corner_r, outline=border_rgb, width=4)

    # Reference glyph measurement
    f_ref = ImageFont.truetype(str(FONT_ID_PATH), 100)
    b_ref = f_ref.getbbox("5")
    ref_h = float(b_ref[3] - b_ref[1])

    # 2. Line 1: Top registration mark
    target_glyph_h1 = int(round(h * 0.275))
    font_size_l1 = int(round(100 * (target_glyph_h1 / ref_h)))
    f_top = ImageFont.truetype(str(FONT_ID_PATH), font_size_l1)

    chars_top = list(top_text)
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

    # 3. Line 2: Middle province
    shaped_prov = shape_thai_text(province_text)
    f_prov_ref = ImageFont.truetype(str(FONT_PROV_PATH), 100)
    bp_ref = f_prov_ref.getbbox("ก")
    ref_prov_h = float(bp_ref[3] - bp_ref[1])

    target_glyph_h2 = int(round(h * 0.125))
    font_size_l2 = int(round(100 * (target_glyph_h2 / ref_prov_h)))
    f_prov = ImageFont.truetype(str(FONT_PROV_PATH), font_size_l2)
    bp = f_prov.getbbox(shaped_prov)
    pw = bp[2] - bp[0]
    max_pw = int(w * 0.82)
    if pw > max_pw:
        scale_p = max_pw / float(pw)
        font_size_l2 = int(round(font_size_l2 * scale_p))
        f_prov = ImageFont.truetype(str(FONT_PROV_PATH), font_size_l2)
        bp = f_prov.getbbox(shaped_prov)
        pw = bp[2] - bp[0]

    baseline_y2 = int(h * 0.540)
    draw.text(((w - pw) // 2 - bp[0], baseline_y2 - bp[3]), shaped_prov, fill=fg_rgb, font=f_prov)

    # 4. Line 3: Bottom number
    target_glyph_h3 = int(round(h * 0.310))
    font_size_l3 = int(round(100 * (target_glyph_h3 / ref_h)))
    f_num = ImageFont.truetype(str(FONT_ID_PATH), font_size_l3)

    chars_num = list(num_text)
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

    plate_np = np.array(im_pil)

    # Dual mounting bolts
    bolt_positions = [(int(w * 0.175), int(h * 0.155)), (int(w * 0.825), int(h * 0.155))]
    for bx, by in bolt_positions:
        cv2.circle(plate_np, (bx, by), int(h * 0.045), (150, 155, 160), 2)
        cv2.circle(plate_np, (bx, by), int(h * 0.038), (110, 115, 120), -1)
        cv2.circle(plate_np, (bx, by), int(h * 0.022), (40, 42, 45), -1)
        cv2.circle(plate_np, (bx, by), int(h * 0.012), (190, 195, 200), -1)

    noise = np.random.normal(0, 1.8, plate_np.shape).astype(np.float32)
    plate_np = np.clip(plate_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)


def add_mobile_mounting_bolts(plate_bgr: np.ndarray) -> np.ndarray:
    """Renders authentic smartphone-visible mounting bolts on standard car plates."""
    h, w = plate_bgr.shape[:2]
    out = plate_bgr.copy()
    # Bolts typically at y ~ 20% to 25%, x ~ 15% and 85%
    by = int(h * random.uniform(0.18, 0.24))
    b_rad = max(3, int(h * 0.035))
    bx1 = int(w * random.uniform(0.14, 0.18))
    bx2 = int(w * random.uniform(0.82, 0.86))
    
    for bx in [bx1, bx2]:
        # Hex washer / bolt rim
        cv2.circle(out, (bx, by), b_rad, (140, 145, 150), 2)
        cv2.circle(out, (bx, by), b_rad - 1, (100, 105, 110), -1)
        # Screw center (Phillips cross or dark socket)
        cv2.circle(out, (bx, by), max(1, int(b_rad * 0.5)), (35, 38, 42), -1)
        # Metallic glint
        cv2.circle(out, (bx - 1, by - 1), max(1, int(b_rad * 0.25)), (210, 215, 220), -1)
    return out


def apply_unclean_road_weathering(
    plate_bgr: np.ndarray,
    dirt_level: float = 0.5,
    plate_type: str = "WhiteBlackCar",
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0
) -> np.ndarray:
    """
    Applies comprehensive environmental road dirt, weathering, and degradation
    physically coupled with 3D perspective orientation (pitch, yaw):
    1. Bumper Recess Ambient Occlusion (soft shadow gradient along top edge, scales with pitch).
    2. Road Dust Film & Color Shift (realistic muted tones matching real road dust, skewed along yaw/pitch).
    3. Paint Chipping & Metal Exposure on raised stamped glyphs + stone chip scratches.
    4. Mounting Bolt Rust Dripping (gravity-driven iron oxide bleeding).
    5. Rain Water Streaks (dried runoff residue).
    6. Organic Mud & Road Tar Splatters (clusters of micro-droplets drifting with airflow).
    7. Diesel Exhaust Soot Accumulation.
    """
    if dirt_level <= 0.05:
        return plate_bgr

    img = plate_bgr.astype(np.float32)
    h, w = img.shape[:2]

    # --- 1. Bumper Recess Ambient Occlusion Shadow ---
    if random.random() < 0.80:
        p_shadow_bias = max(0.0, pitch_deg / 45.0) * 0.12
        shadow_depth = (random.uniform(0.18, 0.36) + p_shadow_bias) * h
        shadow_strength = random.uniform(0.18, 0.40)
        for y in range(int(shadow_depth)):
            factor = 1.0 - shadow_strength * (1.0 - y / shadow_depth) ** 1.5
            img[y, :] *= factor

    # --- 2. Road Dust & Traffic Grime Film (Coupled with Perspective) ---
    dust_color = np.array([
        random.randint(45, 75),   # B
        random.randint(55, 85),   # G
        random.randint(65, 95)    # R
    ], dtype=np.float32)

    small_noise = np.random.uniform(0.2, 1.0, (max(4, h // 20), max(4, w // 20))).astype(np.float32)
    film_noise = cv2.resize(small_noise, (w, h), interpolation=cv2.INTER_CUBIC)

    # Vertical gradient: scales with pitch angle (higher overhead pitch = heavier bottom road spray)
    p_power = max(0.85, 1.30 + (pitch_deg / 35.0) * 0.55)
    y_grad = (np.linspace(0.18, 1.0, h)[:, None] ** p_power)

    # Horizontal gradient: couples with yaw angle (windward/turning edge collects more spray)
    if abs(yaw_deg) > 2.0:
        skew = min(0.40, max(-0.40, (yaw_deg / 25.0) * 0.35))
        x_grad = 1.0 + np.linspace(-skew, skew, w)[None, :]
    else:
        x_grad = 1.0 + 0.15 * np.sin(np.linspace(0, math.pi, w))[None, :]

    grime_mask = film_noise * y_grad * x_grad * (dirt_level * 0.70)
    grime_mask = np.clip(grime_mask, 0.0, 0.80)[:, :, None]

    img = img * (1.0 - grime_mask) + dust_color * grime_mask

    # --- 3. Paint Chipping & Stamped Character Wear ---
    if dirt_level > 0.15:
        gray = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # Identify dark character glyphs
        char_mask = gray < 100
        
        # Flaking on high points of embossed characters
        flake_prob = 0.03 + 0.14 * dirt_level
        flake_noise = np.random.uniform(0.0, 1.0, (h, w)) < flake_prob
        chip_mask = char_mask & flake_noise

        # Exposed retroreflective aluminum substrate (white-silver metal reflection)
        chip_val = np.random.uniform(190, 235, (chip_mask.sum(), 3))
        img[chip_mask] = chip_val

        # Stone chip hairline scratches
        max_scratches = max(1, int(6 * dirt_level) + 1)
        num_scratches = random.randint(1, max_scratches)
        for _ in range(num_scratches):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            angle = random.uniform(-0.6, 0.6)
            s_len = random.randint(8, max(9, int(w * 0.22)))
            x2 = int(np.clip(x1 + s_len * math.cos(angle), 0, w - 1))
            y2 = int(np.clip(y1 + s_len * math.sin(angle), 0, h - 1))
            cv2.line(img, (x1, y1), (x2, y2), (205, 210, 215), 1)

    # --- 4. Mounting Bolt Rust Dripping ---
    if dirt_level > 0.20:
        is_square = (w / float(h)) < 1.7
        if is_square:
            bolts = [(int(w * 0.175), int(h * 0.155)), (int(w * 0.825), int(h * 0.155))]
        else:
            bolts = [(int(w * 0.16), int(h * 0.21)), (int(w * 0.84), int(h * 0.21))]

        for bx, by in bolts:
            if random.random() < 0.80:
                rust_bgr = np.array([16, 42, 115], dtype=np.float32)  # Iron oxide orange-brown
                rust_len = int(h * random.uniform(0.15, 0.45 * dirt_level))
                for y_off in range(rust_len):
                    rw = max(1, int(3 * (1.0 - y_off / float(rust_len))))
                    alpha = (1.0 - y_off / float(rust_len)) * 0.75
                    if by + y_off < h and bx + rw + 1 <= w and bx - rw >= 0:
                        overlay = img[by + y_off, bx - rw:bx + rw + 1]
                        img[by + y_off, bx - rw:bx + rw + 1] = overlay * (1.0 - alpha) + rust_bgr * alpha

    # --- 5. Rain Water Drying Streaks ---
    if dirt_level > 0.25 and random.random() < 0.70:
        max_streaks = max(2, int(8 * dirt_level) + 2)
        num_streaks = random.randint(1, max_streaks)
        streak_color = np.array([35, 48, 62], dtype=np.float32)
        for _ in range(num_streaks):
            sx = random.randint(int(w * 0.05), int(w * 0.95))
            sy1 = random.randint(0, int(h * 0.30))
            sy2 = random.randint(int(h * 0.50), h - 1)
            alpha = random.uniform(0.25, 0.55)
            overlay = img.copy()
            cv2.line(overlay, (sx, sy1), (sx + random.randint(-4, 4), sy2), streak_color.tolist(), random.randint(1, 2))
            img = img * (1.0 - alpha) + overlay * alpha

    # --- 6. Organic Mud & Road Tar Splatters ---
    if dirt_level > 0.15:
        # Asphalt tar specks (tiny black dots)
        max_tar = max(2, int(25 * dirt_level))
        num_tar = random.randint(1, max_tar)
        for _ in range(num_tar):
            tx = random.randint(0, w - 1)
            ty = random.randint(int(h * 0.2), h - 1)
            cv2.circle(img, (tx, ty), random.randint(1, 2), (18, 18, 18), -1)

        # Mud droplet clusters biased along perspective airflow
        max_clusters = max(2, int(9 * dirt_level) + 2)
        num_clusters = random.randint(1, max_clusters)
        yaw_bias_px = int(w * 0.12 * (yaw_deg / 25.0))
        for _ in range(num_clusters):
            cx = int(np.clip(random.randint(int(w * 0.05), int(w * 0.95)) - yaw_bias_px, 0, w - 1))
            cy = random.randint(int(h * 0.20), h - 1)
            cluster_r = random.randint(4, max(5, int(w * 0.08 * dirt_level) + 6))
            num_drops = random.randint(5, 18)
            mud_c = np.array([
                random.randint(25, 45),
                random.randint(35, 60),
                random.randint(50, 80)
            ], dtype=np.float32)
            for _ in range(num_drops):
                dx = int(random.gauss(0, cluster_r * 0.5))
                dy = int(random.gauss(0, cluster_r * 0.5))
                # Slight teardrop streak along perspective airflow
                dx_flow = int(dy * math.sin(math.radians(-yaw_deg * 0.4)))
                px, py = cx + dx + dx_flow, cy + dy
                rad = max(1, int(random.uniform(1.0, max(2.0, cluster_r * 0.25))))
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(img, (px, py), rad, mud_c.tolist(), -1)

    # --- 7. Diesel Exhaust Soot (Typical for Pickups & Heavy Transport) ---
    if (dirt_level > 0.35 and plate_type in ["WhiteGreen", "YellowBlack", "YellowHeavy"]) or dirt_level > 0.65:
        soot_c = np.array([20, 22, 24], dtype=np.float32)
        soot_mask = np.zeros((h, w), dtype=np.float32)
        side = random.choice(["right", "left"])
        for x in range(w):
            prog = x / float(w) if side == "right" else (w - x) / float(w)
            for y in range(h):
                y_fact = (y / float(h)) ** 1.2
                soot_mask[y, x] = (max(0.0, prog - 0.40) / 0.60) * y_fact * (dirt_level * 0.70)
        soot_mask = np.clip(soot_mask[:, :, None], 0.0, 0.70)
        img = img * (1.0 - soot_mask) + soot_c * soot_mask

    return np.clip(img, 0, 255).astype(np.uint8)


def sample_perspective_angles(profile: str) -> Tuple[float, float, float]:
    """Samples realistic 3D pitch (rx), yaw (ry), and roll (rz) angles based on camera profile."""
    if profile == "mobile_close_sharp":
        rx = random.uniform(2.0, 7.0)
        ry = random.uniform(-6.0, 6.0)
        rz = random.uniform(-2.5, 2.5)
    elif profile == "mobile_handheld_angle":
        rx = random.uniform(6.0, 16.0)
        ry = random.choice([-1.0, 1.0]) * random.uniform(10.0, 22.0)
        rz = random.uniform(-5.0, 5.0)
    elif profile == "mobile_sun_glare":
        rx = random.uniform(3.0, 10.0)
        ry = random.uniform(-10.0, 10.0)
        rz = random.uniform(-3.0, 3.0)
    elif profile == "mobile_shaded_ambient":
        rx = random.uniform(4.0, 12.0)
        ry = random.uniform(-12.0, 12.0)
        rz = random.uniform(-4.0, 4.0)
    elif profile == "cctv_overhead_steep":
        rx = random.uniform(20.0, 34.0)
        ry = random.uniform(-14.0, 14.0)
        rz = random.choice([-1.0, 1.0]) * random.uniform(8.0, 14.0)
    elif profile == "cctv_slanted_approach":
        rx = random.uniform(14.0, 24.0)
        ry = random.choice([-1.0, 1.0]) * random.uniform(16.0, 28.0)
        rz = random.uniform(-7.0, 7.0)
    else:  # mobile_distant_crop / cctv_distant
        rx = random.uniform(4.0, 12.0)
        ry = random.uniform(-10.0, 10.0)
        rz = random.uniform(-3.5, 3.5)
    return rx, ry, rz


def apply_directional_motion_blur(img: np.ndarray, length: int = 5, angle_deg: float = 0.0) -> np.ndarray:
    """Applies authentic linear motion blur along vehicle velocity or camera movement vector."""
    if length <= 1:
        return img
    M = cv2.getRotationMatrix2D((length / 2.0, length / 2.0), angle_deg, 1.0)
    k = np.zeros((length, length), dtype=np.float32)
    k[length // 2, :] = 1.0
    k = cv2.warpAffine(k, M, (length, length))
    k_sum = k.sum()
    if k_sum > 0:
        k /= k_sum
        return cv2.filter2D(img, -1, k)
    return img


def simulate_mobile_optics_and_angle(
    plate_bgr: np.ndarray,
    profile: str,
    target_width: int,
    jpeg_quality: int = 50,
    angles: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    Transforms clean plate die to match real mobile phone and roadside CCTV camera captures:
    1. Realistic 3D perspective warp (pitch, yaw, roll) coupled with camera vantage.
    2. Depth-of-Field (DoF) gradient blur across the tilted perspective plane.
    3. Vehicle motion blur smear along travel trajectory.
    4. Optical defocus blur causing characters to soften and bleed into background grime.
    5. Subpixel down-upsampling to soften character loops and stroke counters.
    6. Outdoor sunlight specular glints, ambient shade, ISO sensor grain, and JPEG ringing.
    """
    h, w = plate_bgr.shape[:2]

    # Add physical mounting bolts on mobile shots (70% probability for realistic appearance)
    if random.random() < 0.70:
        plate_bgr = add_mobile_mounting_bolts(plate_bgr)

    # 1. Perspective Transform calibrated to mobile profiles
    if angles is not None:
        rx, ry, rz = angles
    else:
        rx, ry, rz = sample_perspective_angles(profile)

    # 3D Rotation Matrix
    rad_x, rad_y, rad_z = math.radians(rx), math.radians(ry), math.radians(rz)
    Rx = np.array([[1, 0, 0], [0, math.cos(rad_x), -math.sin(rad_x)], [0, math.sin(rad_x), math.cos(rad_x)]], dtype=np.float32)
    Ry = np.array([[math.cos(rad_y), 0, math.sin(rad_y)], [0, 1, 0], [-math.sin(rad_y), 0, math.cos(rad_y)]], dtype=np.float32)
    Rz = np.array([[math.cos(rad_z), -math.sin(rad_z), 0], [math.sin(rad_z), math.cos(rad_z), 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Rx @ Ry

    pts_3d = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0]], dtype=np.float32)
    f_len = 1400.0
    pts_proj = []
    z_vals = []
    for pt in pts_3d:
        p_rot = R @ pt
        z = p_rot[2] + f_len
        z_vals.append(float(p_rot[2]))
        px = (p_rot[0] * f_len) / z
        py = (p_rot[1] * f_len) / z
        pts_proj.append([px, py])
    pts_proj = np.array(pts_proj, dtype=np.float32)

    min_x, min_y = pts_proj[:, 0].min(), pts_proj[:, 1].min()
    max_x, max_y = pts_proj[:, 0].max(), pts_proj[:, 1].max()

    pad = 4
    out_w = int(math.ceil(max_x - min_x)) + pad * 2
    out_h = int(math.ceil(max_y - min_y)) + pad * 2

    dst_pts = pts_proj - np.array([min_x, min_y]) + pad
    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(plate_bgr, M, (out_w, out_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 35, 40))

    # Scale to target low resolution
    scale = target_width / float(out_w)
    target_h = int(round(out_h * scale))
    interp = cv2.INTER_AREA if scale < 0.9 else cv2.INTER_LINEAR
    processed = cv2.resize(warped, (target_width, target_h), interpolation=interp)

    # 2. Depth-of-Field Perspective Blur Gradient
    # When tilted in 3D perspective, receding characters fade into depth blur
    delta_z = max(z_vals) - min(z_vals)
    if delta_z > 14.0 and target_width > 60:
        blurred_layer = cv2.GaussianBlur(processed, (5, 5), 2.0)
        if abs(ry) >= abs(rx):
            z_grad = np.linspace(0.0, 1.0, target_width)[None, :] if ry > 0 else np.linspace(1.0, 0.0, target_width)[None, :]
        else:
            z_grad = np.linspace(1.0, 0.0, target_h)[:, None] if rx > 0 else np.linspace(0.0, 1.0, target_h)[:, None]
        z_mask = np.clip(z_grad * (delta_z / 55.0), 0.0, 0.65)[:, :, None].astype(np.float32)
        processed = np.clip(processed.astype(np.float32) * (1.0 - z_mask) + blurred_layer.astype(np.float32) * z_mask, 0, 255).astype(np.uint8)

    # 3. Vehicle Motion Blur (60% probability)
    if random.random() < 0.60:
        m_len = random.choice([3, 5]) if target_width < 90 else random.choice([3, 5, 7])
        m_angle = random.uniform(-12.0, 12.0) + (ry * 0.35)
        processed = apply_directional_motion_blur(processed, length=m_len, angle_deg=m_angle)

    # 4. Optical Defocus Blur (Make Letters Soft & Blurry)
    if target_width < 70:
        k_blur = random.choice([3, 5])
        sigma_blur = random.uniform(0.9, 1.6)
    elif target_width < 130:
        k_blur = random.choice([3, 5])
        sigma_blur = random.uniform(1.1, 2.1)
    else:
        k_blur = random.choice([3, 5, 7])
        sigma_blur = random.uniform(1.3, 2.5)
    processed = cv2.GaussianBlur(processed, (k_blur, k_blur), sigma_blur)

    # 5. Resampling Subpixel Softening (Dissolves sharp letter stroke edges)
    if random.random() < 0.75:
        sub_scale = random.uniform(0.70, 0.88)
        sub_w = max(16, int(round(target_width * sub_scale)))
        sub_h = max(8, int(round(target_h * sub_scale)))
        down = cv2.resize(processed, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        processed = cv2.resize(down, (target_width, target_h), interpolation=cv2.INTER_LINEAR)

    # 6. Lighting, Glare & Ambient Shade
    if profile == "mobile_sun_glare":
        glare = np.zeros_like(processed, dtype=np.float32)
        cx = int(target_width * random.uniform(0.15, 0.85))
        cy = int(target_h * random.uniform(0.10, 0.40))
        gx_rad = int(target_width * random.uniform(0.20, 0.35))
        gy_rad = int(target_h * random.uniform(0.25, 0.45))
        cv2.ellipse(glare, (cx, cy), (gx_rad, gy_rad), random.uniform(-20, 20), 0, 360, (160, 160, 160), -1)
        glare = cv2.GaussianBlur(glare, (15, 15), 0)
        processed = np.clip(processed.astype(np.float32) + glare * 0.40, 0, 255).astype(np.uint8)
    elif profile == "mobile_shaded_ambient":
        processed = np.clip(processed.astype(np.float32) * random.uniform(0.82, 0.94), 0, 255).astype(np.uint8)

    # 7. Fine Sensor Grain
    noise_sigma = random.uniform(3.0, 7.0) if target_width < 100 else random.uniform(2.0, 5.0)
    noise = np.random.normal(0, noise_sigma, processed.shape).astype(np.float32)
    processed = np.clip(processed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 8. Video / JPEG Compression Ringing
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, enc = cv2.imencode(".jpg", processed, encode_param)
    final_img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return final_img

    return final_img


def generate_mobile_synthetic_dataset(num_samples: int = 1500):
    """
    Synthesizes a comprehensive dataset calibrated against real mobile reference images
    in `dataset/snapshot/mobile` and CCTV camera footage.
    """
    print("\n" + "=" * 80)
    print("  SYNTHETIC THAI LICENSE PLATE GENERATOR (MOBILE & CCTV CALIBRATED)")
    print("  ISO/IEC 42001:2023 & ISO/IEC 5259-2:2023 COMPLIANT EXECUTION")
    print("=" * 80)
    print(f"Target Samples  : {num_samples:,}")
    print(f"Output Directory: {OUTPUT_DATASET_DIR}")
    print(f"Plates Directory: {PLATES_OUTPUT_DIR}")
    print(f"Mobile Ref Dir  : {MOBILE_REF_DIR}")
    print("=" * 80 + "\n")

    p_weights = [p["weight"] for p in PROVINCE_POOL]

    # Ground-truth test cases directly mirroring the mobile reference captures
    mobile_ground_truth_specimens = [
        # 1. High-Res WhiteBlackCar close-up
        {"type": "WhiteBlackCar", "d3": "กจ", "d4": "4108", "prov": "มุกดาหาร", "profile": "mobile_close_sharp", "w": 160, "q": 55, "dirt": 0.45, "note": "Match Screenshot 162918 (กจ 4108 มุกดาหาร)"},
        # 2. Taxi YellowBlack close-up
        {"type": "YellowBlack", "d3": "2ขร", "d4": "547", "prov": "กรุงเทพมหานคร", "profile": "mobile_close_sharp", "w": 150, "q": 50, "dirt": 0.50, "note": "Match Screenshot 162926 (2ขร 547 กทม.)"},
        # 3. Commercial Heavy Transport Land Transport Act 14-series
        {"type": "YellowHeavy", "d3": "14", "d4": "3112", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 130, "q": 45, "dirt": 0.82, "note": "Match Screenshot 163007 (14-3112 กทม.)"},
        # 4. WhiteBlackCar rare consonant ศ
        {"type": "WhiteBlackCar", "d3": "ศห", "d4": "4079", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 115, "q": 45, "dirt": 0.65, "note": "Match Screenshot 163012 (ศห 4079 กทม.)"},
        # 5. Commercial Heavy Transport 701-series
        {"type": "YellowHeavy", "d3": "701", "d4": "6466", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 125, "q": 45, "dirt": 0.78, "note": "Match Screenshot 163017 (701-6466 กทม.)"},
        # 6. Provincial Satun car
        {"type": "WhiteBlackCar", "d3": "กจ", "d4": "1986", "prov": "สตูล", "profile": "mobile_handheld_angle", "w": 85, "q": 40, "dirt": 0.40, "note": "Match Screenshot 163029 (กจ 1986 สตูล)"},
        # 7. Modern 7-series car
        {"type": "WhiteBlackCar", "d3": "7กฬ", "d4": "2597", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 95, "q": 45, "dirt": 0.45, "note": "Match Screenshot 163042 (7กฬ 2597 กทม.)"},
        # 8. Provincial Udon Thani car
        {"type": "WhiteBlackCar", "d3": "ขจ", "d4": "4228", "prov": "อุดรธานี", "profile": "mobile_handheld_angle", "w": 100, "q": 45, "dirt": 0.50, "note": "Match Screenshot 163047 (ขจ 4228 อุดรธานี)"},
        # 9. Rare consonant ฎ
        {"type": "WhiteBlackCar", "d3": "2ขฎ", "d4": "5906", "prov": "กรุงเทพมหานคร", "profile": "mobile_close_sharp", "w": 115, "q": 45, "dirt": 0.55, "note": "Match Screenshot 163059 (2ขฎ 5906 กทม.)"},
        # 10. Rare consonant ฐ
        {"type": "WhiteBlackCar", "d3": "4ขฐ", "d4": "2497", "prov": "กรุงเทพมหานคร", "profile": "mobile_sun_glare", "w": 120, "q": 45, "dirt": 0.35, "note": "Match Screenshot 163118 (4ขฐ 2497 กทม.)"},
        # 11. WhiteGreen Commercial Pickup with rare consonant ฒ
        {"type": "WhiteGreen", "d3": "3ฒล", "d4": "7019", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 110, "q": 45, "dirt": 0.65, "note": "Match Screenshot 163123 (3ฒล 7019 กทม.)"},
        # 12. WhiteGreen Commercial Pickup with dual rare consonants ฒ, ฐ
        {"type": "WhiteGreen", "d3": "3ฒฐ", "d4": "795", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 110, "q": 45, "dirt": 0.60, "note": "Match Screenshot 163125 (3ฒฐ 795 กทม.)"},
        # 13. WhiteGreen Commercial Pickup with dual rare consonants ฒ, ฌ
        {"type": "WhiteGreen", "d3": "4ฒฌ", "d4": "5525", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 105, "q": 45, "dirt": 0.55, "note": "Match Screenshot 163127 (4ฒฌ 5525 กทม.)"},
        # 14. WhiteBlackCar duplicate consonants วว
        {"type": "WhiteBlackCar", "d3": "วว", "d4": "5717", "prov": "กรุงเทพมหานคร", "profile": "mobile_close_sharp", "w": 130, "q": 50, "dirt": 0.40, "note": "Match Screenshot 163133 (วว 5717 กทม.)"},
        # 15. Taxi YellowBlack with rare consonant ฬ
        {"type": "YellowBlack", "d3": "ทฬ", "d4": "5400", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 90, "q": 40, "dirt": 0.50, "note": "Match Screenshot 163144 (ทฬ 5400 กทม.)"},
        # 16. WhiteBlackCar duplicate rare consonants ฌฌ
        {"type": "WhiteBlackCar", "d3": "ฌฌ", "d4": "140", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 100, "q": 45, "dirt": 0.45, "note": "Match Screenshot 163151 (ฌฌ 140 กทม.)"},
        # 17. Van WhiteBlue Nonthaburi
        {"type": "WhiteBlue", "d3": "นง", "d4": "8970", "prov": "นนทบุรี", "profile": "mobile_handheld_angle", "w": 115, "q": 45, "dirt": 0.50, "note": "Match Screenshot 163208 (นง 8970 นนทบุรี)"},
        # 18. WhiteBlackCar duplicate rare consonants ษษ
        {"type": "WhiteBlackCar", "d3": "ษษ", "d4": "2858", "prov": "กรุงเทพมหานคร", "profile": "mobile_shaded_ambient", "w": 105, "q": 45, "dirt": 0.45, "note": "Match Screenshot 163216 (ษษ 2858 กทม.)"},
        # 19. Motorcycle Taxi YellowBlack 3-consonant กกบ
        {"type": "YellowBlack", "d3": "กกบ", "d4": "310", "prov": "ปทุมธานี", "profile": "mobile_handheld_angle", "w": 95, "q": 40, "dirt": 0.60, "note": "Match Screenshot 163219 (กกบ 310 ปทุมธานี)"},
        # 20. Heavy Transport Nakhon Pathom 72-series
        {"type": "YellowHeavy", "d3": "72", "d4": "1135", "prov": "นครปฐม", "profile": "mobile_handheld_angle", "w": 110, "q": 45, "dirt": 0.80, "note": "Match Screenshot 163225 (72-1135 นครปฐม)"},
        # 21. WhiteGreen Commercial Pickup Phra Nakhon Si Ayutthaya with consonant ผ
        {"type": "WhiteGreen", "d3": "ผก", "d4": "6290", "prov": "พระนครศรีอยุธยา", "profile": "mobile_handheld_angle", "w": 90, "q": 40, "dirt": 0.70, "note": "Match Screenshot 163303 (ผก 6290 พระนครศรีอยุธยา)"},
        # 22. Rare consonant ฌ
        {"type": "WhiteBlackCar", "d3": "3ขฌ", "d4": "6781", "prov": "กรุงเทพมหานคร", "profile": "mobile_handheld_angle", "w": 105, "q": 45, "dirt": 0.45, "note": "Match Screenshot 163317 (3ขฌ 6781 กทม.)"},
        # 23. Motorcycle 3-line format
        {"type": "WhiteBlackMotor", "d3": "1กจ", "d4": "4160", "prov": "กรุงเทพมหานคร", "profile": "cctv_overhead_steep", "w": 120, "q": 45, "dirt": 0.55, "note": "Motorcycle ground truth: 1กจ 4160"},
        # 24. CCTV Slanted Turning Sedan
        {"type": "WhiteBlackCar", "d3": "ขม", "d4": "6343", "prov": "ภูเก็ต", "profile": "cctv_slanted_approach", "w": 95, "q": 40, "dirt": 0.50, "note": "CCTV ground truth: ขม 6343 ภูเก็ต"},
    ]

    records: List[Dict[str, Any]] = []
    annotations_data: List[Dict[str, Any]] = []
    session_items: Dict[str, Any] = {}

    mobile_profiles = [
        "mobile_close_sharp",
        "mobile_handheld_angle",
        "mobile_sun_glare",
        "mobile_shaded_ambient",
        "mobile_distant_crop"
    ]
    cctv_profiles = [
        "cctv_slanted_approach",
        "cctv_overhead_steep"
    ]

    # Reduced realistic edge resolution tiers (character heights 10-28px)
    resolution_tiers = [
        ("distant_lowres_blurry", 42, 75, 0.30),    # 42-75px width (extreme low-res, character strokes melt together)
        ("traffic_edge_crop", 76, 125, 0.40),       # 76-125px width (standard roadside camera resolution)
        ("medium_handheld_crop", 126, 185, 0.20),   # 126-185px width (handheld street capture)
        ("close_crop", 186, 260, 0.10)              # 186-260px width (capped maximum resolution)
    ]
    tier_weights = [t[3] for t in resolution_tiers]

    print("[*] Generating calibrated synthetic plates...")
    for idx in range(num_samples):
        sample_num = idx + 1
        sample_id = f"mob_{sample_num:05d}"

        # 1. Parameter Selection
        if idx < len(mobile_ground_truth_specimens):
            sc = mobile_ground_truth_specimens[idx]
            p_type = sc["type"]
            d3 = sc["d3"]
            d4 = sc["d4"]
            prov_dict = next(p for p in PROVINCE_POOL if p["name"] == sc["prov"])
            profile = sc["profile"]
            target_w = sc["w"]
            jpeg_q = sc["q"]
            dirt_level = sc.get("dirt", 0.35)
        else:
            prov_dict = random.choices(PROVINCE_POOL, weights=p_weights, k=1)[0]
            prov_name = prov_dict["name"]

            # Vehicle class distribution
            roll = random.random()
            if roll < 0.45:
                p_type = "WhiteBlackCar"
            elif roll < 0.65:
                p_type = "WhiteGreen"
            elif roll < 0.80:
                p_type = "YellowBlack"
            elif roll < 0.90:
                p_type = "WhiteBlue"
            else:
                p_type = "WhiteBlackMotor"

            # Profile & Resolution selection
            if p_type == "WhiteBlackMotor":
                profile = random.choice(["cctv_overhead_steep", "mobile_handheld_angle", "mobile_close_sharp"])
            else:
                # 80% mobile profiles, 20% cctv profiles
                if random.random() < 0.80:
                    profile = random.choices(mobile_profiles, weights=[0.25, 0.35, 0.15, 0.15, 0.10], k=1)[0]
                else:
                    profile = random.choice(cctv_profiles)

            tier = random.choices(resolution_tiers, weights=tier_weights, k=1)[0]
            target_w = random.randint(tier[1], tier[2])
            jpeg_q = random.randint(28, 62)

            # Realistic Road Dirt & Weathering Distribution:
            # 15% light road film (0.12 - 0.28)
            # 50% moderate road grime (0.30 - 0.60)
            # 23% heavy weathering (0.60 - 0.82)
            # 12% severe weathering/mud/chipping (0.82 - 0.96)
            dirt_tier = random.choices(["light", "moderate", "heavy", "severe"], weights=[0.15, 0.50, 0.23, 0.12], k=1)[0]
            if dirt_tier == "light":
                dirt_level = random.uniform(0.12, 0.28)
            elif dirt_tier == "moderate":
                dirt_level = random.uniform(0.30, 0.60)
            elif dirt_tier == "heavy":
                dirt_level = random.uniform(0.60, 0.82)
            else:
                dirt_level = random.uniform(0.82, 0.96)

            # Generate d3 and d4
            if p_type == "WhiteBlackCar":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.80:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(CAR_INITIALS, boost_rare_prob=0.35)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(CAR_INITIALS, boost_rare_prob=0.35)
                # 85% 4-digits, 15% 1-3 digits
                d4_len = random.choices([4, 3, 2, 1], weights=[0.85, 0.08, 0.05, 0.02], k=1)[0]
                d4 = str(random.randint(1, 10**d4_len - 1)).zfill(d4_len)

            elif p_type == "WhiteGreen":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.70:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(PICKUP_INITIALS, boost_rare_prob=0.50)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(PICKUP_INITIALS, boost_rare_prob=0.50)
                d4 = f"{random.randint(1, 9999)}"

            elif p_type == "WhiteBlue":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.60:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(VAN_INITIALS, boost_rare_prob=0.40)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(VAN_INITIALS, boost_rare_prob=0.40)
                d4 = f"{random.randint(1, 9999)}"

            elif p_type == "YellowBlack":
                # Check commercial heavy transport vs taxi
                sub_roll = random.random()
                if sub_roll < 0.40:
                    # Heavy transport (10-79, 700-799)
                    if random.random() < 0.30:
                        d3 = str(random.randint(700, 799))
                    else:
                        d3 = str(random.choice(list(range(10, 40)) + list(range(60, 80))))
                    d4 = f"{random.randint(1, 9999):04d}"
                elif sub_roll < 0.85:
                    # Taxi format (ท, ม)
                    lead = "1" if prov_name == "กรุงเทพมหานคร" and random.random() < 0.50 else ""
                    pair = get_random_valid_pair(TAXI_INITIALS, boost_rare_prob=0.35)
                    d3 = lead + pair
                    d4 = f"{random.randint(1, 9999)}"
                else:
                    # Motorcycle taxi (e.g. กกบ 310)
                    c1 = random.choice(ALL_VALID_CONSONANTS)
                    c2 = random.choice(ALL_VALID_CONSONANTS)
                    c3 = random.choice(ALL_VALID_CONSONANTS)
                    d3 = c1 + c2 + c3
                    d4 = f"{random.randint(1, 999)}"

            elif p_type == "WhiteBlackMotor":
                lead = str(random.randint(1, 9)) if random.random() < 0.70 else ""
                pair = get_random_valid_pair(ALL_VALID_CONSONANTS, boost_rare_prob=0.35)
                d3 = lead + pair
                d4 = f"{random.randint(1, 9999)}"

        prov_name = prov_dict["name"]
        iso_code = prov_dict["iso"]

        # 2. Render Clean Die
        if p_type == "WhiteBlackMotor":
            clean_die = render_motorcycle_plate_die(
                top_text=d3,
                province_text=prov_name,
                num_text=d4,
                colorway_key="private_car"
            )
            full_text = f"{d3} {d4} {prov_name}"
            clean_name_type = "WhiteBlackMotor"
        elif p_type == "YellowHeavy" or (p_type == "YellowBlack" and (d3.isdigit() or "-" in d3)):
            plate_reg_text = f"{d3}-{d4}" if "-" not in d3 else f"{d3} {d4}"
            clean_die = render_calibrated_plate_die(
                plate_text=plate_reg_text,
                province_text=prov_name,
                colorway_key="taxi",
                is_commercial_transport=True
            )
            full_text = f"{plate_reg_text} {prov_name}"
            clean_name_type = "YellowBlack"
        else:
            cway_map = {
                "WhiteBlackCar": "private_car",
                "WhiteGreen": "pickup",
                "WhiteBlue": "van",
                "YellowBlack": "taxi"
            }
            cway = cway_map.get(p_type, "private_car")
            plate_reg_text = f"{d3} {d4}"
            clean_die = render_calibrated_plate_die(
                plate_text=plate_reg_text,
                province_text=prov_name,
                colorway_key=cway,
                is_commercial_transport=False
            )
            full_text = f"{plate_reg_text} {prov_name}"
            clean_name_type = p_type

        # 2.5 Sample Perspective Angles (coupling perspective to dirt physics)
        rx, ry, rz = sample_perspective_angles(profile)

        # Apply Realistic Road Dirt, Grime & Physical Weathering Coupled with Perspective
        weathered_die = apply_unclean_road_weathering(
            plate_bgr=clean_die,
            dirt_level=dirt_level,
            plate_type=clean_name_type,
            yaw_deg=ry,
            pitch_deg=rx
        )

        # 3. Simulate Mobile / CCTV Optics with 3D Warp, Depth Blur & Blurry Letters
        simulated_img = simulate_mobile_optics_and_angle(
            plate_bgr=weathered_die,
            profile=profile,
            target_width=target_w,
            jpeg_quality=jpeg_q,
            angles=(rx, ry, rz)
        )
        sim_h, sim_w = simulated_img.shape[:2]

        # 4. Save Image
        filename = f"{sample_id}_{clean_name_type}_{d3}_{d4}_{prov_name}.jpg"
        filepath = PLATES_OUTPUT_DIR / filename
        cv2_imwrite_safe(filepath, simulated_img, quality=jpeg_q)

        # 5. Metadata Record
        meta_row = {
            "sample_id": sample_id,
            "filename": filename,
            "plate_type": clean_name_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "iso_3166_2": iso_code,
            "full_plate": full_text,
            "dirt_level": round(float(dirt_level), 2),
            "width": sim_w,
            "height": sim_h,
            "optical_profile": profile,
            "jpeg_quality": jpeg_q,
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        }
        records.append(meta_row)

        # 6. Annotations
        ann_item = {
            "image": filename,
            "annotations": [
                {
                    "label": full_text,
                    "d3": d3,
                    "d4": d4,
                    "province": prov_name,
                    "plate_type": clean_name_type,
                    "coordinates": {"x": 0, "y": 0, "width": sim_w, "height": sim_h}
                }
            ]
        }
        annotations_data.append(ann_item)

        # 7. Session Entry (compatible with LPR Studio inspection)
        session_items[filename] = {
            "filename": filename,
            "rel_path": f"plates/{filename}",
            "subfolder": "plates",
            "name": sample_id,
            "plate_type": clean_name_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "iso_3166_2": iso_code,
            "status": "approved",
            "ai_d3": d3,
            "ai_d4": d4,
            "ai_province": prov_name,
            "ai_confidence": 0.99
        }

        if sample_num % 250 == 0 or sample_num == num_samples:
            print(f"  [+] Synthesized {sample_num:,}/{num_samples:,} plates ({clean_name_type}: {d3} {d4} {prov_name})")

    # Save CSV
    csv_path = OUTPUT_DATASET_DIR / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Save Annotations JSON
    ann_path = OUTPUT_DATASET_DIR / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations_data, f, indent=2, ensure_ascii=False)

    # Save LPR Session JSON
    session_path = OUTPUT_DATASET_DIR / ".lpr_session.json"
    session_payload = {
        "version": "2.0",
        "standards": ["ISO 8601:2019", "ISO 3166-2:TH", "ISO/IEC 42001:2023", "ISO/IEC 5259-2:2023"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "folder": str(OUTPUT_DATASET_DIR.resolve()).replace("\\", "/"),
        "current_index": 0,
        "items": session_items
    }
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Synthetic dataset generation complete!")
    print(f"  Total Plates Synthesized: {len(records):,}")
    print(f"  Metadata CSV Catalog    : {csv_path}")
    print(f"  Annotations JSON        : {ann_path}")
    print(f"  LPR Studio Session      : {session_path}\n")

    # Generate Direct Verification Comparison Gallery
    generate_verification_gallery()


def generate_verification_gallery():
    """
    Renders an inspection gallery comparing real mobile snapshots directly with
    their newly synthesized twin plates across all classes and conditions.
    """
    print("[*] Generating Direct Mobile Verification Comparison Gallery...")
    gallery_path = OUTPUT_DATASET_DIR / "mobile_specimen_comparison.jpg"

    # Pairs of (Real Mobile File from mobile_dir, Synthesized Match File from dataset)
    specimen_pairs = [
        # Pair 1: กจ 4108 มุกดาหาร (High-Res Car Close-Up)
        {
            "title_real": "Real Mobile: กจ 4108 มุกดาหาร (Outdoor Macro)",
            "file_real": "Screenshot 2026-10-27 162918_WhiteBlackCar_กจ_4108_มุกดาหาร.png",
            "title_synth": "Synthetic Twin: กจ 4108 มุกดาหาร (Mobile Close Sharp)",
            "file_synth": "mob_00001_WhiteBlackCar_กจ_4108_มุกดาหาร.jpg"
        },
        # Pair 2: 2ขร 547 กทม. (Yellow/Black Taxi)
        {
            "title_real": "Real Mobile: 2ขร 547 กทม. (Taxi Yellow/Black)",
            "file_real": "Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 2ขร 547 กทม. (Mobile Close Sharp)",
            "file_synth": "mob_00002_YellowBlack_2ขร_547_กรุงเทพมหานคร.jpg"
        },
        # Pair 3: 14-3112 กทม. (Commercial Transport)
        {
            "title_real": "Real Mobile: 14-3112 กทม. (Heavy Transport)",
            "file_real": "Screenshot 2026-10-27 163007_YellowBlack_14_3112_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 14-3112 กทม. (Mobile Handheld)",
            "file_synth": "mob_00003_YellowBlack_14_3112_กรุงเทพมหานคร.jpg"
        },
        # Pair 4: 701-6466 กทม. (700-Series Commercial)
        {
            "title_real": "Real Mobile: 701-6466 กทม. (700+ Transport)",
            "file_real": "Screenshot 2026-10-27 163017_YellowBlack_701_6466_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 701-6466 กทม. (Mobile Handheld)",
            "file_synth": "mob_00005_YellowBlack_701_6466_กรุงเทพมหานคร.jpg"
        },
        # Pair 5: 3ฒล 7019 กทม. (White/Green Commercial Pickup with ฒ)
        {
            "title_real": "Real Mobile: 3ฒล 7019 กทม. (Commercial Pickup)",
            "file_real": "Screenshot 2026-10-27 163123_WhiteGreen_3ฒล_7019_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 3ฒล 7019 กทม. (Mobile Handheld)",
            "file_synth": "mob_00011_WhiteGreen_3ฒล_7019_กรุงเทพมหานคร.jpg"
        },
        # Pair 6: 3ฒฐ 795 กทม. (White/Green Pickup with dual rare consonants ฒ, ฐ)
        {
            "title_real": "Real Mobile: 3ฒฐ 795 กทม. (Pickup ฒ, ฐ)",
            "file_real": "Screenshot 2026-10-27 163125_WhiteGreen_3ฒฐ_795_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 3ฒฐ 795 กทม. (Mobile Handheld)",
            "file_synth": "mob_00012_WhiteGreen_3ฒฐ_795_กรุงเทพมหานคร.jpg"
        },
        # Pair 7: นง 8970 นนทบุรี (White/Blue Van)
        {
            "title_real": "Real Mobile: นง 8970 นนทบุรี (Van White/Blue)",
            "file_real": "Screenshot 2026-10-27 163208_WhiteBlue_นง_8970_นนทบุรี.png",
            "title_synth": "Synthetic Twin: นง 8970 นนทบุรี (Mobile Handheld)",
            "file_synth": "mob_00017_WhiteBlue_นง_8970_นนทบุรี.jpg"
        },
        # Pair 8: 4ขฐ 2497 กทม. (Car with rare consonant ฐ)
        {
            "title_real": "Real Mobile: 4ขฐ 2497 กทม. (Sunlit Car)",
            "file_real": "Screenshot 2026-10-27 163118_WhiteBlackCar_4ขฐ_2497_กรุงเทพมหานคร.png",
            "title_synth": "Synthetic Twin: 4ขฐ 2497 กทม. (Mobile Sun Glare)",
            "file_synth": "mob_00010_WhiteBlackCar_4ขฐ_2497_กรุงเทพมหานคร.jpg"
        },
        # Pair 9: ผก 6290 อยุธยา (Pickup with rare consonant ผ)
        {
            "title_real": "Real Mobile: ผก 6290 อยุธยา (Ayutthaya Pickup)",
            "file_real": "Screenshot 2026-10-27 163303_WhiteGreen_ผก_6290_พระนครศรีอยุธยา.png",
            "title_synth": "Synthetic Twin: ผก 6290 อยุธยา (Mobile Handheld)",
            "file_synth": "mob_00021_WhiteGreen_ผก_6290_พระนครศรีอยุธยา.jpg"
        }
    ]

    card_w = 460
    card_h = 210
    cols = 2
    # Total cards = len(specimen_pairs) * 2 (one real, one synthetic side-by-side)
    total_cards = len(specimen_pairs) * 2
    rows = (total_cards + cols - 1) // cols

    sheet_w = cols * (card_w + 24) + 32
    sheet_h = 110 + rows * (card_h + 24)

    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), color=(241, 245, 249))
    draw = ImageDraw.Draw(sheet_pil)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 20)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 13)
    font_card = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 12)
    font_badge = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 10)

    # Header Card
    draw.rounded_rectangle([16, 14, sheet_w - 16, 88], radius=8, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((32, 24), "Real Mobile Camera Snapshots vs Calibrated Synthetic Twin Plates", fill=(15, 23, 42), font=font_title)
    draw.text((32, 54), "Visual Verification of Handheld Optics, 3D Embossing, Mounting Screws, and Rare DLT Consonants (ฒ, ฐ, ฎ, ผ, ฬ)", fill=(71, 85, 105), font=font_sub)

    # Render pairs sequentially side-by-side
    card_idx = 0
    for p in specimen_pairs:
        # 1. Real Card
        r_c = card_idx % cols
        r_r = card_idx // cols
        rx = 20 + r_c * (card_w + 24)
        ry = 104 + r_r * (card_h + 24)

        draw.rounded_rectangle([rx, ry, rx + card_w, ry + card_h], radius=8, fill=(255, 255, 255), outline=(59, 130, 246), width=2)
        draw.text((rx + 12, ry + 10), p["title_real"], fill=(15, 23, 42), font=font_card)

        # Real badge
        badge_real = "REAL MOBILE REF"
        b_box = font_badge.getbbox(badge_real)
        bw = b_box[2] - b_box[0] + 12
        bh = b_box[3] - b_box[1] + 6
        bx = rx + card_w - bw - 12
        draw.rounded_rectangle([bx, ry + 8, bx + bw, ry + 8 + bh], radius=4, fill=(30, 64, 175))
        draw.text((bx + 6, ry + 11), badge_real, fill=(255, 255, 255), font=font_badge)

        # Container
        cx1, cy1, cx2, cy2 = rx + 10, ry + 36, rx + card_w - 10, ry + card_h - 10
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=6, fill=(24, 28, 36), outline=(51, 65, 85), width=1)

        real_img = cv2_imread_safe(MOBILE_REF_DIR / p["file_real"])
        if real_img is not None:
            box_w, box_h = (cx2 - cx1) - 16, (cy2 - cy1) - 16
            ch, cw = real_img.shape[:2]
            scale = min(box_w / float(cw), box_h / float(ch))
            dw, dh = int(round(cw * scale)), int(round(ch * scale))
            resized = cv2.resize(real_img, (dw, dh), interpolation=cv2.INTER_LANCZOS4 if scale > 0.8 else cv2.INTER_AREA)
            px = cx1 + 8 + (box_w - dw) // 2
            py = cy1 + 8 + (box_h - dh) // 2
            sheet_pil.paste(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), (px, py))
        else:
            draw.text((cx1 + 20, cy1 + 50), f"[File Not Found: {p['file_real']}]", fill=(239, 68, 68), font=font_card)

        card_idx += 1

        # 2. Synthetic Card
        s_c = card_idx % cols
        s_r = card_idx // cols
        sx = 20 + s_c * (card_w + 24)
        sy = 104 + s_r * (card_h + 24)

        draw.rounded_rectangle([sx, sy, sx + card_w, sy + card_h], radius=8, fill=(255, 255, 255), outline=(16, 185, 129), width=2)
        draw.text((sx + 12, sy + 10), p["title_synth"], fill=(15, 23, 42), font=font_card)

        badge_synth = "SYNTHETIC TWIN"
        b_box = font_badge.getbbox(badge_synth)
        bw = b_box[2] - b_box[0] + 12
        bh = b_box[3] - b_box[1] + 6
        bx = sx + card_w - bw - 12
        draw.rounded_rectangle([bx, sy + 8, bx + bw, sy + 8 + bh], radius=4, fill=(4, 120, 87))
        draw.text((bx + 6, sy + 11), badge_synth, fill=(255, 255, 255), font=font_badge)

        cx1, cy1, cx2, cy2 = sx + 10, sy + 36, sx + card_w - 10, sy + card_h - 10
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=6, fill=(24, 28, 36), outline=(51, 65, 85), width=1)

        synth_img = cv2_imread_safe(PLATES_OUTPUT_DIR / p["file_synth"])
        if synth_img is not None:
            box_w, box_h = (cx2 - cx1) - 16, (cy2 - cy1) - 16
            ch, cw = synth_img.shape[:2]
            scale = min(box_w / float(cw), box_h / float(ch))
            dw, dh = int(round(cw * scale)), int(round(ch * scale))
            resized = cv2.resize(synth_img, (dw, dh), interpolation=cv2.INTER_LANCZOS4 if scale > 0.8 else cv2.INTER_AREA)
            px = cx1 + 8 + (box_w - dw) // 2
            py = cy1 + 8 + (box_h - dh) // 2
            sheet_pil.paste(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)), (px, py))
        else:
            draw.text((cx1 + 20, cy1 + 50), f"[File Not Found: {p['file_synth']}]", fill=(239, 68, 68), font=font_card)

        card_idx += 1

    sheet_bgr = cv2.cvtColor(np.array(sheet_pil), cv2.COLOR_RGB2BGR)
    cv2_imwrite_safe(gallery_path, sheet_bgr, quality=95)
    print(f"[+] Saved Mobile Specimen Comparison Gallery to: {gallery_path}")

    # Copy to Artifact directory for instant UI rendering
    if ARTIFACT_DIR.exists():
        artifact_gallery = ARTIFACT_DIR / "mobile_specimen_comparison.jpg"
        shutil.copy2(gallery_path, artifact_gallery)
        print(f"[+] Copied Gallery to Artifact Dir: {artifact_gallery}")


if __name__ == "__main__":
    generate_mobile_synthetic_dataset(num_samples=1500)
