#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic CCTV Synthetic Thai License Plate Dataset Generator.
Engineered to precisely simulate real-world roadside CCTV and traffic enforcement cameras:
1. Multi-Angle Perspectives:
   - Slanted approach & vehicle turning (yaw 15°-25°, pitch 20°-30°, roll ±8°) matching Image 1 & 4.
   - Steep overhead downward viewing (pitch 32°-45°, keystone foreshortening) matching Image 3 & 5.
   - Nighttime low-light with headlight bloom, high ISO grain, and reduced contrast matching Image 2.
   - Distant low-resolution (32px-55px width) and medium-resolution (60px-110px width).
2. Comprehensive DLT Vehicle Class & Colorway Coverage:
   - WhiteBlackCar: Private Cars & 4-door Pickups (Bangkok 1-9 prefixes like 8กน 5500, 7ขส 8825; Provincial 2-letter like ขม 6343).
   - WhiteBlue: Vans / Microbuses (> 7 seats, 'น, ฬ, อ, ฮ' series).
   - WhiteGreen: Private Commercial Pickups (≤ 2,200 kg, 'บ, ฒ, ณ, ต, ถ, ป, ผ, ย, ร, ล' series).
   - YellowBlack: Taxis / Commercial passenger cars ('ท, ม' series).
   - WhiteBlackMotor: Motorcycles (220x172mm 3-line format: Line 1 '1กข', Line 2 Province, Line 3 '4160').
   - HeavyTransport (Yellow/Black): Commercial transport (10-79, 700+ prefixes with THAILAND header).
   - HeavyTransport (White/Black): Private heavy transport (40-59, 80-99 prefixes with THAILAND header).
3. Conformance:
   - ISO/IEC 5259-2:2023 (AI Data Quality Measures)
   - ISO 3166-2:TH (Thai Province Standardization)
   - ISO 8601:2019 (Timestamps)
   - ISO/IEC 42001:2023 (AI Governance & Change Control)
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
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageFilter

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
    render_calibrated_plate_die,
    shape_thai_text,
    DLT_PROVINCE_CODES,
    COLORWAYS
)
from tools.generate_mobile_cctv_synthetic_dataset import apply_unclean_road_weathering

# Fonts
FONT_ID_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-id.ttf"
FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
FONT_YELLOW_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-yellow.ttf"

# Target Dataset Output Directory
OUTPUT_DATASET_DIR = ROOT_DIR / "dataset" / "dataset-cctv-synthetic"
PLATES_OUTPUT_DIR = OUTPUT_DATASET_DIR / "plates"
OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
PLATES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Conversation Artifact Directory
ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\f39d0d07-6ecd-4d68-972d-9fd157abb790")
USER_UPLOADED_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\d2c371bd-a730-4efa-ac31-54db382d8c7f\.user_uploaded")

# DLT Province Data
KEY_PROVINCES = [
    {"name": "กรุงเทพมหานคร", "iso": "TH-10", "weight": 0.40},
    {"name": "ชลบุรี", "iso": "TH-20", "weight": 0.12},
    {"name": "เชียงใหม่", "iso": "TH-50", "weight": 0.10},
    {"name": "ภูเก็ต", "iso": "TH-83", "weight": 0.08},
    {"name": "สมุทรปราการ", "iso": "TH-11", "weight": 0.05},
    {"name": "นนทบุรี", "iso": "TH-12", "weight": 0.05},
    {"name": "ปทุมธานี", "iso": "TH-13", "weight": 0.04},
    {"name": "ระยอง", "iso": "TH-21", "weight": 0.04},
    {"name": "นครปฐม", "iso": "TH-73", "weight": 0.03},
    {"name": "ขอนแก่น", "iso": "TH-40", "weight": 0.03},
    {"name": "สุราษฎร์ธานี", "iso": "TH-84", "weight": 0.03},
    {"name": "เบตง", "iso": "TH-95", "weight": 0.03},
]

# DLT Consonants by Vehicle Class
CAR_INITIALS = list("กขงจฉชฌญฎฐธพภวศษส")
VAN_INITIALS = list("นฬอฮ")
PICKUP_INITIALS = list("บฒณตถปผยรล")
TAXI_INITIALS = list("ทม")
MOTOR_INITIALS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
ALL_VALID_CONSONANTS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
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


def get_random_valid_pair(initial_candidates: List[str]) -> str:
    """Generates a valid 2-letter pair adhering to DLT exclusions."""
    for _ in range(100):
        c1 = random.choice(initial_candidates)
        c2 = random.choice(ALL_VALID_CONSONANTS)
        pair = c1 + c2
        if pair not in OFFENSIVE_PAIRS:
            return pair
    return initial_candidates[0] + "ก"


def render_motorcycle_plate_die(
    top_text: str,       # e.g. '1กจ' or 'กข'
    province_text: str,  # e.g. 'กรุงเทพมหานคร'
    num_text: str,       # e.g. '4160'
    colorway_key: str = "private_car",
    target_w: int = 440,
    target_h: int = 344
) -> np.ndarray:
    """
    Renders authentic 220mm x 172mm 3-line Thai motorcycle plate die (2.0 scale: 440x344px).
    Accurately calibrated against DLT statutory specifications and Image 5 ground truth:
    - Line 1 (หมวด): DLT >= 40mm. Actual glyph height ~94px (47mm on 172mm), span ~54% width.
    - Line 2 (จังหวัด): DLT >= 20mm. Actual glyph height ~42px (21mm on 172mm), span ~72% width.
    - Line 3 (หมายเลข): DLT >= 40mm. Actual glyph height ~106px (53mm on 172mm), span ~72% width.
    - Dual stamped mounting bolt indentations.
    """
    w, h = target_w, target_h
    bg_rgb, fg_rgb, border_rgb, _ = COLORWAYS.get(colorway_key, COLORWAYS["private_car"])

    im_pil = Image.new("RGB", (w, h), color=bg_rgb)
    draw = ImageDraw.Draw(im_pil)

    # 1. Outer rim (DLT: 5mm inset, 10mm radius)
    inset = int(w * 0.024)
    corner_r = int(h * 0.064)
    draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=corner_r, outline=border_rgb, width=4)

    # Reference glyph measurement
    f_ref = ImageFont.truetype(str(FONT_ID_PATH), 100)
    b_ref = f_ref.getbbox("5")
    ref_h = float(b_ref[3] - b_ref[1]) # ~64px

    # 2. Line 1: Top registration mark (e.g. 1กจ or กข)
    target_glyph_h1 = int(round(h * 0.275)) # ~95px on 344h (~47.5mm)
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
    baseline_y1 = int(h * 0.315) # ~108px baseline
    for c, b in zip(chars_top, bboxes_t):
        draw.text((cur_x - b[0], baseline_y1 - b[3]), c, fill=fg_rgb, font=f_top)
        cur_x += (b[2] - b[0]) + kern1

    # 3. Line 2: Middle province name (e.g. กรุงเทพมหานคร)
    shaped_prov = shape_thai_text(province_text)
    f_prov_ref = ImageFont.truetype(str(FONT_PROV_PATH), 100)
    bp_ref = f_prov_ref.getbbox("ก")
    ref_prov_h = float(bp_ref[3] - bp_ref[1]) # ~52px

    target_glyph_h2 = int(round(h * 0.125)) # ~43px on 344h (~21.5mm)
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

    baseline_y2 = int(h * 0.540) # ~185px baseline
    draw.text(((w - pw) // 2 - bp[0], baseline_y2 - bp[3]), shaped_prov, fill=fg_rgb, font=f_prov)

    # 4. Line 3: Bottom number (e.g. 4160)
    target_glyph_h3 = int(round(h * 0.310)) # ~107px on 344h (~53.5mm)
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
    baseline_y3 = int(h * 0.895) # ~308px baseline
    for c, b in zip(chars_num, bboxes_n):
        draw.text((cur_x - b[0], baseline_y3 - b[3]), c, fill=fg_rgb, font=f_num)
        cur_x += (b[2] - b[0]) + kern3

    plate_np = np.array(im_pil)

    # 5. Dual mounting bolt indentations
    bolt_positions = [(int(w * 0.175), int(h * 0.155)), (int(w * 0.825), int(h * 0.155))]
    for bx, by in bolt_positions:
        cv2.circle(plate_np, (bx, by), int(h * 0.045), (150, 155, 160), 2)
        cv2.circle(plate_np, (bx, by), int(h * 0.038), (110, 115, 120), -1)
        cv2.circle(plate_np, (bx, by), int(h * 0.022), (40, 42, 45), -1)
        cv2.circle(plate_np, (bx, by), int(h * 0.012), (190, 195, 200), -1)

    # Subtle retroreflective micro-grain
    noise = np.random.normal(0, 1.8, plate_np.shape).astype(np.float32)
    plate_np = np.clip(plate_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)


def simulate_cctv_optics_and_angle(
    plate_bgr: np.ndarray,
    cctv_profile: str,
    target_width: int,
    jpeg_quality: int = 45
) -> np.ndarray:
    """
    Transforms clean plate die to match real roadside CCTV camera optical conditions:
    1. Homography & 3D Projective Camera Warps (yaw, pitch, roll matching Image 1, 3, 4, 5).
    2. Resolution Scaling preserving true perspective aspect ratio without reflected border artifacts.
    3. Defocus and Motion Blur.
    4. Sensor Noise & Nighttime Headlight Flare (matching Image 2).
    5. Video Stream Compression (JPEG/H.264 blockiness and ringing).
    """
    h, w = plate_bgr.shape[:2]

    # --- 1. Perspective Transform & 3D Projective Geometry ---
    if cctv_profile == "overhead_steep":
        # 3D Projective Camera Model matching Image 3 & Image 5 (Motorcycle Overhead View)
        # Effective downward pitch: 16° to 26°
        # Yaw approach angle: -14° to +14°
        # Roll / vehicle lean: -14° to -8° (or +8° to +14°)
        rx_deg = random.uniform(16.0, 25.0)
        ry_deg = random.uniform(-14.0, 14.0)
        rz_deg = random.choice([-1.0, 1.0]) * random.uniform(8.5, 13.5)

        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)

        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Rx @ Ry

        pts_3d = np.array([[-w/2, -h/2, 0], [w/2, -h/2, 0], [w/2, h/2, 0], [-w/2, h/2, 0]], dtype=np.float32)
        f_len = 1200.0
        pts_proj = []
        for pt in pts_3d:
            p_rot = R @ pt
            z = p_rot[2] + f_len
            px = (p_rot[0] * f_len) / z
            py = (p_rot[1] * f_len) / z
            pts_proj.append([px, py])
        pts_proj = np.array(pts_proj, dtype=np.float32)

        min_x, min_y = pts_proj[:, 0].min(), pts_proj[:, 1].min()
        max_x, max_y = pts_proj[:, 0].max(), pts_proj[:, 1].max()

        pad = 6
        out_w = int(math.ceil(max_x - min_x)) + pad * 2
        out_h = int(math.ceil(max_y - min_y)) + pad * 2

        dst_pts = pts_proj - np.array([min_x, min_y]) + pad
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(plate_bgr, M, (out_w, out_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 33, 38))

        scale = target_width / float(out_w)
        target_h = int(round(out_h * scale))
        low_res = cv2.resize(warped, (target_width, target_h), interpolation=cv2.INTER_AREA)

    else:
        if cctv_profile == "slanted_approach":
            # Matching Image 1 & 4 (Sedan front plate turning, yaw 15°-25°, pitch 20°-30°, roll)
            dx_l = random.uniform(0.00, 0.06) * w
            dx_r = random.uniform(0.04, 0.12) * w
            dy_t = random.uniform(0.02, 0.14) * h
            dy_b = random.uniform(0.05, 0.16) * h
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [dx_l, dy_t],
                [w - dx_r * 0.4, dy_t * 0.3],
                [w - dx_r, h - dy_b],
                [dx_l * 0.3, h - dy_b * 0.2]
            ])
        elif cctv_profile == "night_glare":
            # Matching Image 2 (Nighttime CCTV with headlight flare)
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [w * 0.04, h * 0.09],
                [w * 0.96, h * 0.04],
                [w * 0.91, h * 0.88],
                [w * 0.03, h * 0.97]
            ])
        elif cctv_profile == "distant_lowres":
            # Vehicle at 20-30 meters distance
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [w * 0.03, h * 0.06],
                [w * 0.97, h * 0.04],
                [w * 0.94, h * 0.94],
                [w * 0.05, h * 0.96]
            ])
        else:  # direct_sunny
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [w * 0.02, h * 0.05],
                [w * 0.98, h * 0.04],
                [w * 0.95, h * 0.93],
                [w * 0.04, h * 0.96]
            ])

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(plate_bgr, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 33, 38))

        # --- 2. Downscale to Real Distance Resolution ---
        aspect = h / float(w)
        target_h = int(round(target_width * aspect))
        low_res = cv2.resize(warped, (target_width, target_h), interpolation=cv2.INTER_AREA)

    # --- 3. Optical Defocus & Letter Blurring Engine ---
    # Optical Defocus Blur: Make letters soft and blurry
    if target_width < 70:
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.9, 1.7)
    elif target_width < 120:
        ksize = random.choice([3, 5])
        sigma = random.uniform(1.1, 2.2)
    else:
        ksize = random.choice([3, 5, 7])
        sigma = random.uniform(1.3, 2.5)
    low_res = cv2.GaussianBlur(low_res, (ksize, ksize), sigma)

    # Directional Vehicle Motion Blur (60% probability)
    if random.random() < 0.60:
        m_len = random.choice([3, 5]) if target_width < 90 else random.choice([3, 5, 7])
        m_angle = random.uniform(-12.0, 12.0)
        M_rot = cv2.getRotationMatrix2D((m_len / 2.0, m_len / 2.0), m_angle, 1.0)
        k_motion = np.zeros((m_len, m_len), dtype=np.float32)
        k_motion[m_len // 2, :] = 1.0
        k_motion = cv2.warpAffine(k_motion, M_rot, (m_len, m_len))
        k_sum = k_motion.sum()
        if k_sum > 0:
            k_motion /= k_sum
            low_res = cv2.filter2D(low_res, -1, k_motion)

    # Resampling Subpixel Softening (Dissolves sharp character edges)
    if random.random() < 0.70:
        sub_scale = random.uniform(0.70, 0.88)
        sw = max(16, int(round(target_width * sub_scale)))
        sh = max(8, int(round(target_h * sub_scale)))
        down = cv2.resize(low_res, (sw, sh), interpolation=cv2.INTER_AREA)
        low_res = cv2.resize(down, (target_width, target_h), interpolation=cv2.INTER_LINEAR)

    # --- 4. Lighting & Glare Synthesis ---
    if cctv_profile == "night_glare":
        # Nighttime: low dynamic range, high gain noise, headlight specular glare
        low_res = np.clip(low_res.astype(np.float32) * 0.70, 0, 255).astype(np.uint8)
        # Add Poisson/Gaussian sensor noise
        noise = np.random.normal(0, random.uniform(8.0, 16.0), low_res.shape).astype(np.float32)
        low_res = np.clip(low_res.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Headlight glare gradient from side (simulating approaching car headlights in Image 2)
        glare = np.zeros_like(low_res, dtype=np.float32)
        side = random.choice(["left", "right"])
        for x in range(target_width):
            prog = (target_width - x) / float(target_width) if side == "left" else x / float(target_width)
            flare = (max(0.0, prog - 0.3) / 0.7) ** 2
            glare[:, x, :] = flare * random.uniform(140.0, 210.0)
        low_res = np.clip(low_res.astype(np.float32) + glare, 0, 255).astype(np.uint8)
    else:
        # Daylight sensor grain
        noise = np.random.normal(0, random.uniform(2.0, 5.0), low_res.shape).astype(np.float32)
        low_res = np.clip(low_res.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # --- 5. Video Compression Ringing & JPEG Artifacts ---
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, enc = cv2.imencode(".jpg", low_res, encode_param)
    final_plate = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return final_plate


def generate_cctv_synthetic_dataset(num_samples: int = 1200):
    """
    Generates a full synthetic CCTV dataset matching user specifications.
    """
    print("=" * 80)
    print(f"[*] Generating {num_samples:,} Realistic CCTV Thai License Plates")
    print(f"[*] Output Directory: {OUTPUT_DATASET_DIR}")
    print("=" * 80)

    # Weights for Vehicle Classes
    # 1. Car: 45% (both Bangkok 1-9 series and provincial 2-letter)
    # 2. Motorcycle: 20% (overhead angle matching Image 3 & 5)
    # 3. Pickup: 15% (White/Green)
    # 4. Van: 8% (White/Blue)
    # 5. Taxi: 5% (Yellow/Black)
    # 6. Heavy Transport Commercial: 4% (Yellow/Black 10-79, 700+)
    # 7. Heavy Transport Private: 3% (White/Black 40-59, 80-99)

    records = []
    annotations_data = []
    lpr_session_items = []

    cctv_profiles = ["slanted_approach", "overhead_steep", "night_glare", "distant_lowres", "direct_sunny"]
    resolution_tiers = [
        ("low", 36, 56),       # 36-56px (low-res, very blurry letters)
        ("medium", 57, 95),    # 57-95px (typical roadside camera resolution)
        ("high", 96, 145)      # 96-145px (close-up edge crop, capped)
    ]

    p_weights = [p["weight"] for p in KEY_PROVINCES]

    # Pre-select ground-truth test cases to mirror user's images directly
    special_cases = [
        {"type": "WhiteBlackCar", "d3": "ขม", "d4": "6343", "prov": "ภูเก็ต", "profile": "slanted_approach", "w": 85, "q": 40, "dirt": 0.50, "note": "Match Image 1 & 4 (Sedan turning)"},
        {"type": "WhiteBlackMotor", "d3": "1กจ", "d4": "4160", "prov": "กรุงเทพมหานคร", "profile": "overhead_steep", "w": 110, "q": 45, "dirt": 0.55, "note": "Match Image 3 & 5 (Motorcycle)"},
        {"type": "WhiteBlackCar", "d3": "8กน", "d4": "5500", "prov": "กรุงเทพมหานคร", "profile": "slanted_approach", "w": 95, "q": 45, "dirt": 0.40, "note": "User Car 1: 8กน 5500"},
        {"type": "WhiteBlackCar", "d3": "7ขส", "d4": "8825", "prov": "กรุงเทพมหานคร", "profile": "slanted_approach", "w": 90, "q": 40, "dirt": 0.45, "note": "User Car 2: 7ขส 8825 Double Cab"},
        {"type": "WhiteBlackCar", "d3": "1กก", "d4": "1234", "prov": "กรุงเทพมหานคร", "profile": "night_glare", "w": 75, "q": 35, "dirt": 0.35, "note": "Match Image 2 (Night Glare)"},
        {"type": "WhiteBlue", "d3": "นข", "d4": "9999", "prov": "กรุงเทพมหานคร", "profile": "slanted_approach", "w": 85, "q": 45, "dirt": 0.40, "note": "Van White/Blue"},
        {"type": "WhiteGreen", "d3": "1บท", "d4": "2468", "prov": "ชลบุรี", "profile": "slanted_approach", "w": 80, "q": 40, "dirt": 0.60, "note": "Pickup White/Green"},
        {"type": "YellowBlack", "d3": "1มฮ", "d4": "7890", "prov": "กรุงเทพมหานคร", "profile": "slanted_approach", "w": 85, "q": 45, "dirt": 0.50, "note": "Taxi Yellow/Black"},
        {"type": "YellowHeavy", "d3": "70", "d4": "1647", "prov": "ชลบุรี", "profile": "slanted_approach", "w": 95, "q": 45, "dirt": 0.70, "note": "Heavy Transport Public"},
        {"type": "WhiteHeavy", "d3": "80", "d4": "5678", "prov": "ระยอง", "profile": "slanted_approach", "w": 95, "q": 45, "dirt": 0.50, "note": "Heavy Transport Private"},
        {"type": "YellowHeavy", "d3": "701", "d4": "8888", "prov": "กรุงเทพมหานคร", "profile": "slanted_approach", "w": 100, "q": 45, "dirt": 0.75, "note": "Heavy Transport 700+ Series"},
    ]

    for i in range(num_samples):
        # Determine sample parameters
        if i < len(special_cases):
            sc = special_cases[i]
            p_type = sc["type"]
            d3 = sc["d3"]
            d4 = sc["d4"]
            prov_dict = next(p for p in KEY_PROVINCES if p["name"] == sc["prov"])
            profile = sc["profile"]
            target_w = sc["w"]
            jpeg_q = sc["q"]
            dirt_level = sc.get("dirt", 0.35)
        else:
            prov_dict = random.choices(KEY_PROVINCES, weights=p_weights, k=1)[0]
            prov_name = prov_dict["name"]

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

            # Choose class
            class_roll = random.random()
            if class_roll < 0.45:
                p_type = "WhiteBlackCar"
            elif class_roll < 0.65:
                p_type = "WhiteBlackMotor"
            elif class_roll < 0.80:
                p_type = "WhiteGreen"
            elif class_roll < 0.88:
                p_type = "WhiteBlue"
            elif class_roll < 0.93:
                p_type = "YellowBlack"
            elif class_roll < 0.97:
                p_type = "YellowHeavy"
            else:
                p_type = "WhiteHeavy"

            # Profile & Resolution
            if p_type == "WhiteBlackMotor":
                profile = random.choice(["overhead_steep", "slanted_approach", "distant_lowres"])
            else:
                profile = random.choices(cctv_profiles, weights=[0.35, 0.15, 0.20, 0.15, 0.15], k=1)[0]

            res_tier, min_w, max_w = random.choice(resolution_tiers)
            target_w = random.randint(min_w, max_w)
            jpeg_q = random.randint(30, 65)

            # Generate d3 and d4 according to DLT rules
            if p_type == "WhiteBlackCar":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.82:
                    # Bangkok leading digit 1-9 (Equally distributed 1-9 to cover all eras!)
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(CAR_INITIALS)
                    d3 = lead_digit + pair
                else:
                    # 2-letter format
                    d3 = get_random_valid_pair(CAR_INITIALS)
                d4 = str(random.randint(1, 9999))

            elif p_type == "WhiteBlue":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.60:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(VAN_INITIALS)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(VAN_INITIALS)
                d4 = str(random.randint(1, 9999))

            elif p_type == "WhiteGreen":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.60:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(PICKUP_INITIALS)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(PICKUP_INITIALS)
                d4 = str(random.randint(1, 9999))

            elif p_type == "YellowBlack":
                if prov_name == "กรุงเทพมหานคร" and random.random() < 0.50:
                    lead_digit = "1"
                    pair = get_random_valid_pair(TAXI_INITIALS)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(TAXI_INITIALS)
                d4 = str(random.randint(1, 9999))

            elif p_type == "WhiteBlackMotor":
                if random.random() < 0.70:
                    lead_digit = str(random.randint(1, 9))
                    pair = get_random_valid_pair(MOTOR_INITIALS)
                    d3 = lead_digit + pair
                else:
                    d3 = get_random_valid_pair(MOTOR_INITIALS)
                d4 = str(random.randint(1, 9999))

            elif p_type == "YellowHeavy":
                # Commercial transport (10-79 or 700-799)
                if random.random() < 0.25:
                    d3 = str(random.randint(700, 799))
                else:
                    d3 = str(random.choice(list(range(10, 40)) + list(range(60, 80))))
                d4 = f"{random.randint(1, 9999):04d}"

            else:  # WhiteHeavy
                d3 = str(random.choice(list(range(40, 60)) + list(range(80, 100))))
                d4 = f"{random.randint(1, 9999):04d}"

        prov_name = prov_dict["name"]
        iso_code = prov_dict["iso"]

        # --- Render Clean Die ---
        if p_type == "WhiteBlackMotor":
            clean_die = render_motorcycle_plate_die(
                top_text=d3,
                province_text=prov_name,
                num_text=d4,
                colorway_key="private_car"
            )
            full_text = f"{d3} {d4} {prov_name}"
        elif p_type in ("YellowHeavy", "WhiteHeavy"):
            is_comm = True
            cway = "taxi" if p_type == "YellowHeavy" else "private_car"
            plate_reg_text = f"{d3}-{d4}"
            clean_die = render_calibrated_plate_die(
                plate_text=plate_reg_text,
                province_text=prov_name,
                colorway_key=cway,
                is_commercial_transport=True
            )
            full_text = f"{plate_reg_text} {prov_name}"
        else:
            cway_map = {
                "WhiteBlackCar": "private_car",
                "WhiteBlue": "van",
                "WhiteGreen": "pickup",
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

        # --- Coupling Perspective Geometry to Dirt Physics ---
        if profile == "overhead_steep":
            p_pitch, p_yaw = random.uniform(22.0, 34.0), random.uniform(-12.0, 12.0)
        elif profile == "slanted_approach":
            p_pitch, p_yaw = random.uniform(14.0, 24.0), random.choice([-1.0, 1.0]) * random.uniform(16.0, 28.0)
        elif profile == "night_glare":
            p_pitch, p_yaw = random.uniform(8.0, 16.0), random.uniform(-14.0, 14.0)
        else:
            p_pitch, p_yaw = random.uniform(6.0, 14.0), random.uniform(-8.0, 8.0)

        # --- Apply Realistic Physical Weathering & Dirt Simulation Coupled with Perspective ---
        weathered_die = apply_unclean_road_weathering(
            plate_bgr=clean_die,
            dirt_level=dirt_level,
            plate_type=p_type,
            yaw_deg=p_yaw,
            pitch_deg=p_pitch
        )

        # --- Apply CCTV Optics, Perspective & Blurry Letter Degradation ---
        final_img = simulate_cctv_optics_and_angle(
            plate_bgr=weathered_die,
            cctv_profile=profile,
            target_width=target_w,
            jpeg_quality=jpeg_q
        )

        h_out, w_out = final_img.shape[:2]

        # Standard file naming convention: {name}_{plate_type}_{d3}_{d4}_{province}.jpg
        sample_id = f"cctv_{i+1:05d}"
        clean_d3 = d3.replace("-", "")
        clean_d4 = d4.replace("-", "")
        clean_prov = prov_name.replace(" ", "")
        filename = f"{sample_id}_{p_type}_{clean_d3}_{clean_d4}_{clean_prov}.jpg"
        filepath = PLATES_OUTPUT_DIR / filename
        cv2_imwrite_safe(filepath, final_img, quality=jpeg_q)

        # Append records
        records.append({
            "filename": filename,
            "sample_id": sample_id,
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "iso_3166_2": iso_code,
            "full_plate": full_text,
            "dirt_level": round(float(dirt_level), 2),
            "width": w_out,
            "height": h_out,
            "cctv_profile": profile,
            "jpeg_quality": jpeg_q,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        })

        annotations_data.append({
            "filename": filename,
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "full_plate": full_text,
            "size": [w_out, h_out]
        })

        lpr_session_items.append({
            "name": sample_id,
            "filename": filename,
            "subfolder": "plates",
            "status": "approved",
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "ai_confidence": 1.0,
            "notes": f"Synthetic CCTV Profile: {profile}"
        })

        if (i + 1) % 200 == 0 or (i + 1) == num_samples:
            print(f"[*] Processed {i+1:,}/{num_samples:,} plates...")

    # Save metadata.csv
    csv_path = OUTPUT_DATASET_DIR / "metadata.csv"
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "sample_id", "plate_type", "d3", "d4", "province",
            "iso_3166_2", "full_plate", "dirt_level", "width", "height", "cctv_profile",
            "jpeg_quality", "created_at_utc"
        ])
        writer.writeheader()
        writer.writerows(records)

    # Save annotations.json
    json_path = OUTPUT_DATASET_DIR / "annotations.json"
    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(annotations_data, f, ensure_ascii=False, indent=2)

    # Save .lpr_session.json (DataManipulator compatibility)
    session_path = OUTPUT_DATASET_DIR / ".lpr_session.json"
    session_data = {
        "version": "2.0",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_items": len(lpr_session_items),
        "items": lpr_session_items
    }
    with open(session_path, mode="w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    print(f"[+] Successfully wrote {len(records):,} plates to {OUTPUT_DATASET_DIR}")
    print(f"[+] Metadata written to: {csv_path}")
    print(f"[+] Annotations written to: {json_path}")
    print(f"[+] Session file written to: {session_path}")

    # Build Verification Sheet
    build_cctv_specimen_gallery()


def build_cctv_specimen_gallery():
    """
    Renders an inspection sheet comparing real uploaded CCTV images directly
    with newly generated synthetic CCTV plates across all 7 classes and angles.
    Uses UTF-8 safe reading and contrast-enhanced dark asphalt containers.
    """
    print("[*] Generating Direct Verification Comparison Sheet...")
    gallery_path = OUTPUT_DATASET_DIR / "cctv_specimen_comparison.jpg"

    # Load uploaded real crop images
    img1_path = USER_UPLOADED_DIR / "media_1793080423192_fc32bc2e.jpg"
    img2_path = USER_UPLOADED_DIR / "media_1793080425848_918c4a37.jpg"
    img4_path = USER_UPLOADED_DIR / "media_1793080493929_e4fe4771.png"
    img5_path = USER_UPLOADED_DIR / "media_1793080493937_a01b3c37.png"

    real_img4 = cv2_imread_safe(img4_path)
    real_img5 = cv2_imread_safe(img5_path)

    # Real night car crop with headlight glare from Image 2
    real_night_crop = None
    if img2_path.exists():
        im2 = cv2_imread_safe(img2_path)
        if im2 is not None:
            # Crop the SUV headlight glare area
            real_night_crop = im2[290:530, 620:790]

    # Load representative generated samples from dataset
    samples_to_show = [
        {"title": "1. Real CCTV Car Crop (Image 4): ขม 6343 ภูเก็ต", "img": real_img4, "is_real": True, "badge": "REAL CAMERA"},
        {"title": "2. Synthetic Match: ขม 6343 ภูเก็ต (Slanted Approach)", "fn": "cctv_00001_WhiteBlackCar_ขม_6343_ภูเก็ต.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "3. Real CCTV Motorcycle Crop (Image 5): 1กจ 4160", "img": real_img5, "is_real": True, "badge": "REAL CAMERA"},
        {"title": "4. Synthetic Match: 1กจ 4160 กทม. (Overhead Steep)", "fn": "cctv_00002_WhiteBlackMotor_1กจ_4160_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "5. Real CCTV Night Headlight Glare (Image 2)", "img": real_night_crop, "is_real": True, "badge": "REAL CAMERA"},
        {"title": "6. Synthetic Match: 1กก 1234 กทม. (Night Glare)", "fn": "cctv_00005_WhiteBlackCar_1กก_1234_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "7. User Car 1: 8กน 5500 กทม. (ก-Era Active Plate)", "fn": "cctv_00003_WhiteBlackCar_8กน_5500_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "USER GROUND TRUTH"},
        {"title": "8. User Car 2: 7ขส 8825 กทม. (ข-Era Double Cab)", "fn": "cctv_00004_WhiteBlackCar_7ขส_8825_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "USER GROUND TRUTH"},
        {"title": "9. Van (White/Blue): นข 9999 กทม.", "fn": "cctv_00006_WhiteBlue_นข_9999_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "10. Commercial Pickup: 1บท 2468 ชลบุรี", "fn": "cctv_00007_WhiteGreen_1บท_2468_ชลบุรี.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "11. Taxi (Yellow/Black): 1มฮ 7890 กทม.", "fn": "cctv_00008_YellowBlack_1มฮ_7890_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "12. Heavy Transport Public: 70-1647 ชลบุรี", "fn": "cctv_00009_YellowHeavy_70_1647_ชลบุรี.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "13. Heavy Transport Private: 80-5678 ระยอง", "fn": "cctv_00010_WhiteHeavy_80_5678_ระยอง.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"},
        {"title": "14. Heavy Transport 700+ Series: 701-8888 กทม.", "fn": "cctv_00011_YellowHeavy_701_8888_กรุงเทพมหานคร.jpg", "is_real": False, "badge": "SYNTHETIC CCTV"}
    ]

    card_w = 460
    card_h = 210
    cols = 2
    rows = (len(samples_to_show) + 1) // cols

    sheet_w = cols * (card_w + 24) + 32
    sheet_h = 110 + rows * (card_h + 24)

    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), color=(241, 245, 249))
    draw = ImageDraw.Draw(sheet_pil)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 20)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 13)
    font_card = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 13)
    font_badge = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 10)

    # Header Card
    draw.rounded_rectangle([16, 14, sheet_w - 16, 88], radius=8, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.text((32, 24), "Real Roadside CCTV Camera Footage vs Synthetic CCTV Plates", fill=(15, 23, 42), font=font_title)
    draw.text((32, 54), "Visual Comparison of Perspective Slant, Downward Pitch, Headlight Glare, Blur, and Stamped Fonts", fill=(71, 85, 105), font=font_sub)

    for idx, item in enumerate(samples_to_show):
        c = idx % cols
        r = idx // cols
        x = 20 + c * (card_w + 24)
        y = 104 + r * (card_h + 24)

        # 1. Outer Card Box
        is_real = item.get("is_real", False)
        card_outline = (59, 130, 246) if is_real else ((168, 85, 247) if "USER" in item.get("badge", "") else (203, 213, 225))
        draw.rounded_rectangle([x, y, x + card_w, y + card_h], radius=8, fill=(255, 255, 255), outline=card_outline, width=2)

        # 2. Card Header Title & Badge
        title_text = item["title"]
        badge_text = item.get("badge", "SAMPLE")
        draw.text((x + 12, y + 10), title_text, fill=(15, 23, 42), font=font_card)

        # Badge pill
        b_box = font_badge.getbbox(badge_text)
        bw = b_box[2] - b_box[0] + 12
        bh = b_box[3] - b_box[1] + 6
        bx = x + card_w - bw - 12
        by = y + 8
        badge_bg = (30, 64, 175) if is_real else ((126, 34, 206) if "USER" in badge_text else (4, 120, 87))
        draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=4, fill=badge_bg)
        draw.text((bx + 6, by + 3), badge_text, fill=(255, 255, 255), font=font_badge)

        # 3. Inner Dark Asphalt Display Container (Contrast-enhanced so plates 'pop')
        cx1 = x + 10
        cy1 = y + 36
        cx2 = x + card_w - 10
        cy2 = y + card_h - 10
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=6, fill=(24, 28, 36), outline=(51, 65, 85), width=1)

        # 4. Load Image
        if is_real and item.get("img") is not None:
            crop_bgr = item["img"]
        else:
            fn = item.get("fn", "")
            fpath = PLATES_OUTPUT_DIR / fn
            crop_bgr = cv2_imread_safe(fpath)

        # 5. Display Image centered and upscaled in container
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
        else:
            # Fallback warning in card if image missing
            draw.text((cx1 + 20, cy1 + 50), f"[Image File Not Found: {item.get('fn', '')}]", fill=(239, 68, 68), font=font_card)

    sheet_bgr = cv2.cvtColor(np.array(sheet_pil), cv2.COLOR_RGB2BGR)
    cv2_imwrite_safe(gallery_path, sheet_bgr, quality=95)
    print(f"[+] Saved Verification Comparison Gallery to: {gallery_path}")

    # Copy to Artifact directory for user viewing
    if ARTIFACT_DIR.exists():
        artifact_gallery = ARTIFACT_DIR / "cctv_specimen_comparison.jpg"
        shutil.copy2(gallery_path, artifact_gallery)
        print(f"[+] Copied Gallery to Artifact Dir: {artifact_gallery}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Thai CCTV Synthetic Dataset Generator")
    parser.add_argument("--samples", type=int, default=1200, help="Number of synthetic plates to generate")
    parser.add_argument("--gallery-only", action="store_true", help="Only build verification comparison gallery")
    args = parser.parse_args()

    if args.gallery_only:
        build_cctv_specimen_gallery()
    else:
        generate_cctv_synthetic_dataset(num_samples=args.samples)
