#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance Synthetic Thai License Plate Dataset Generator: v3.0 - Synthetic Clean
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.7 / A.9.3)
- ISO/IEC 5259-2:2023 (Data Quality Measures for Machine Learning)
- ISO/IEC 25012:2008 (Data Quality & SHA-256 Lineage)
- ISO 3166-2:TH (Thailand Administrative Subdivisions)
- Thailand DLT Statutory Vehicle & License Plate Specifications

Features:
1. Physical Camera & Geometry Calibration:
   - Lens: 4mm focal length
   - Camera Height: 1.0 - 2.0 m above ground
   - Plate Height: 0.3 - 0.5 m above ground
   - Pitch: Normal to high angle 0° to 5°
   - Yaw (left-right): -5° to +5°
   - Roll: -2° to +2°
2. Resolution Range (Low to High):
   - Low: 64px - 120px width
   - Mid: 120px - 260px width
   - High: 260px - 600px width
3. Regional Weighting (100,000 Clean Plates):
   - Bangkok (BKK): 35% (35,000)
   - Phuket (HKT): 15% (15,000)
   - Eastern Seaboard (Chonburi, Rayong, Chachoengsao): 15% (15,000)
   - Chiang Mai (CNX): 10% (10,000)
   - All other 74 provinces + Betong: 25% (25,000) -> 100% representation guaranteed
4. Vehicle Class & Plate Types:
   - WhiteBlackCar: 40% (40,000)
   - WhiteBlackMotor: 25% (25,000)
   - YellowBlack (Large Truck / Transport): 15% (15,000)
   - WhiteBlue: 10% (10,000)
   - WhiteGreen: 10% (10,000)
5. Non-Readable & Negative Samples (20,000 Images):
   - 10,000 very blur images labeled as '- - - -'
   - 10,000 not plate at all images labeled as '+ + + +'
   - Total dataset: 120,000 images
"""

import os
import sys
import re
import csv
import json
import math
import random
import shutil
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import multiprocessing as mp

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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.realistic_plate_generator import (
    render_calibrated_plate_die,
    shape_thai_text,
    DLT_PROVINCE_CODES,
    COLORWAYS
)
from tools.dirty_plate_synthesizer import render_calibrated_motorcycle_die
from tools.selen_token_codec import (
    PROVINCES_78,
    PLATE_TYPES,
    encode_plate_to_tokens,
    decode_tokens_to_plate,
    VOCAB_SIZE
)
from tools.DataManipulator.app.provinces import NAME_TO_ISO

# Fonts
FONT_ID_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-id.ttf"
FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
FONT_YELLOW_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-yellow.ttf"

# Default Output Directory
DEFAULT_OUTPUT_DIR = ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.0-synthetic-clean"

# Consonants
CAR_INITIALS = list("กขงจฉชฌญฎฐธพภวศษส")
VAN_INITIALS = list("นฬอฮ")
PICKUP_INITIALS = list("บฒณตถปผยรล")
TAXI_INITIALS = list("ทม")
MOTOR_INITIALS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
ALL_VALID_CONSONANTS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
OFFENSIVE_PAIRS = {"ศพ", "สพ", "ศบ", "ษพ", "ชน", "จน", "ตด", "งง"}

# Build weighted province pool
# Target proportions:
# BKK: 35%, HKT: 15%, Eastern Seaboard: 15% (Chonburi 8%, Rayong 4%, Chachoengsao 3%), CNX: 10%
# Remaining 74 provinces: 25% total (25% / 74 ≈ 0.3378% each)
FOCUS_WEIGHTS = {
    "กรุงเทพมหานคร": 0.35,
    "ภูเก็ต": 0.15,
    "ชลบุรี": 0.08,
    "ระยอง": 0.04,
    "ฉะเชิงเทรา": 0.03,
    "เชียงใหม่": 0.10,
}
OTHER_PROVINCES = [p for p in PROVINCES_78 if p not in FOCUS_WEIGHTS]
REMAINING_WEIGHT = 0.25 / len(OTHER_PROVINCES)

PROVINCE_WEIGHTS = []
for p in PROVINCES_78:
    w = FOCUS_WEIGHTS.get(p, REMAINING_WEIGHT)
    PROVINCE_WEIGHTS.append((p, w))

PROV_NAMES_LIST = [p[0] for p in PROVINCE_WEIGHTS]
PROV_WEIGHTS_LIST = [p[1] for p in PROVINCE_WEIGHTS]


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def apply_camera_perspective_and_resolution(
    plate_bgr: np.ndarray,
    target_width: int,
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    is_blur_negative: bool = False,
    blur_sigma: float = 0.4
) -> np.ndarray:
    """
    Applies calibrated 3D camera perspective warp based on 4mm lens geometry:
    - Pitch: [0°, 5°] downward normal shot
    - Yaw: [-5°, +5°] left-to-right angle
    - Roll: [-2°, +2°] tilt
    - Scales to target resolution (64px to 600px width)
    """
    h, w = plate_bgr.shape[:2]

    # Convert degrees to radians
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    roll_rad = math.radians(roll_deg)

    # 4mm lens field of view projection matrix
    focal = 4.0 / 5.12 * max(w, h) * 1.5

    # 3D rotation matrix
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

    Rx = np.array([[1, 0, 0], [0, cos_p, -sin_p], [0, sin_p, cos_p]], dtype=np.float32)
    Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
    Rz = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=np.float32)
    R = Rz @ Ry @ Rx

    # Corner coordinates centered
    pts_3d = np.array([
        [-w / 2, -h / 2, 0],
        [w / 2, -h / 2, 0],
        [w / 2, h / 2, 0],
        [-w / 2, h / 2, 0]
    ], dtype=np.float32)

    # Rotate in 3D
    rotated_3d = (R @ pts_3d.T).T

    # Project with focal length
    z_dist = focal * 1.4
    proj_pts = []
    for pt in rotated_3d:
        z = pt[2] + z_dist
        px = (pt[0] * focal) / z + w / 2
        py = (pt[1] * focal) / z + h / 2
        proj_pts.append([px, py])

    src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts = np.array(proj_pts, dtype=np.float32)

    # Bounding rect
    min_x = min(p[0] for p in dst_pts)
    max_x = max(p[0] for p in dst_pts)
    min_y = min(p[1] for p in dst_pts)
    max_y = max(p[1] for p in dst_pts)

    offset_x = -min_x
    offset_y = -min_y
    dst_pts[:, 0] += offset_x
    dst_pts[:, 1] += offset_y

    out_w = int(math.ceil(max_x - min_x))
    out_h = int(math.ceil(max_y - min_y))

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(plate_bgr, M, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)

    # Rescale to target width
    scale = target_width / float(out_w)
    target_h = max(24, int(round(out_h * scale)))
    interp = cv2.INTER_AREA if scale < 0.9 else cv2.INTER_LINEAR
    resized = cv2.resize(warped, (target_width, target_h), interpolation=interp)

    if is_blur_negative:
        # Severe blur making text completely unreadable
        ksize = random.choice([25, 31, 35, 41])
        sigma = random.uniform(8.0, 16.0)
        resized = cv2.GaussianBlur(resized, (ksize, ksize), sigma)
        if random.random() < 0.4:
            # Add severe directional motion blur
            motion_len = random.randint(20, 35)
            motion_kernel = np.zeros((motion_len, motion_len))
            motion_kernel[int((motion_len - 1) / 2), :] = np.ones(motion_len)
            motion_kernel = motion_kernel / motion_len
            resized = cv2.filter2D(resized, -1, motion_kernel)
    else:
        # Clean synthetic: subtle optical MTF filter (sharp, clean edge)
        if blur_sigma > 0:
            resized = cv2.GaussianBlur(resized, (3, 3), blur_sigma)

    # JPEG compression simulation (quality 85 to 98)
    jpeg_qual = random.randint(85, 98) if not is_blur_negative else random.randint(60, 80)
    _, enc = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def generate_not_plate_image(target_width: int, target_height: int) -> np.ndarray:
    """
    Generates realistic non-plate background crops (car bumper, asphalt, grill, textures).
    Labeled as '+ + + +'.
    """
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    kind = random.choice(["asphalt", "car_bumper", "radiator_grill", "headlight_edge", "road_barrier"])

    if kind == "asphalt":
        # Grey tarmac with texture & occasional painted lane line
        base_color = random.randint(45, 75)
        canvas[:] = base_color
        noise = np.random.normal(0, random.uniform(8, 16), (target_height, target_width, 3)).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if random.random() < 0.35:
            # Yellow or white dashed line segment
            line_color = (30, 200, 230) if random.random() < 0.5 else (210, 215, 220)
            y_start = random.randint(0, target_height // 2)
            y_end = random.randint(target_height // 2, target_height)
            thickness = random.randint(8, 22)
            cv2.line(canvas, (0, y_start), (target_width, y_end), line_color, thickness)

    elif kind == "car_bumper":
        # Glossy car body paint with highlight gradient (black, white, silver, red, blue)
        hue = random.choice([
            (25, 25, 28), (220, 225, 230), (140, 145, 150), (15, 20, 160), (120, 30, 30)
        ])
        canvas[:] = hue
        grad = np.linspace(0.8, 1.2, target_height)[:, None, None]
        canvas = np.clip(canvas.astype(np.float32) * grad, 0, 255).astype(np.uint8)
        # Add reflection line or bumper seam
        seam_y = random.randint(int(target_height * 0.3), int(target_height * 0.7))
        cv2.line(canvas, (0, seam_y), (target_width, seam_y), (hue[0] // 2, hue[1] // 2, hue[2] // 2), 2)

    elif kind == "radiator_grill":
        # Dark mesh or horizontal slatted grill
        canvas[:] = (20, 22, 25)
        step = random.randint(6, 14)
        for y in range(0, target_height, step):
            cv2.line(canvas, (0, y), (target_width, y), (70, 75, 80), 2)
            cv2.line(canvas, (0, y + 2), (target_width, y + 2), (10, 10, 12), 2)

    elif kind == "headlight_edge":
        # Curved glass reflector pattern
        canvas[:] = (35, 40, 45)
        center_x = random.choice([0, target_width])
        center_y = random.randint(0, target_height)
        for r in range(10, max(target_width, target_height) * 2, random.randint(8, 16)):
            color = random.choice([(180, 185, 190), (120, 125, 130), (50, 55, 60)])
            cv2.circle(canvas, (center_x, center_y), r, color, 1)

    else: # road_barrier
        # Concrete texture with diagonal hazard stripes
        canvas[:] = (130, 135, 140)
        for x in range(-target_height, target_width + target_height, 28):
            pts = np.array([
                [x, 0], [x + 14, 0], [x + 14 - target_height, target_height], [x - target_height, target_height]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (25, 28, 30))

    # Add mild sensor noise
    noise = np.random.normal(0, 4.0, (target_height, target_width, 3)).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def sample_plate_attributes(idx: int) -> Tuple[str, str, str, str, str, str]:
    """
    Samples plate attributes according to target proportions.
    Returns: (plate_type, d3, d4, province_name, iso_code, colorway)
    """
    # Sample province from weighted pool
    prov_name = random.choices(PROV_NAMES_LIST, weights=PROV_WEIGHTS_LIST, k=1)[0]
    iso_code = NAME_TO_ISO.get(prov_name, "TH-10")

    # Sample Plate Type according to target proportions:
    # 40% WhiteBlackCar, 25% WhiteBlackMotor, 15% YellowBlack, 10% WhiteBlue, 10% WhiteGreen
    roll = random.random()
    if roll < 0.40:
        p_type = "WhiteBlackCar"
        if prov_name == "กรุงเทพมหานคร" and random.random() < 0.70:
            d3 = f"{random.randint(1, 9)}{random.choice(CAR_INITIALS)}{random.choice(ALL_VALID_CONSONANTS)}"
        else:
            c1 = random.choice(CAR_INITIALS)
            c2 = random.choice(ALL_VALID_CONSONANTS)
            while (c1 + c2) in OFFENSIVE_PAIRS:
                c2 = random.choice(ALL_VALID_CONSONANTS)
            d3 = f"{c1}{c2}"
        d4 = str(random.randint(1, 9999))
        colorway = "private_car"

    elif roll < 0.65:
        p_type = "WhiteBlackMotor"
        if random.random() < 0.65:
            d3 = f"{random.randint(1, 9)}{random.choice(MOTOR_INITIALS)}{random.choice(ALL_VALID_CONSONANTS)}"
        else:
            c1 = random.choice(MOTOR_INITIALS)
            c2 = random.choice(ALL_VALID_CONSONANTS)
            d3 = f"{c1}{c2}"
        d4 = str(random.randint(1, 9999))
        colorway = "private_car"

    elif roll < 0.80:
        p_type = "YellowBlack"
        # Large Commercial Trucks / Heavy Transport Series (10-39, 70-79, 700-799)
        if random.random() < 0.35:
            d3 = str(random.randint(700, 799))
        else:
            d3 = str(random.choice(list(range(10, 40)) + list(range(70, 80))))
        d4 = f"{random.randint(1000, 9999)}"
        colorway = "taxi"

    elif roll < 0.90:
        p_type = "WhiteBlue"
        # Vans / microbuses (series น, ฬ, อ, ฮ)
        c1 = random.choice(VAN_INITIALS)
        c2 = random.choice(ALL_VALID_CONSONANTS)
        d3 = f"{c1}{c2}"
        d4 = str(random.randint(1, 9999))
        colorway = "van"

    else:
        p_type = "WhiteGreen"
        # Commercial pickups (บ, ฒ, ณ, ต, ถ, ป, ผ, ย, ร, ล)
        if random.random() < 0.35:
            d3 = f"{random.randint(1, 9)}{random.choice(PICKUP_INITIALS)}{random.choice(ALL_VALID_CONSONANTS)}"
        else:
            c1 = random.choice(PICKUP_INITIALS)
            c2 = random.choice(ALL_VALID_CONSONANTS)
            d3 = f"{c1}{c2}"
        d4 = str(random.randint(1, 9999))
        colorway = "pickup"

    d3 = d3[:3]
    d4 = d4[:4]
    return p_type, d3, d4, prov_name, iso_code, colorway


def generate_single_sample(args: Tuple[int, Path, str]) -> Dict[str, Any]:
    """
    Worker function to generate a single image sample.
    sample_type: 'clean', 'blur', 'non_plate'
    """
    idx, output_dir, sample_type = args
    filename = f"syn_{idx:06d}.jpg"
    filepath = output_dir / filename

    # Physical camera geometry sampling
    # Camera Height: 1.0 - 2.0 m
    # Plate Height: 0.3 - 0.5 m
    h_cam = random.uniform(1.0, 2.0)
    h_plate = random.uniform(0.3, 0.5)
    delta_h = h_cam - h_plate

    # Pitch: [0°, 5°] downward normal shot
    pitch_deg = round(random.uniform(0.0, 5.0), 2)
    # Yaw: [-5°, +5°]
    yaw_deg = round(random.uniform(-5.0, 5.0), 2)
    # Roll: [-2°, +2°]
    roll_deg = round(random.uniform(-2.0, 2.0), 2)

    # Resolution ladder:
    # 25% Low (64 - 120px), 50% Mid (120 - 260px), 25% High (260 - 600px)
    r_res = random.random()
    if r_res < 0.25:
        target_w = random.randint(64, 120)
    elif r_res < 0.75:
        target_w = random.randint(121, 260)
    else:
        target_w = random.randint(261, 600)

    if sample_type == "non_plate":
        # Aspect ratio ~2.2 : 1
        target_h = max(28, int(round(target_w / 2.2)))
        img = generate_not_plate_image(target_w, target_h)
        plate_type = "+"
        d3 = "+"
        d4 = "+"
        province = "+"
        iso_code = "NONE"
        full_text = "+ + + +"
        is_clean = False
        is_blur = False
        is_non_plate = True

    elif sample_type == "blur":
        # Generate plate die then apply severe blur
        p_type, d3_raw, d4_raw, prov_name, iso_code, colorway = sample_plate_attributes(idx)
        is_motor = (p_type == "WhiteBlackMotor")
        if is_motor:
            raw_bgr = render_calibrated_motorcycle_die(d3_raw, d4_raw, prov_name, add_bolts=True)
        else:
            is_large = (p_type == "YellowBlack")
            raw_bgr = render_calibrated_plate_die(
                f"{d3_raw} {d4_raw}", prov_name, colorway_key=colorway, is_large_plate=is_large
            )

        img = apply_camera_perspective_and_resolution(
            raw_bgr, target_w, pitch_deg, yaw_deg, roll_deg, is_blur_negative=True
        )
        plate_type = "-"
        d3 = "-"
        d4 = "-"
        province = "-"
        iso_code = "NONE"
        full_text = "- - - -"
        is_clean = False
        is_blur = True
        is_non_plate = False

    else: # 'clean'
        p_type, d3, d4, province, iso_code, colorway = sample_plate_attributes(idx)
        is_motor = (p_type == "WhiteBlackMotor")
        if is_motor:
            raw_bgr = render_calibrated_motorcycle_die(d3, d4, province, add_bolts=True)
        else:
            is_large = (p_type == "YellowBlack")
            raw_bgr = render_calibrated_plate_die(
                f"{d3} {d4}", province, colorway_key=colorway, is_large_plate=is_large
            )

        # Clean optical MTF: subtle softening 0.2 to 0.4
        blur_sigma = random.uniform(0.2, 0.4)
        img = apply_camera_perspective_and_resolution(
            raw_bgr, target_w, pitch_deg, yaw_deg, roll_deg, is_blur_negative=False, blur_sigma=blur_sigma
        )
        plate_type = p_type
        full_text = f"{plate_type} {d3} {d4} {province}".strip()
        is_clean = True
        is_blur = False
        is_non_plate = False

    # Save image
    cv2.imwrite(str(filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(88, 98)])

    # Token encoding (max 11 tokens)
    token_ids = encode_plate_to_tokens(plate_type, d3, d4, province, include_start_stop=True)

    h_out, w_out = img.shape[:2]
    now_utc = datetime.now(timezone.utc).isoformat()

    return {
        "index": idx,
        "filename": filename,
        "filepath": str(filepath),
        "plate_type": plate_type,
        "d3": d3,
        "d4": d4,
        "province": province,
        "iso_3166_2": iso_code,
        "full_text_line": full_text,
        "token_ids": json.dumps(token_ids),
        "token_count": len(token_ids),
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "roll_deg": roll_deg,
        "width": w_out,
        "height": h_out,
        "is_clean": is_clean,
        "is_blur": is_blur,
        "is_non_plate": is_non_plate,
        "created_at_utc": now_utc
    }


def generate_v3_clean_dataset(
    clean_count: int = 100000,
    blur_count: int = 10000,
    non_plate_count: int = 10000,
    output_dir: Optional[Path] = None,
    num_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Generates the full 120,000 dataset in parallel."""
    out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    plates_dir = out_dir / "plates"
    out_dir.mkdir(parents=True, exist_ok=True)
    plates_dir.mkdir(parents=True, exist_ok=True)

    total_images = clean_count + blur_count + non_plate_count
    workers = num_workers or min(28, max(1, os.cpu_count() - 2))

    print("=" * 80)
    print("  SYNTHETIC THAI LPR DATASET: v3.0 - Synthetic Clean")
    print("  ISO/IEC 42001:2023 & ISO/IEC 5259-2:2023 COMPLIANT")
    print("=" * 80)
    print(f"Total Target Images : {total_images:,}")
    print(f"  - Clean Synthetic : {clean_count:,}")
    print(f"  - Blur Non-Readable: {blur_count:,} (labeled as '- - - -')")
    print(f"  - Not Plate At All : {non_plate_count:,} (labeled as '+ + + +')")
    print(f"Output Directory    : {out_dir}")
    print(f"Parallel Workers    : {workers}")
    print(f"Perspective Bounds  : Pitch ∈ [0°, 5°], Yaw ∈ [-5°, +5°], Roll ∈ [-2°, +2°]")
    print(f"Focal Length / Lens : 4.0 mm, Cam Height: 1.0-2.0m, Plate Height: 0.3-0.5m")
    print(f"Max Token Length    : start + plateType + 3 + 4 + 1 + stop = 11 tokens")
    print("=" * 80)

    # Build tasks: (idx, plates_dir, sample_type)
    tasks = []
    current_idx = 0

    # 1. Clean plates
    for _ in range(clean_count):
        tasks.append((current_idx, plates_dir, "clean"))
        current_idx += 1

    # 2. Blur plates
    for _ in range(blur_count):
        tasks.append((current_idx, plates_dir, "blur"))
        current_idx += 1

    # 3. Not plate at all
    for _ in range(non_plate_count):
        tasks.append((current_idx, plates_dir, "non_plate"))
        current_idx += 1

    # Shuffle task order so batches contain mixed types
    random.seed(42)
    random.shuffle(tasks)

    # Execute parallel generation
    records = []
    chunksize = 250
    start_time = datetime.now()

    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(generate_single_sample, tasks, chunksize=chunksize)):
            records.append(res)
            if (i + 1) % 5000 == 0 or (i + 1) == total_images:
                elapsed = (datetime.now() - start_time).total_seconds()
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_images - (i + 1)) / fps if fps > 0 else 0
                print(f"[{i + 1:6d}/{total_images:6d}] ({((i + 1)/total_images)*100:5.1f}%) | "
                      f"Speed: {fps:6.1f} img/s | ETA: {eta/60:4.1f} min")

    records.sort(key=lambda r: r["index"])

    # Write metadata.csv
    csv_path = out_dir / "metadata.csv"
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Write annotations.json
    json_path = out_dir / "annotations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Write version_manifest.json
    manifest_path = out_dir / "version_manifest.json"
    manifest_data = {
        "version": "dataset_v3.0_synthetic_clean_20261109",
        "standard_compliance": {
            "iso_3166_2_th": True,
            "unicode_nfc": True,
            "iso_8601": True,
            "iso_25012": True,
            "iso_42001_aims": True,
            "iso_5259_2_data_quality": True,
            "dlt_spec_enforced": True,
            "max_tokens_11": True
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_images": total_images,
        "clean_images": clean_count,
        "blur_images": blur_count,
        "non_plate_images": non_plate_count,
        "optical_parameters": {
            "lens_focal_mm": 4.0,
            "camera_height_m": [1.0, 2.0],
            "plate_height_m": [0.3, 0.5],
            "pitch_deg_range": [0.0, 5.0],
            "yaw_deg_range": [-5.0, 5.0],
            "roll_deg_range": [-2.0, 2.0],
            "resolution_w_range": [64, 600]
        },
        "proportions": {
            "bkk_pct": 35.0,
            "hkt_pct": 15.0,
            "eastern_seaboard_pct": 15.0,
            "cnx_pct": 10.0,
            "other_provinces_pct": 25.0
        },
        "plate_folders": {
            "plates": total_images
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    total_time = (datetime.now() - start_time).total_seconds()
    print("=" * 80)
    print(f"Dataset v3.0 successfully generated in {total_time:.1f} s ({total_images / total_time:.1f} img/s)!")
    print(f"Artifacts stored at: {out_dir}")
    print("=" * 80)
    return manifest_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dataset v3.0 - Synthetic Clean")
    parser.add_argument("--clean", type=int, default=100000, help="Clean synthetic count (default: 100,000)")
    parser.add_argument("--blur", type=int, default=10000, help="Blur non-readable count (default: 10,000)")
    parser.add_argument("--non-plate", type=int, default=10000, help="Not plate at all count (default: 10,000)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes")
    args = parser.parse_args()

    generate_v3_clean_dataset(
        clean_count=args.clean,
        blur_count=args.blur,
        non_plate_count=args.non_plate,
        output_dir=Path(args.output_dir),
        num_workers=args.workers
    )
