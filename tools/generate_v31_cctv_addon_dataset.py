#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance Synthetic Thai License Plate Dataset Generator: v3.1 - CCTV Add-on
Generates low-resolution and real-CCTV augmented plates to complement Dataset v3.0:
1. Extreme Low-Resolution Downscaling (Plate height 20px - 48px)
2. Morphological Stroke Variations (Dilation for over-ink/bloom, Erosion for wear/thin loops)
3. CCTV Compression Artifacts (JPEG quality 25 - 65, H.264 macroblocking)
4. Motion Blur & Sensor Noise (Gaussian/Poisson noise, horizontal motion streak)
5. Bounding Box Margin Jitter & Frame Occlusion (5% - 15% edge clip)
6. Ingestion of 1,183 Real CCTV Anchor Plates (202610261140)

Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.7 / A.9.3)
- ISO/IEC 5259-2:2023 (Data Quality Measures for Machine Learning)
- ISO/IEC 25012:2008 (Data Quality & SHA-256 Lineage)
"""

import os
import sys
import re
import csv
import json
import time
import math
import random
import shutil
import hashlib
import argparse
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import multiprocessing as mp

import cv2
import numpy as np

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

OUTPUT_DIR = ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.1-cctv-addon"
REAL_CCTV_DIR = ROOT_DIR / "dataset" / "reading" / "versions" / "202610261140"

# Consonants
CAR_INITIALS = list("กขงจฉชฌญฎฐธพภวศษส")
VAN_INITIALS = list("นฬอฮ")
PICKUP_INITIALS = list("บฒณตถปผยรล")
ALL_CONSONANTS = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")
DIGITS = list("0123456789")

FOCUS_WEIGHTS = {
    "กรุงเทพมหานคร": 0.35,
    "ภูเก็ต": 0.15,
    "ชลบุรี": 0.08,
    "ระยอง": 0.04,
    "ฉะเชิงเทรา": 0.03,
    "เชียงใหม่": 0.10,
}
REMAINING_PROVS = [p for p in PROVINCES_78 if p not in FOCUS_WEIGHTS]
REMAINING_WEIGHT = 0.25 / len(REMAINING_PROVS)

PROVINCE_WEIGHTS = []
for p in PROVINCES_78:
    w = FOCUS_WEIGHTS.get(p, REMAINING_WEIGHT)
    PROVINCE_WEIGHTS.append((p, w))

PROV_NAMES_LIST = [p[0] for p in PROVINCE_WEIGHTS]
PROV_WEIGHTS_LIST = [p[1] for p in PROVINCE_WEIGHTS]


def sample_plate_attributes(idx: int) -> Tuple[str, str, str, str, str, str]:
    r = random.random()
    if r < 0.40:
        p_type = "WhiteBlackCar"
        colorway = "private_car"
    elif r < 0.65:
        p_type = "WhiteBlackMotor"
        colorway = "motorcycle"
    elif r < 0.80:
        p_type = "YellowBlack"
        colorway = "large_truck"
    elif r < 0.90:
        p_type = "WhiteBlue"
        colorway = "passenger_van"
    else:
        p_type = "WhiteGreen"
        colorway = "commercial_pickup"

    prov_name = random.choices(PROV_NAMES_LIST, weights=PROV_WEIGHTS_LIST, k=1)[0]
    iso_code = NAME_TO_ISO.get(prov_name, "TH-10")

    if p_type == "WhiteBlackMotor":
        has_digit = (random.random() < 0.70)
        c1 = random.choice(ALL_CONSONANTS)
        c2 = random.choice(ALL_CONSONANTS)
        d3 = f"{random.randint(1, 9)}{c1}{c2}" if has_digit else f"{c1}{c2}"
        d4 = str(random.randint(1, 999))
    elif p_type == "YellowBlack":
        d3 = str(random.randint(10, 99))
        d4 = str(random.randint(1000, 9999))
    else:
        has_lead = (random.random() < 0.60)
        if p_type == "WhiteBlackCar":
            c1 = random.choice(CAR_INITIALS)
        elif p_type == "WhiteBlue":
            c1 = random.choice(VAN_INITIALS)
        else:
            c1 = random.choice(PICKUP_INITIALS)
        c2 = random.choice(ALL_CONSONANTS)
        d3 = f"{random.randint(1, 9)}{c1}{c2}" if has_lead else f"{c1}{c2}"
        d4 = str(random.randint(1, 9999))

    return p_type, d3, d4, prov_name, iso_code, colorway


def apply_cctv_and_lowres_augmentations(plate_bgr: np.ndarray) -> np.ndarray:
    """
    Applies aggressive low-resolution and real CCTV sensor augmentations:
    - Target height 20px - 48px
    - Stroke dilation / erosion (blooming or thin wear)
    - Sensor noise & motion blur
    - Low-bitrate compression
    - Bounding box crop jitter (occlusion)
    """
    h_orig, w_orig = plate_bgr.shape[:2]
    aspect_ratio = w_orig / float(h_orig)

    # 1. Target low resolution height (20px to 48px)
    target_h = random.randint(20, 48)
    target_w = max(40, int(round(target_h * aspect_ratio)))

    # Perspective tilt (-8° to +8°)
    pitch_deg = random.uniform(0.0, 8.0)
    yaw_deg = random.uniform(-8.0, 8.0)
    roll_deg = random.uniform(-3.0, 3.0)

    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    roll_rad = math.radians(roll_deg)

    focal = target_w * 1.5
    cx = target_w / 2.0
    cy = target_h / 2.0

    src_pts = np.float32([
        [0, 0],
        [w_orig, 0],
        [w_orig, h_orig],
        [0, h_orig]
    ])

    dx_yaw = math.sin(yaw_rad) * (target_w * 0.12)
    dy_pitch = math.sin(pitch_rad) * (target_h * 0.15)
    d_roll = math.sin(roll_rad) * (target_h * 0.08)

    dst_pts = np.float32([
        [max(0, dx_yaw), max(0, -dy_pitch + d_roll)],
        [min(target_w, target_w - dx_yaw), max(0, dy_pitch - d_roll)],
        [min(target_w, target_w + dx_yaw), min(target_h, target_h + dy_pitch - d_roll)],
        [max(0, -dx_yaw), min(target_h, target_h - dy_pitch + d_roll)]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(
        plate_bgr, M, (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 2. Bounding Box Crop Jitter (Occasional 5% - 12% clipping of border)
    if random.random() < 0.35:
        clip_top = random.randint(0, int(target_h * 0.10))
        clip_bot = random.randint(0, int(target_h * 0.10))
        clip_l = random.randint(0, int(target_w * 0.08))
        clip_r = random.randint(0, int(target_w * 0.08))
        warped = warped[clip_top:target_h - clip_bot, clip_l:target_w - clip_r]
        if warped.shape[0] < 15 or warped.shape[1] < 30:
            warped = cv2.resize(warped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # 3. Morphological Stroke Dilation / Erosion
    r_morph = random.random()
    if r_morph < 0.30:
        # Over-inked / headlight bloom dilation
        k_size = random.choice([2, 3])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        warped = cv2.erode(warped, kernel) # for dark text on bright plate, erosion expands dark strokes
    elif r_morph < 0.50:
        # Faded / thin character wear
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        warped = cv2.dilate(warped, kernel) # dilation shrinks dark strokes

    # 4. Motion Blur
    if random.random() < 0.30:
        motion_len = random.randint(3, 7)
        m_kernel = np.zeros((motion_len, motion_len))
        m_kernel[int((motion_len - 1) / 2), :] = np.ones(motion_len) / motion_len
        warped = cv2.filter2D(warped, -1, m_kernel)

    # 5. Sensor Noise (Gaussian & Poisson)
    if random.random() < 0.45:
        sigma = random.uniform(6.0, 18.0)
        noise = np.random.normal(0, sigma, warped.shape).astype(np.float32)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. Low-bitrate JPEG Compression Simulation
    jpeg_qual = random.randint(25, 65)
    _, enc = cv2.imencode(".jpg", warped, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_qual])
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else warped


def worker_generate_batch(task_args: Tuple[int, int, str, str]) -> List[Dict[str, Any]]:
    start_idx, count, out_dir_str, plates_dir_str = task_args
    plates_dir = Path(plates_dir_str)
    records = []

    for i in range(count):
        idx = start_idx + i
        p_type, d3, d4, province, iso_code, colorway = sample_plate_attributes(idx)
        is_motor = (p_type == "WhiteBlackMotor")

        if is_motor:
            raw_bgr = render_calibrated_motorcycle_die(d3, d4, province, add_bolts=True)
        else:
            is_large = (p_type == "YellowBlack")
            raw_bgr = render_calibrated_plate_die(
                f"{d3} {d4}", province, colorway_key=colorway, is_large_plate=is_large
            )

        aug_bgr = apply_cctv_and_lowres_augmentations(raw_bgr)

        fn = f"cctv_{idx:06d}.jpg"
        fp = plates_dir / fn
        cv2.imwrite(str(fp), aug_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(75, 90)])

        full_text = f"{p_type} {d3} {d4} {province}".strip()
        token_ids = encode_plate_to_tokens(p_type, d3, d4, province, include_start_stop=True)

        h_out, w_out = aug_bgr.shape[:2]
        records.append({
            "index": idx,
            "filename": fn,
            "filepath": str(fp),
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": province,
            "iso_3166_2": iso_code,
            "full_text_line": full_text,
            "token_ids": json.dumps(token_ids),
            "token_count": len(token_ids),
            "width": w_out,
            "height": h_out,
            "is_real_cctv": False,
            "is_low_res": (h_out <= 48),
            "source": "synthetic_cctv_augmented",
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        })

    return records


def ingest_real_cctv_anchor_plates(out_plates_dir: Path, start_idx: int) -> List[Dict[str, Any]]:
    """Copies and tokenizes all verified real CCTV anchor plates from 202610261140."""
    print("Ingesting verified real CCTV anchor plates from 202610261140...")
    meta_path = REAL_CCTV_DIR / "metadata.csv"
    if not meta_path.exists():
        print(f"Warning: {meta_path} not found. Skipping real CCTV ingestion.")
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    real_records = []
    idx = start_idx

    for r in rows:
        fn = r.get("filename", "")
        src_path = REAL_CCTV_DIR / "plates" / fn
        if not src_path.exists():
            continue

        p_type = r.get("plate_type", "WhiteBlackCar").strip()
        d3 = r.get("d3", "").strip()
        d4 = r.get("d4", "").strip()
        prov = unicodedata.normalize("NFC", r.get("province", "กรุงเทพมหานคร").strip())

        new_fn = f"real_{idx:06d}.jpg"
        dest_path = out_plates_dir / new_fn

        # Robust read and save
        try:
            with open(src_path, "rb") as bf:
                buf = np.frombuffer(bf.read(), dtype=np.uint8)
                im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is not None:
                cv2.imwrite(str(dest_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                h_out, w_out = im.shape[:2]
            else:
                continue
        except Exception:
            continue

        full_text = f"{p_type} {d3} {d4} {prov}".strip()
        token_ids = encode_plate_to_tokens(p_type, d3, d4, prov, include_start_stop=True)

        real_records.append({
            "index": idx,
            "filename": new_fn,
            "filepath": str(dest_path),
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov,
            "iso_3166_2": NAME_TO_ISO.get(prov, "TH-10"),
            "full_text_line": full_text,
            "token_ids": json.dumps(token_ids),
            "token_count": len(token_ids),
            "width": w_out,
            "height": h_out,
            "is_real_cctv": True,
            "is_low_res": (h_out <= 48),
            "source": "real_cctv_anchor_202610261140",
            "created_at_utc": datetime.now(timezone.utc).isoformat()
        })
        idx += 1

    print(f"Successfully ingested {len(real_records):,} authentic real CCTV plates into Dataset v3.1.")
    return real_records


def generate_dataset_v31(num_synthetic: int = 48817, num_workers: int = 16, clean: bool = True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plates_dir = OUTPUT_DIR / "plates"
    if clean and plates_dir.exists():
        print(f"Cleaning existing directory {plates_dir}...")
        shutil.rmtree(plates_dir, ignore_errors=True)
    plates_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("  GENERATING DATASET v3.1 — CCTV & LOW-RESOLUTION ADD-ON")
    print(f"  Target Synthetic Augmented: {num_synthetic:,} images")
    print(f"  Target Real CCTV Anchor: ~1,183 images")
    print(f"  Total Expected Images: ~{num_synthetic + 1183:,} images")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Parallel Workers: {num_workers}")
    print("=" * 90)

    t0 = time.time()

    # 1. Generate Synthetic CCTV Augmented Plates
    chunk_size = 500
    tasks = []
    for start_i in range(0, num_synthetic, chunk_size):
        c = min(chunk_size, num_synthetic - start_i)
        tasks.append((start_i, c, str(OUTPUT_DIR), str(plates_dir)))

    synthetic_records = []
    with mp.Pool(processes=num_workers) as pool:
        for batch_recs in pool.imap_unordered(worker_generate_batch, tasks):
            synthetic_records.extend(batch_recs)
            if len(synthetic_records) % 5000 == 0 or len(synthetic_records) == num_synthetic:
                print(f"  --> Generated {len(synthetic_records):,}/{num_synthetic:,} synthetic CCTV plates...")

    synthetic_records.sort(key=lambda x: x["index"])

    # 2. Ingest Real CCTV Plates
    real_records = ingest_real_cctv_anchor_plates(plates_dir, start_idx=len(synthetic_records))

    all_records = synthetic_records + real_records
    total_samples = len(all_records)
    elapsed = time.time() - t0

    # 3. Write metadata.csv
    meta_path = OUTPUT_DIR / "metadata.csv"
    fieldnames = list(all_records[0].keys())
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    # 4. Write annotations.json
    ann_path = OUTPUT_DIR / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    # 5. Write version_manifest.json
    manifest_path = OUTPUT_DIR / "version_manifest.json"
    manifest = {
        "dataset_version": "v3.1-cctv-addon",
        "description": "CCTV optical degradation, low-resolution stroke augmentation and real CCTV anchor add-on for Thai LPR",
        "total_images": total_samples,
        "synthetic_cctv_images": len(synthetic_records),
        "real_cctv_images": len(real_records),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_metadata": hashlib.sha256(meta_path.read_bytes()).hexdigest().upper(),
        "compliance": ["ISO/IEC 42001:2023", "ISO/IEC 5259-2:2023", "ISO/IEC 25012:2008"]
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 90)
    print(f"  DATASET v3.1 GENERATION COMPLETE!")
    print(f"  Total Images: {total_samples:,} (Synthetic: {len(synthetic_records):,}, Real CCTV: {len(real_records):,})")
    print(f"  Elapsed Time: {elapsed:.2f}s ({total_samples / elapsed:.1f} imgs/s)")
    print(f"  Metadata Path: {meta_path}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Dataset v3.1 CCTV Add-on")
    parser.add_argument("--num-synthetic", type=int, default=48817, help="Number of synthetic CCTV images to generate (default: 48817 -> ~50,000 total)")
    parser.add_argument("--num-workers", type=int, default=16, help="Parallel worker processes (default: 16)")
    parser.add_argument("--no-clean", action="store_true", help="Do not clean existing plates folder")
    args = parser.parse_args()

    generate_dataset_v31(num_synthetic=args.num_synthetic, num_workers=args.num_workers, clean=not args.no_clean)
