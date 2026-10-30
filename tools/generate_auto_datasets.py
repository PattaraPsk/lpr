#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Dataset Generator for Thai LPR Pipeline:
1. Auto-labeled Plate Dataset (manual_expert OCR + DLT Classification):
   - Cropped license plates extracted from uncropped scenes using vehicle_plate_yolov11n
   - Pre-cropped plates from curated dataset
   - Auto-labeled via manual_expert (d3, d4, province, plate_type)
   - Saved to: dataset/dataset-auto-label-a1341
   - Includes .lpr_session.json, annotations.json, metadata.csv

2. Front/Rear Detection + Vehicle Classification Dataset:
   - YOLO object detection dataset combining Front/Rear detector & YOLOv8s vehicle classifier
   - Multi-class bounding boxes:
       0: Front
       1: Rear
       2: LicensePlate
       3: Car
       4: Motorcycle
       5: Truck
       6: Bus
   - Saved to: dataset/dataset-vehicle-orientation
   - Includes data.yaml and ISO/IEC 5259-2:2023 quality audit report

Compliance:
- ISO 8601:2019 (Timestamps & Durations)
- ISO 3166-2:TH (Thai Province Codes)
- ISO/IEC 10646 / Unicode NFC (Character Normalization)
- ISO/IEC 25012:2008 (Data Quality & Integrity)
- ISO/IEC 5259-2:2023 (AI ML Quality Measures)
"""

import os
import sys
import re
import json
import csv
import time
import shutil
import random
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.DataManipulator.app.plate_reader import PlateReader, decode_tokens_to_plate
from tools.DataManipulator.app.provinces import normalize_thai_script, get_iso_code, match_province_value
from tools.DataManipulator.app.plate_types import match_plate_type, PLATE_TYPES_DATA
from tools.owod.core.quality_auditor import audit_dataset

# =============================================================================
# Constants & Configuration
# =============================================================================

RANDOM_SEED = 42
VAL_SPLIT = 0.15

# Output paths
AUTO_LABEL_DIR = ROOT_DIR / "dataset" / "labeled" / "dataset-auto-label-a1341"
AUTO_LABEL_PLATES_DIR = AUTO_LABEL_DIR / "plates"

VEHICLE_ORIENTATION_DIR = ROOT_DIR / "dataset" / "labeled" / "dataset-vehicle-orientation"
FR_IMG_TRAIN_DIR = VEHICLE_ORIENTATION_DIR / "images" / "train"
FR_IMG_VAL_DIR = VEHICLE_ORIENTATION_DIR / "images" / "val"
FR_LBL_TRAIN_DIR = VEHICLE_ORIENTATION_DIR / "labels" / "train"
FR_LBL_VAL_DIR = VEHICLE_ORIENTATION_DIR / "labels" / "val"

# Model paths
RF_MODEL_PATH = ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt"
if not RF_MODEL_PATH.exists():
    RF_MODEL_PATH = ROOT_DIR / "models" / "detector" / "vehicle_plate_detector.pt"
if not RF_MODEL_PATH.exists():
    RF_MODEL_PATH = ROOT_DIR / "vehicle_plate_detector.pt"

VEH_MODEL_PATH = ROOT_DIR / "models" / "detector" / "yolov8s.pt"
if not VEH_MODEL_PATH.exists():
    VEH_MODEL_PATH = ROOT_DIR / "yolov8s.pt"

READER_MODEL_PATH = ROOT_DIR / "models" / "reader" / "manual_expert" / "best_norm_ED.pth"
if not READER_MODEL_PATH.exists():
    READER_MODEL_PATH = ROOT_DIR / "best_norm_ED.pth"

# Input directories
UNCROPPED_DIR = ROOT_DIR / "archive" / "zero_shot_license_plate_detection" / "input_images"
CURATED_PLATES_DIR = ROOT_DIR / "dataset" / "verified" / "curated_lpr_dataset" / "plates"
if not CURATED_PLATES_DIR.exists():
    CURATED_PLATES_DIR = ROOT_DIR / "dataset" / "curated_lpr_dataset" / "plates"

# Multi-class detection names
CLASS_NAMES = [
    "Front",         # 0
    "Rear",          # 1
    "LicensePlate",  # 2
    "Car",           # 3
    "Motorcycle",    # 4
    "Truck",         # 5
    "Bus"            # 6
]

# Source model class mappings
# rf_model: 0->Front (0), 1->LicensePlate (2), 2->Rear (1)
RF_TO_TARGET = {
    0: 0,  # Front -> 0
    1: 2,  # LicensePlate -> 2
    2: 1   # Rear -> 1
}

# veh_model: 2->Car (3), 3->Motorcycle (4), 7->Truck (5), 5->Bus (6)
COCO_TO_TARGET = {
    2: 3,  # car -> Car (3)
    3: 4,  # motorcycle -> Motorcycle (4)
    7: 5,  # truck -> Truck (5)
    5: 6   # bus -> Bus (6)
}


# =============================================================================
# Helper Utilities
# =============================================================================

def safe_load_image(path: Path) -> Optional[Image.Image]:
    """Safely opens an image on Windows handling Thai / Unicode characters."""
    try:
        im = Image.open(path)
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGB")
        return im
    except Exception:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
        except Exception:
            pass
    return None


def safe_save_image(img: Image.Image, path: Path):
    """Safely writes PIL image to disk supporting Unicode paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=95)


def sanitize_filename_part(text: str) -> str:
    """Sanitizes text for filesystem usage while preserving Thai characters."""
    if not text:
        return "-"
    cleaned = re.sub(r'[\\/*?:"<>|]', "", text).strip()
    return cleaned if cleaned else "-"


def classify_dlt_plate_type(
    vehicle_type: Optional[str],
    d3: str,
    d4: str,
    aspect_ratio: float,
    hint_type: Optional[str] = None
) -> str:
    """
    Classifies plate into DLT categories:
    - WhiteBlackCar: Private passenger cars <= 7 seats
    - WhiteBlackMotor: Private motorcycles
    - WhiteGreen: Private pickup trucks / trucks
    - WhiteBlue: Vans / microbuses > 7 seats
    - YellowBlack: Commercial taxi
    """
    if hint_type:
        matched = match_plate_type(hint_type)
        if matched and matched not in ["+", "-"]:
            return matched

    # Check vehicle type association first
    if vehicle_type == "Motorcycle":
        return "WhiteBlackMotor"
    elif vehicle_type == "Truck":
        return "WhiteGreen"
    elif vehicle_type == "Bus":
        return "WhiteBlue"
    elif vehicle_type == "Car":
        if d3.startswith(("ท", "ม")) and len(d3) in [2, 3]:
            return "YellowBlack"
        return "WhiteBlackCar"

    # Heuristics based on plate morphology & text
    if aspect_ratio < 1.45:
        # Motorcycle plates in Thailand are square-like
        return "WhiteBlackMotor"

    if d3.startswith(("ท", "ม")) and len(d3) in [2, 3]:
        return "YellowBlack"

    return "WhiteBlackCar"


# =============================================================================
# Main Pipeline Implementation
# =============================================================================

def setup_workspace():
    """Initializes output directory structures."""
    AUTO_LABEL_PLATES_DIR.mkdir(parents=True, exist_ok=True)
    for d in [FR_IMG_TRAIN_DIR, FR_IMG_VAL_DIR, FR_LBL_TRAIN_DIR, FR_LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[INIT] Output directory dataset-auto-label-a1341 ready: {AUTO_LABEL_DIR}")
    print(f"[INIT] Output directory dataset-vehicle-orientation ready: {VEHICLE_ORIENTATION_DIR}")


def run_pipeline():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    setup_workspace()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Models
    print(f"[INFO] Loading Front/Rear detector from {RF_MODEL_PATH.name}...")
    rf_model = YOLO(str(RF_MODEL_PATH)).to(device)

    print(f"[INFO] Loading Vehicle classifier from {VEH_MODEL_PATH.name}...")
    veh_model = YOLO(str(VEH_MODEL_PATH)).to(device)

    print(f"[INFO] Initializing Thai OCR Reader manual_expert...")
    ocr_reader = PlateReader.get_instance(str(READER_MODEL_PATH))

    # Containers for collection
    fr_train_count = 0
    fr_val_count = 0
    fr_class_counts = {name: 0 for name in CLASS_NAMES}

    plates_to_ocr = []  # list of dicts: {'crop': PIL.Image, 'meta': dict}
    session_items: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    # =========================================================================
    # Step 1: Process Uncropped Scenes (Front/Rear + Vehicles + Extract Plates)
    # =========================================================================
    uncropped_files = []
    if UNCROPPED_DIR.exists():
        uncropped_files = list(UNCROPPED_DIR.glob("*.jpg")) + list(UNCROPPED_DIR.glob("*.png"))
    print(f"\n[STEP 1] Scanning {len(uncropped_files)} uncropped images from {UNCROPPED_DIR.name}...")

    BATCH_SIZE = 32
    extracted_plates_count = 0

    for i in tqdm(range(0, len(uncropped_files), BATCH_SIZE), desc="Detecting Scenes"):
        batch_paths = uncropped_files[i:i + BATCH_SIZE]
        batch_images = []
        valid_paths = []

        for p in batch_paths:
            img = safe_load_image(p)
            if img is not None:
                batch_images.append(img)
                valid_paths.append(p)

        if not batch_images:
            continue

        # Batched inference
        rf_results = rf_model(batch_images, batch=len(batch_images), conf=0.25, verbose=False, device=device)
        veh_results = veh_model(batch_images, batch=len(batch_images), classes=[2, 3, 5, 7], conf=0.25, verbose=False, device=device)

        for img_idx, (img, path, rf_res, veh_res) in enumerate(zip(batch_images, valid_paths, rf_results, veh_results)):
            w, h = img.size
            all_bboxes = []  # list of (target_cls, [x1, y1, x2, y2], conf)
            extracted_plate_boxes = []
            vehicle_boxes = []

            # Front/Rear & LicensePlate detections
            for b in rf_res.boxes:
                src_cls = int(b.cls[0])
                if src_cls in RF_TO_TARGET:
                    target_cls = RF_TO_TARGET[src_cls]
                    conf = float(b.conf[0])
                    xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    all_bboxes.append((target_cls, xyxy, conf))
                    if target_cls == 2:  # LicensePlate
                        extracted_plate_boxes.append((xyxy, conf))

            # Vehicle detections
            for b in veh_res.boxes:
                src_cls = int(b.cls[0])
                if src_cls in COCO_TO_TARGET:
                    target_cls = COCO_TO_TARGET[src_cls]
                    conf = float(b.conf[0])
                    xyxy = [float(x) for x in b.xyxy[0].tolist()]
                    all_bboxes.append((target_cls, xyxy, conf))
                    vehicle_boxes.append((target_cls, xyxy, conf))

            # If any detections, save to dataset-vehicle-orientation
            if all_bboxes:
                is_val = random.random() < VAL_SPLIT
                split_str = "val" if is_val else "train"
                img_dest_dir = FR_IMG_VAL_DIR if is_val else FR_IMG_TRAIN_DIR
                lbl_dest_dir = FR_LBL_VAL_DIR if is_val else FR_LBL_TRAIN_DIR

                sample_stem = f"scene_{path.stem}"
                img_dest_path = img_dest_dir / f"{sample_stem}.jpg"
                lbl_dest_path = lbl_dest_dir / f"{sample_stem}.txt"

                # Save image
                safe_save_image(img, img_dest_path)

                # Format YOLO txt annotations
                lbl_lines = []
                for tgt_cls, (x1, y1, x2, y2), conf in all_bboxes:
                    cx = ((x1 + x2) / 2.0) / w
                    cy = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.001, min(1.0, bw))
                    bh = max(0.001, min(1.0, bh))
                    lbl_lines.append(f"{tgt_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    fr_class_counts[CLASS_NAMES[tgt_cls]] += 1

                with open(lbl_dest_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lbl_lines))

                if is_val:
                    fr_val_count += 1
                else:
                    fr_train_count += 1

            # Extract license plate crops for OCR
            for p_idx, (p_box, p_conf) in enumerate(extracted_plate_boxes):
                px1, py1, px2, py2 = p_box
                pw = px2 - px1
                ph = py2 - py1

                # Filter tiny or degenerate boxes
                if pw < 14 or ph < 8 or pw > w * 0.95 or ph > h * 0.95:
                    continue
                aspect = pw / max(ph, 1.0)
                if aspect < 0.6 or aspect > 5.5:
                    continue

                # Add 4% margin padding
                margin_x = pw * 0.04
                margin_y = ph * 0.04
                cx1 = max(0, int(px1 - margin_x))
                cy1 = max(0, int(py1 - margin_y))
                cx2 = min(w, int(px2 + margin_x))
                cy2 = min(h, int(py2 + margin_y))

                crop_img = img.crop((cx1, cy1, cx2, cy2))
                if crop_img.width < 10 or crop_img.height < 6:
                    continue

                # Find associated vehicle type
                matched_v_name = None
                p_cx = (px1 + px2) / 2.0
                p_cy = (py1 + py2) / 2.0
                for v_cls, (vx1, vy1, vx2, vy2), v_conf in vehicle_boxes:
                    if (vx1 <= p_cx <= vx2) and (vy1 <= p_cy <= vy2):
                        matched_v_name = CLASS_NAMES[v_cls]
                        break

                plate_type = classify_dlt_plate_type(
                    vehicle_type=matched_v_name,
                    d3="",
                    d4="",
                    aspect_ratio=aspect
                )

                extracted_plates_count += 1
                unique_name = f"ext_{path.stem}_p{p_idx:02d}"
                plates_to_ocr.append({
                    "crop": crop_img,
                    "source": "extracted",
                    "orig_file": path.name,
                    "unique_name": unique_name,
                    "vehicle_type": matched_v_name or "Car",
                    "plate_type": plate_type,
                    "gt_d3": "",
                    "gt_d4": "",
                    "gt_province": ""
                })

    print(f"[STEP 1 DONE] Generated {fr_train_count} train + {fr_val_count} val scene annotations.")
    print(f"             Extracted {extracted_plates_count} cropped plates for OCR auto-labeling.")

    # =========================================================================
    # Step 2: Queue Pre-Cropped Plates (dataset/curated_lpr_dataset/plates)
    # =========================================================================
    curated_files = []
    if CURATED_PLATES_DIR.exists():
        curated_files = list(CURATED_PLATES_DIR.glob("*.jpg")) + list(CURATED_PLATES_DIR.glob("*.png"))
    print(f"\n[STEP 2] Queuing {len(curated_files)} pre-cropped plates from {CURATED_PLATES_DIR.name}...")

    for p in curated_files:
        img = safe_load_image(p)
        if img is None:
            continue

        parts = p.stem.split("_")
        name_id = parts[0] if len(parts) > 0 else p.stem
        v_token = parts[1] if len(parts) > 1 else "car"
        gt_d3 = parts[2] if len(parts) > 2 else ""
        gt_d4 = parts[3] if len(parts) > 3 else ""
        gt_prov = parts[4] if len(parts) > 4 else ""

        plate_type = match_plate_type(v_token) or "WhiteBlackCar"
        aspect = img.width / max(img.height, 1)

        plates_to_ocr.append({
            "crop": img,
            "source": "curated",
            "orig_file": p.name,
            "unique_name": f"cur_{name_id}",
            "vehicle_type": v_token,
            "plate_type": plate_type,
            "gt_d3": gt_d3,
            "gt_d4": gt_d4,
            "gt_province": gt_prov
        })

    # =========================================================================
    # Step 3: Run High-Throughput Batch OCR with manual_expert
    # =========================================================================
    print(f"\n[STEP 3] Running Batch OCR with manual_expert on {len(plates_to_ocr)} total plate crops...")
    OCR_BATCH_SIZE = 64
    processed_plates_count = 0

    for i in tqdm(range(0, len(plates_to_ocr), OCR_BATCH_SIZE), desc="OCR manual_expert"):
        batch = plates_to_ocr[i:i + OCR_BATCH_SIZE]
        tensors = []
        for item in batch:
            t = ocr_reader.preprocess_image(item["crop"])
            tensors.append(t)

        batch_tensor = torch.cat(tensors, dim=0).to(ocr_reader.device)
        with torch.no_grad():
            preds = ocr_reader.model(batch_tensor)
            probs = F.softmax(preds, dim=-1)
            max_probs, indices = probs.max(dim=-1)

        # Decode each item in batch
        for b_idx, item in enumerate(batch):
            tokens = indices[b_idx].tolist()
            conf = float(max_probs[b_idx].mean().item())

            pred_d3, pred_d4, pred_prov, full_plate = decode_tokens_to_plate(tokens)

            # Determine final values
            if item["source"] == "curated":
                final_d3 = pred_d3 if pred_d3 else item["gt_d3"]
                final_d4 = pred_d4 if pred_d4 else item["gt_d4"]
                final_prov = pred_prov if pred_prov else item["gt_province"]
                plate_type = item["plate_type"]
            else:
                final_d3 = pred_d3
                final_d4 = pred_d4
                final_prov = pred_prov
                aspect = item["crop"].width / max(item["crop"].height, 1)
                plate_type = classify_dlt_plate_type(
                    vehicle_type=item["vehicle_type"],
                    d3=final_d3,
                    d4=final_d4,
                    aspect_ratio=aspect,
                    hint_type=item["plate_type"]
                )

            # Canonical Thai NFC normalization & ISO 3166-2:TH code
            final_d3 = normalize_thai_script(final_d3)
            final_d4 = normalize_thai_script(final_d4)
            final_prov = match_province_value(final_prov) if final_prov else ""
            iso_code = get_iso_code(final_prov) or ""

            # Format canonical filename
            safe_d3 = sanitize_filename_part(final_d3)
            safe_d4 = sanitize_filename_part(final_d4)
            safe_prov = sanitize_filename_part(final_prov)
            out_filename = f"{item['unique_name']}_{plate_type}_{safe_d3}_{safe_d4}_{safe_prov}.jpg"
            out_filepath = AUTO_LABEL_PLATES_DIR / out_filename

            # Save plate image
            safe_save_image(item["crop"], out_filepath)

            # Record session entry
            session_items[out_filename] = {
                "filename": out_filename,
                "path": str(out_filepath.resolve()),
                "rel_path": out_filename,
                "subfolder": "",
                "name": item["unique_name"],
                "plate_type": plate_type,
                "d3": final_d3,
                "d4": final_d4,
                "province": final_prov,
                "iso_3166_2": iso_code,
                "status": "approved" if (final_d3 and final_d4) else "pending",
                "ai_d3": pred_d3,
                "ai_d4": pred_d4,
                "ai_province": pred_prov,
                "ai_confidence": round(conf, 4)
            }

            # Record metadata CSV entry
            csv_rows.append({
                "filename": out_filename,
                "source_type": item["source"],
                "original_file": item["orig_file"],
                "plate_type": plate_type,
                "d3": final_d3,
                "d4": final_d4,
                "province": final_prov,
                "iso_3166_2": iso_code,
                "ai_confidence": round(conf, 4),
                "full_plate": f"{final_d3} {final_d4} {final_prov}".strip(),
                "width": item["crop"].width,
                "height": item["crop"].height
            })
            processed_plates_count += 1

    # Save .lpr_session.json (DataManipulator compatible)
    session_data = {
        "version": "1.3",
        "standards": [
            "ISO 8601:2019",
            "ISO 3166-2:TH",
            "ISO/IEC 10646",
            "ISO/IEC 25012:2008"
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "folder": str(AUTO_LABEL_PLATES_DIR.resolve()),
        "current_index": 0,
        "items": session_items
    }
    session_path = AUTO_LABEL_DIR / ".lpr_session.json"
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    # Save annotations.json
    ann_path = AUTO_LABEL_DIR / "annotations.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    # Save metadata.csv
    csv_path = AUTO_LABEL_DIR / "metadata.csv"
    if csv_rows:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"[STEP 3 DONE] Auto-labeled {processed_plates_count} plates saved to: {AUTO_LABEL_DIR}")

    # =========================================================================
    # Step 4: Finalize dataset-vehicle-orientation (YAML & Quality Audit)
    # =========================================================================
    print(f"\n[STEP 4] Finalizing dataset-vehicle-orientation...")
    yaml_data = {
        "path": str(VEHICLE_ORIENTATION_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    yaml_path = VEHICLE_ORIENTATION_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(yaml_data, f, sort_keys=False)
    print(f"  -> data.yaml generated: {yaml_path}")

    # Run ISO/IEC 5259-2 Quality Audit
    audit_report = audit_dataset(VEHICLE_ORIENTATION_DIR, save_report=True)
    report_path = VEHICLE_ORIENTATION_DIR / "data_quality_report.json"
    print(f"  -> ISO 5259 Quality Audit Report generated: {report_path}")

    # Summary Output
    print("\n" + "=" * 70)
    print("                    DATASET GENERATION SUMMARY")
    print("=" * 70)
    print(f"1. OCR Auto-labeled Dataset (manual_expert):")
    print(f"   - Target Folder : {AUTO_LABEL_DIR}")
    print(f"   - Total Plates  : {processed_plates_count}")
    print(f"   - Session File  : {session_path.name} (DataManipulator compatible)")
    print(f"   - Metadata CSV  : {csv_path.name}")
    print(f"2. Front/Rear + Vehicle Classification Dataset:")
    print(f"   - Target Folder : {VEHICLE_ORIENTATION_DIR}")
    print(f"   - Train Images  : {fr_train_count}")
    print(f"   - Val Images    : {fr_val_count}")
    print(f"   - Class Counts  : {fr_class_counts}")
    print(f"   - Quality Grade : {audit_report.get('quality_measures', {}).get('quality_grade', 'N/A')}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
