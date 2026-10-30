#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Augmented Dataset Integration & Balancer for Thai LPR Detector:
Integrates downloaded Roboflow datasets (Front, Motorcycle, Truck, Plates)
into dataset/labeled/dataset-vehicle-orientation.

Canonical 7-Class Schema:
  0: Front
  1: Rear
  2: LicensePlate
  3: Car
  4: Motorcycle
  5: Truck
  6: Bus
"""

import os
import sys
import shutil
import random
from pathlib import Path
from collections import Counter
import yaml
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = ROOT_DIR / "dataset" / "labeled" / "dataset-vehicle-orientation"
TRAIN_IMG = TARGET_DIR / "images" / "train"
TRAIN_LBL = TARGET_DIR / "labels" / "train"
VAL_IMG = TARGET_DIR / "images" / "val"
VAL_LBL = TARGET_DIR / "labels" / "val"

RF_DIR = ROOT_DIR / "dataset" / "roboflow_downloads"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def merge_psk10230():
    """Merges Front and LicensePlate annotations from user's psk10230_v1."""
    src = RF_DIR / "psk10230_v1"
    if not src.exists():
        print("[!] psk10230_v1 not found, skipping.")
        return 0

    print("\n[*] Merging psk10230_v1 (Front & LicensePlate)...")
    added = 0
    # Map: 0->Front(0), 1->LicensePlate(2)
    cls_map = {0: 0, 1: 2}

    for split in ["train", "valid", "test"]:
        s_img = src / split / "images"
        s_lbl = src / split / "labels"
        if not s_img.exists():
            continue

        dest_split = "val" if split in ["valid", "test"] else "train"
        d_img_dir = VAL_IMG if dest_split == "val" else TRAIN_IMG
        d_lbl_dir = VAL_LBL if dest_split == "val" else TRAIN_LBL

        for img_p in s_img.glob("*.*"):
            if img_p.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue
            lbl_p = s_lbl / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            valid_lines = []
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 5:
                        src_c = int(p[0])
                        if src_c in cls_map:
                            valid_lines.append(f"{cls_map[src_c]} {' '.join(p[1:])}\n")

            if not valid_lines:
                continue

            prefix = f"rf_psk_{img_p.stem}"
            dest_img = d_img_dir / f"{prefix}{img_p.suffix.lower()}"
            dest_lbl = d_lbl_dir / f"{prefix}.txt"

            shutil.copy2(str(img_p), str(dest_img))
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)
            added += 1

    print(f"[+] psk10230: Added {added:,} images.")
    return added


def merge_cleverest_trucks(max_samples: int = 1500):
    """Merges Trucks and Truck License Plates from cleverest_v1."""
    src = RF_DIR / "cleverest_v1"
    if not src.exists():
        print("[!] cleverest_v1 not found, skipping.")
        return 0

    print(f"\n[*] Merging cleverest_v1 (Truck & Plate, max {max_samples})...")
    # cleverest: 0->license-plate(2), 1->truck(5)
    cls_map = {0: 2, 1: 5}
    added = 0

    img_files = list((src / "train" / "images").glob("*.*"))
    random.shuffle(img_files)

    for img_p in img_files[:max_samples]:
        if img_p.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        lbl_p = src / "train" / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        valid_lines = []
        with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    src_c = int(p[0])
                    if src_c in cls_map:
                        valid_lines.append(f"{cls_map[src_c]} {' '.join(p[1:])}\n")

        if not valid_lines:
            continue

        # Split 85% train, 15% val
        dest_split = "val" if random.random() < 0.15 else "train"
        d_img_dir = VAL_IMG if dest_split == "val" else TRAIN_IMG
        d_lbl_dir = VAL_LBL if dest_split == "val" else TRAIN_LBL

        prefix = f"rf_trk_{img_p.stem}"
        dest_img = d_img_dir / f"{prefix}{img_p.suffix.lower()}"
        dest_lbl = d_lbl_dir / f"{prefix}.txt"

        shutil.copy2(str(img_p), str(dest_img))
        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.writelines(valid_lines)
        added += 1

    print(f"[+] cleverest: Added {added:,} truck & plate images.")
    return added


def merge_motorcycle_datasets(device: str = "cuda:0"):
    """
    Merges motorcycle datasets:
    - zeroexperiments_v2
    - zusture_v5
    - lpr_v23
    Uses YOLOv8s to pseudo-label the motorcycle vehicle bounding box (class 4)
    and combines with ground-truth license plate (class 2) + Rear orientation (class 1).
    """
    coco_model_path = ROOT_DIR / "models" / "detector" / "yolov8s.pt"
    if not coco_model_path.exists():
        coco_model_path = ROOT_DIR / "yolov8s.pt"
    
    print(f"\n[*] Loading COCO detector {coco_model_path.name} for motorcycle body detection...")
    coco = YOLO(str(coco_model_path)).to(device)

    sources = [
        ("rf_zero", RF_DIR / "zeroexperiments_v2", {0: 2}), # 0: license-plate -> 2
        ("rf_zus", RF_DIR / "zusture_v5", {1: 2}),          # 1: license-plate -> 2
        ("rf_lpr", RF_DIR / "lpr_v23", {0: 2})              # 0: license-plate -> 2
    ]

    total_added = 0

    for prefix, src_dir, plate_cls_map in sources:
        if not src_dir.exists():
            continue

        print(f"[*] Processing motorcycle dataset {src_dir.name}...")
        img_candidates = []
        for split in ["train", "valid", "test"]:
            s_img = src_dir / split / "images"
            s_lbl = src_dir / split / "labels"
            if not s_img.exists():
                continue
            for img_p in s_img.glob("*.*"):
                if img_p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    lbl_p = s_lbl / f"{img_p.stem}.txt"
                    if lbl_p.exists():
                        img_candidates.append((img_p, lbl_p, split))

        print(f"  Found {len(img_candidates)} motorcycle candidate images.")
        
        # Batch inference with COCO for motorcycle bodies
        BATCH = 32
        for i in tqdm(range(0, len(img_candidates), BATCH), desc=f"  Mining {prefix}"):
            batch = img_candidates[i:i + BATCH]
            batch_imgs = [cv2.imread(str(item[0])) for item in batch]
            valid_batch_indices = [idx for idx, im in enumerate(batch_imgs) if im is not None]
            
            if not valid_batch_indices:
                continue

            sub_imgs = [batch_imgs[idx] for idx in valid_batch_indices]
            sub_items = [batch[idx] for idx in valid_batch_indices]

            # Detect motorcycles (COCO class 3)
            det_results = coco(sub_imgs, classes=[3], conf=0.30, verbose=False, device=device)

            for (img_p, lbl_p, src_split), im_mat, det_res in zip(sub_items, sub_imgs, det_results):
                h, w = im_mat.shape[:2]
                annotations = []

                # 1. Ground truth license plate
                with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            s_cls = int(parts[0])
                            if s_cls in plate_cls_map:
                                annotations.append(f"2 {' '.join(parts[1:5])}\n")

                if not annotations:
                    continue

                # 2. Add detected motorcycle bounding boxes (Class 4)
                has_motorcycle = False
                for b in det_res.boxes:
                    bx = b.xywhn[0].tolist() # [xc, yc, w, h] normalized
                    annotations.append(f"4 {bx[0]:.6f} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f}\n")
                    has_motorcycle = True

                # If COCO missed motorcycle, add whole scene / motorcycle proxy if plate is centered
                if not has_motorcycle:
                    annotations.append(f"4 0.500000 0.500000 0.850000 0.850000\n")

                # 3. Add Rear orientation (Class 1) - Thai motorcycles strictly only have rear plates!
                annotations.append(f"1 0.500000 0.500000 0.900000 0.900000\n")

                dest_split = "val" if src_split in ["valid", "test"] else "train"
                d_img_dir = VAL_IMG if dest_split == "val" else TRAIN_IMG
                d_lbl_dir = VAL_LBL if dest_split == "val" else TRAIN_LBL

                dest_name = f"{prefix}_{img_p.stem}"
                dest_img = d_img_dir / f"{dest_name}{img_p.suffix.lower()}"
                dest_lbl = d_lbl_dir / f"{dest_name}.txt"

                cv2.imwrite(str(dest_img), im_mat)
                with open(dest_lbl, "w", encoding="utf-8") as f:
                    f.writelines(annotations)
                total_added += 1

    print(f"[+] Motorcycle datasets: Added {total_added:,} images with motorcycle bodies & plates!")
    return total_added


def audit_final_dataset():
    """Computes final class distributions and balance statistics."""
    print("\n" + "=" * 60)
    print(" FINAL AUDIT OF AUGMENTED DATASET")
    print("=" * 60)

    class_names = {0: "Front", 1: "Rear", 2: "LicensePlate", 3: "Car", 4: "Motorcycle", 5: "Truck", 6: "Bus"}
    overall_counts = Counter()

    for split in ["train", "val"]:
        img_dir = TARGET_DIR / "images" / split
        lbl_dir = TARGET_DIR / "labels" / split
        imgs = list(img_dir.glob("*.*"))
        lbls = list(lbl_dir.glob("*.txt"))
        split_counts = Counter()
        for lf in lbls:
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    p = line.strip().split()
                    if p:
                        split_counts[int(p[0])] += 1
                        overall_counts[int(p[0])] += 1

        print(f"Split '{split}': {len(imgs):,} images, {len(lbls):,} labels")
        print(f"  Class counts: { {class_names.get(k, k): v for k, v in sorted(split_counts.items())} }")

    print("\n[SUMMARY] Total annotations across entire dataset:")
    for k in sorted(overall_counts.keys()):
        print(f"  {class_names.get(k, k)} (Class {k}): {overall_counts[k]:,} instances")


if __name__ == "__main__":
    print("=" * 65)
    print(" ROBOTFLOW AUGMENTATION & MERGE PIPELINE")
    print("=" * 65)

    dev = "cuda:0"
    merge_psk10230()
    merge_cleverest_trucks(max_samples=1500)
    merge_motorcycle_datasets(device=dev)
    audit_final_dataset()
