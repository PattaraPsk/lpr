#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roboflow Universe Downloader & Merger for Front/Rear + Vehicle + Plate Detector
Pulls targeted motorcycle and license plate datasets from Roboflow Universe,
re-maps class labels to the canonical 7-class schema, and integrates them into
dataset/labeled/dataset-vehicle-orientation.

Target Schema:
  0: Front
  1: Rear
  2: LicensePlate (all vehicles + motorcycles)
  3: Car
  4: Motorcycle
  5: Truck
  6: Bus
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import Counter
import yaml
from PIL import Image

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

TARGET_DATASET_DIR = ROOT_DIR / "dataset" / "labeled" / "dataset-vehicle-orientation"
TARGET_CLASSES = {
    0: "Front",
    1: "Rear",
    2: "LicensePlate",
    3: "Car",
    4: "Motorcycle",
    5: "Truck",
    6: "Bus"
}

# Heuristic mapping from arbitrary dataset class names to target 7 classes
CLASS_MAPPING_RULES = {
    # 0: Front
    "front": 0,
    "front_vehicle": 0,
    "front-vehicle": 0,
    "front_view": 0,
    "front-view": 0,
    "vehicle_front": 0,

    # 1: Rear
    "rear": 1,
    "rear_vehicle": 1,
    "rear-vehicle": 1,
    "rear_view": 1,
    "rear-view": 1,
    "vehicle_rear": 1,
    "back": 1,

    # 2: LicensePlate (including motorcycle plates)
    "licenseplate": 2,
    "license_plate": 2,
    "license-plate": 2,
    "plate": 2,
    "plates": 2,
    "numberplate": 2,
    "number_plate": 2,
    "number-plate": 2,
    "motorcycle_plate": 2,
    "motorcycle-plate": 2,
    "motorcycle_license_plate": 2,
    "motorcycle-license-plate": 2,
    "bike_plate": 2,
    "bike-plate": 2,
    "rear_plate": 2,
    "rear-plate": 2,
    "front_plate": 2,
    "front-plate": 2,
    "lp": 2,

    # 3: Car
    "car": 3,
    "cars": 3,
    "automobile": 3,
    "sedan": 3,
    "suv": 3,
    "van": 3,
    "pickup": 3,
    "auto": 3,

    # 4: Motorcycle
    "motorcycle": 4,
    "motorcycles": 4,
    "motorbike": 4,
    "motorbikes": 4,
    "bike": 4,
    "bikes": 4,
    "two-wheeler": 4,
    "two_wheeler": 4,
    "scooter": 4,

    # 5: Truck
    "truck": 5,
    "trucks": 5,
    "lorry": 5,

    # 6: Bus
    "bus": 6,
    "buses": 6
}


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def map_source_classes(source_names: list) -> dict:
    """Creates a dictionary mapping source class index -> target class index."""
    src_to_tgt = {}
    for idx, name in enumerate(source_names):
        norm = normalize_class_name(str(name))
        if norm in CLASS_MAPPING_RULES:
            src_to_tgt[idx] = CLASS_MAPPING_RULES[norm]
        else:
            # Substring matching fallback
            matched = False
            for rule_key, tgt_idx in CLASS_MAPPING_RULES.items():
                if rule_key in norm:
                    src_to_tgt[idx] = tgt_idx
                    matched = True
                    break
            if not matched:
                print(f"  [WARN] Unrecognized source class '{name}' (idx {idx}); skipping this class.")
    return src_to_tgt


def download_and_merge_project(
    api_key: str,
    workspace: str,
    project_name: str,
    version_num: int = 1,
    target_dir: Path = TARGET_DATASET_DIR
):
    from roboflow import Roboflow

    print(f"\n============================================================")
    print(f" Downloading from Roboflow Universe: {workspace}/{project_name} (v{version_num})")
    print(f"============================================================")

    rf = Roboflow(api_key=api_key)
    try:
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
    except Exception as e:
        print(f"[ERROR] Failed to access Roboflow project {workspace}/{project_name}: {e}")
        return 0

    download_cache = ROOT_DIR / "dataset" / "roboflow_downloads" / f"{workspace}_{project_name}_v{version_num}"
    download_cache.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading YOLOv8 dataset format into cache: {download_cache.name}...")
    try:
        version.download("yolov8", location=str(download_cache))
    except Exception as e:
        print(f"[ERROR] Failed downloading version: {e}")
        return 0

    yaml_file = download_cache / "data.yaml"
    if not yaml_file.exists():
        print(f"[ERROR] data.yaml not found in downloaded dataset: {yaml_file}")
        return 0

    with open(yaml_file, "r", encoding="utf-8") as f:
        src_yaml = yaml.safe_load(f)

    src_names = src_yaml.get("names", [])
    if isinstance(src_names, dict):
        src_names = [src_names[k] for k in sorted(src_names.keys())]

    print(f"[*] Source classes: {src_names}")
    mapping = map_source_classes(src_names)
    print(f"[*] Resolved class mapping: { {src_names[k]: TARGET_CLASSES[v] for k, v in mapping.items()} }")

    if not mapping:
        print(f"[WARN] No compatible classes found in project {project_name}.")
        return 0

    # Ensure target directories exist
    train_img_tgt = target_dir / "images" / "train"
    train_lbl_tgt = target_dir / "labels" / "train"
    val_img_tgt = target_dir / "images" / "val"
    val_lbl_tgt = target_dir / "labels" / "val"
    for d in [train_img_tgt, train_lbl_tgt, val_img_tgt, val_lbl_tgt]:
        d.mkdir(parents=True, exist_ok=True)

    imported_images = 0
    imported_boxes = Counter()

    for split in ["train", "valid", "val", "test"]:
        src_split_img = download_cache / split / "images"
        src_split_lbl = download_cache / split / "labels"
        if not src_split_img.exists():
            continue

        # Map 'valid' and 'test' into 'val', 'train' into 'train'
        dest_split = "val" if split in ["valid", "val", "test"] else "train"
        dest_img_dir = val_img_tgt if dest_split == "val" else train_img_tgt
        dest_lbl_dir = val_lbl_tgt if dest_split == "val" else train_lbl_tgt

        img_files = list(src_split_img.glob("*.*"))
        print(f"[*] Processing split '{split}' ({len(img_files)} images) -> targeting '{dest_split}'...")

        for img_p in img_files:
            if img_p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            lbl_p = src_split_lbl / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue

            # Read and filter annotations
            valid_lines = []
            with open(lbl_p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        src_cls = int(parts[0])
                        if src_cls in mapping:
                            tgt_cls = mapping[src_cls]
                            valid_lines.append(f"{tgt_cls} {' '.join(parts[1:])}\n")
                            imported_boxes[tgt_cls] += 1

            if not valid_lines:
                continue

            # Safe copy with unique prefix
            prefix = f"rf_{project_name[:12]}_{img_p.stem}"
            dest_img = dest_img_dir / f"{prefix}{img_p.suffix.lower()}"
            dest_lbl = dest_lbl_dir / f"{prefix}.txt"

            shutil.copy2(str(img_p), str(dest_img))
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.writelines(valid_lines)

            imported_images += 1

    print(f"\n[+] Successfully imported {imported_images:,} images from {project_name}!")
    print(f"[+] Imported boxes breakdown: { {TARGET_CLASSES[k]: v for k, v in sorted(imported_boxes.items())} }")
    return imported_images


def main():
    parser = argparse.ArgumentParser(description="Download and integrate Roboflow datasets into detector")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--projects", nargs="+", default=[
        "sed698-zva75/thai-motorcycle-license-plate/1",
        "zusture/motorcycle-license-plate-thai/1"
    ], help="List of workspace/project/version strings")
    args = parser.parse_args()

    api_key = args.api_key.strip()
    if not api_key:
        print("[!] ERROR: Roboflow API Key is required.")
        print("[!] Pass via --api-key <YOUR_KEY> or set ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)

    total_merged = 0
    for proj_spec in args.projects:
        parts = proj_spec.strip().split("/")
        if len(parts) == 3:
            ws, proj, ver = parts[0], parts[1], int(parts[2])
        elif len(parts) == 2:
            ws, proj, ver = parts[0], parts[1], 1
        else:
            print(f"[!] Invalid project format: {proj_spec}. Expected workspace/project[/version]")
            continue

        n = download_and_merge_project(api_key, ws, proj, ver)
        total_merged += n

    print(f"\n[DONE] Finished merging {total_merged:,} images across all Roboflow projects.")


if __name__ == "__main__":
    main()
