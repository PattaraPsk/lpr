#!/usr/bin/env python3
"""
Dataset Preparation & Unpacker for YOLO License Plate Detection Training.
Extracts YOLOv8/YOLOv11 zip datasets into clean train/val/test splits and generates data.yaml.
"""

import os
import sys
import zipfile
import shutil
import argparse
import yaml
from pathlib import Path


def extract_zip(zip_path: Path, extract_dir: Path):
    print(f"[INFO] Extracting {zip_path.name} -> {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[SUCCESS] Extracted to {extract_dir}")


def generate_data_yaml(target_dir: Path, class_names: list, output_yaml: Path):
    data_dict = {
        "path": str(target_dir.resolve()),
        "train": "images/train" if (target_dir / "images" / "train").exists() else "train/images",
        "val": "images/val" if (target_dir / "images" / "val").exists() else "valid/images",
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    test_dir = target_dir / "images" / "test"
    if not test_dir.exists():
        test_dir = target_dir / "test" / "images"
    if test_dir.exists():
        data_dict["test"] = str(test_dir.relative_to(target_dir))
        
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_dict, f, default_flow_style=False, sort_keys=False)
        
    print(f"[SUCCESS] Generated dataset YAML: {output_yaml}")


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO Plate Detection Dataset")
    parser.add_argument("--zip-file", type=str, default="", help="Path to specific zip archive")
    parser.add_argument("--output-dir", type=str, default="data/unpacked", help="Target output directory")
    parser.add_argument("--extract-all", action="store_true", help="Extract all zip files in datasets/")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    datasets_dir = base_dir / "datasets"
    target_dir = base_dir / args.output_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    if args.extract_all:
        zip_files = list(datasets_dir.glob("*.zip"))
        if not zip_files:
            print(f"[WARN] No zip files found in {datasets_dir}")
            return
        for zf in zip_files:
            folder_name = zf.stem.replace(" ", "_").replace(".", "_")
            out = target_dir / folder_name
            out.mkdir(parents=True, exist_ok=True)
            extract_zip(zf, out)
    elif args.zip_file:
        zf = Path(args.zip_file)
        if not zf.is_absolute():
            zf = base_dir / zf
        if zf.exists():
            folder_name = zf.stem.replace(" ", "_").replace(".", "_")
            out = target_dir / folder_name
            out.mkdir(parents=True, exist_ok=True)
            extract_zip(zf, out)
        else:
            print(f"[ERROR] Zip file {zf} does not exist.")
    else:
        print("[INFO] Available datasets in datasets/:")
        for z in datasets_dir.glob("*.zip"):
            print(f"  - {z.name} ({round(z.stat().st_size / (1024*1024), 2)} MB)")
        print("\nRun with --extract-all or --zip-file <path> to extract.")


if __name__ == "__main__":
    main()
