#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roboflow Universe Thai Character Downloader & Merger
Downloads character detection datasets from Roboflow Universe,
extracts bounding boxes for characters, and merges them into dataset/characters/<class>/.

Usage:
  python tools/download_roboflow_dataset.py --api-key YOUR_KEY --project thai-license-plate-character-recognition-d679u --workspace sed698-zva75
"""

import os
import sys
import argparse
from pathlib import Path
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
CHARS_DIR = ROOT_DIR / "dataset" / "characters"


def download_and_merge_roboflow(api_key: str, workspace: str, project_name: str, version_num: int = 1):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[!] Installing roboflow package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow"])
        from roboflow import Roboflow

    print(f"[*] Authenticating with Roboflow Universe for {workspace}/{project_name}...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)

    download_dir = ROOT_DIR / "dataset" / "roboflow_downloads" / project_name
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading YOLOv8 annotations to {download_dir}...")
    dataset = version.download("yolov8", location=str(download_dir))

    # Read data.yaml for class names
    import yaml
    yaml_path = download_dir / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    class_names = config.get("names", [])
    print(f"[*] Loaded {len(class_names)} classes: {class_names}")

    merged_count = 0
    for split in ["train", "valid", "test"]:
        img_dir = download_dir / split / "images"
        lbl_dir = download_dir / split / "labels"
        if not img_dir.exists():
            continue

        for lbl_file in lbl_dir.glob("*.txt"):
            stem = lbl_file.stem
            img_file = None
            for ext in [".jpg", ".png", ".jpeg"]:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if img_file is None:
                continue

            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]

            with open(lbl_file, "r") as fp:
                for idx, line in enumerate(fp):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        x1 = max(0, int((xc - bw / 2) * w))
                        y1 = max(0, int((yc - bh / 2) * h))
                        x2 = min(w, int((xc + bw / 2) * w))
                        y2 = min(h, int((yc + bh / 2) * h))

                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 4:
                            continue

                        char_label = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
                        dest_folder = CHARS_DIR / char_label
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_folder / f"rf_{project_name}_{stem}_{idx}_{char_label}.jpg"
                        
                        success, buf = cv2.imencode(".jpg", crop)
                        if success:
                            buf.tofile(str(dest_path))
                            merged_count += 1

    print(f"[+] Successfully merged {merged_count:,} character crops from Roboflow into {CHARS_DIR}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY", ""))
    parser.add_argument("--workspace", type=str, default="sed698-zva75")
    parser.add_argument("--project", type=str, default="thai-license-plate-character-recognition-d679u")
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()

    if not args.api_key:
        print("[!] Warning: Roboflow API Key is required to download via API.")
        print("[!] Obtain your free key from https://app.roboflow.com and pass via --api-key YOUR_KEY")
    else:
        download_and_merge_roboflow(args.api_key, args.workspace, args.project, args.version)
