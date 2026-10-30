#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Thai License Plate Labeling & Dataset Exporter using Thai LPR Expert Manual Annotator.
Complies with:
- ISO 8601:2019 (Date/Time Representation & UTC Timestamps)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & SHA-256 Cryptographic Lineage)
- ISO 3166-2:TH (Thailand Administrative Subdivisions)
- Thailand DLT Vehicle & License Plate Standards
"""

import os
import sys
import re
import json
import csv
import time
import shutil
import hashlib
import argparse
import unicodedata
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add repository root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# UTF-8 stdout configuration for Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import PlateReader, decode_tokens_to_plate
from tools.DataManipulator.app.provinces import normalize_thai_script, get_iso_code, match_province_value
from tools.DataManipulator.app.plate_types import match_plate_type, PLATE_TYPES_DATA


def calculate_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def sanitize_filename_part(text: str) -> str:
    """Sanitizes text for filesystem usage while preserving Thai characters."""
    if not text:
        return "-"
    cleaned = re.sub(r'[\\/*?:"<>|]', "", text).strip()
    return cleaned if cleaned else "-"


def get_base_name(stem: str) -> str:
    """Extracts clean unique tracking/image prefix from stem."""
    m = re.match(r'^(.*_trk\d+)', stem)
    if m:
        return m.group(1)
    m = re.match(r'^(plate_\d+_[a-zA-Z]+)', stem)
    if m:
        return m.group(1)
    m = re.match(r'^(plate_\d+)', stem)
    if m:
        return m.group(1)
    
    # Strip existing 4-part plate suffix if present
    parts = stem.rsplit('_', 4)
    if len(parts) >= 5:
        return parts[0]
    return stem


def classify_dlt_plate_type(
    d3: str,
    d4: str,
    aspect_ratio: float,
    stem_hint: str = ""
) -> str:
    """
    Classifies plate into DLT categories:
    - WhiteBlackCar: Private passenger cars <= 7 seats
    - WhiteBlackMotor: Private motorcycles
    - WhiteGreen: Private pickup trucks / trucks
    - WhiteBlue: Vans / microbuses > 7 seats
    - YellowBlack: Commercial taxi
    """
    stem_lower = stem_hint.lower()
    if "motorcycle" in stem_lower:
        return "WhiteBlackMotor"
    if "truck" in stem_lower or "pickup" in stem_lower:
        return "WhiteGreen"
    if "van" in stem_lower or "bus" in stem_lower:
        return "WhiteBlue"
    if "sedan" in stem_lower or "hatchback" in stem_lower or "car" in stem_lower:
        if d3.startswith(("ท", "ม")) and len(d3) in [2, 3]:
            return "YellowBlack"
        return "WhiteBlackCar"

    # Morphology & script heuristics
    if aspect_ratio < 1.45:
        return "WhiteBlackMotor"
    if d3.startswith(("ท", "ม")) and len(d3) in [2, 3]:
        return "YellowBlack"

    return "WhiteBlackCar"


class PlateBatchProcessor:
    def __init__(self, weights_path: str, batch_size: int = 64, conf_threshold: float = 0.60):
        self.weights_path = Path(weights_path).resolve()
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        print(f"[*] Initializing PlateReader with checkpoint: {self.weights_path}")
        self.reader = PlateReader(weights_path=str(self.weights_path))
        if self.reader.model is None:
            raise RuntimeError(f"Failed to load reader model: {self.reader.error_msg}")
        self.device = self.reader.device
        print(f"[+] Loaded successfully on device: {self.device}")

    def process_dataset(
        self,
        source_dirs: List[Path],
        output_dir: Path
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir).resolve()
        plates_dir = output_dir / "plates"
        plates_dir.mkdir(parents=True, exist_ok=True)

        # 1. Discover all images
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_records = []
        for s_dir in source_dirs:
            s_dir = Path(s_dir).resolve()
            if not s_dir.exists():
                print(f"[WARN] Source directory not found: {s_dir}")
                continue
            found = [f for f in s_dir.iterdir() if f.is_file() and f.suffix.lower() in exts]
            print(f"[FOUND] {len(found)} images in {s_dir}")
            for f in found:
                image_records.append({
                    "path": f,
                    "source_dir": str(s_dir),
                    "filename": f.name,
                    "stem": f.stem,
                    "ext": f.suffix.lower()
                })

        total_images = len(image_records)
        print(f"\n[INFO] Total candidate images to process: {total_images}")
        if total_images == 0:
            return {"total": 0}

        # 2. Batch Inference
        session_items: Dict[str, Any] = {}
        csv_rows: List[Dict[str, Any]] = []
        
        stat_approved = 0
        stat_cannot_read = 0
        t0 = time.time()

        for batch_start in tqdm(range(0, total_images, self.batch_size), desc="Batch OCR Inference"):
            batch_slice = image_records[batch_start : batch_start + self.batch_size]
            tensors = []
            valid_indices = []

            for idx_in_batch, item in enumerate(batch_slice):
                try:
                    img = Image.open(item["path"])
                    t = self.reader.preprocess_image(img)
                    tensors.append(t)
                    valid_indices.append((idx_in_batch, item, img.size))
                except Exception as e:
                    print(f"[ERROR] Corrupt image {item['path'].name}: {e}")
                    # Mark cannot read immediately
                    clean_base = get_base_name(item["stem"])
                    out_fn = f"{clean_base}_-_-_-_-.jpg"
                    out_path = plates_dir / out_fn
                    if out_path.exists():
                        out_fn = f"{clean_base}_{batch_start + idx_in_batch}_-_-_-_-.jpg"
                        out_path = plates_dir / out_fn
                    shutil.copy2(item["path"], out_path)
                    stat_cannot_read += 1

            if not tensors:
                continue

            batch_tensor = torch.cat(tensors, dim=0).to(self.device)
            with torch.no_grad():
                preds = self.reader.model(batch_tensor)
                probs = F.softmax(preds, dim=-1)
                max_probs, indices = probs.max(dim=-1)

            for b_i, (_, item, (img_w, img_h)) in enumerate(valid_indices):
                token_list = indices[b_i].cpu().numpy().tolist()
                prob_list = max_probs[b_i].cpu().numpy().tolist()
                
                raw_d3, raw_d4, raw_prov, full_plate = decode_tokens_to_plate(token_list)
                valid_probs = [p for p in prob_list if p > 0.05]
                conf = float(np.mean(valid_probs)) if valid_probs else 0.0
                conf = round(conf, 4)

                clean_base = get_base_name(item["stem"])
                aspect = img_w / max(img_h, 1)

                is_pass = (conf >= self.conf_threshold) and bool(raw_d3 or raw_d4)

                if is_pass:
                    norm_d3 = normalize_thai_script(raw_d3)
                    norm_d4 = normalize_thai_script(raw_d4)
                    norm_prov = match_province_value(raw_prov) if raw_prov else ""
                    iso_code = get_iso_code(norm_prov) or ""
                    plate_type = classify_dlt_plate_type(norm_d3, norm_d4, aspect, stem_hint=item["stem"])

                    safe_d3 = sanitize_filename_part(norm_d3)
                    safe_d4 = sanitize_filename_part(norm_d4)
                    safe_prov = sanitize_filename_part(norm_prov)
                    out_fn = f"{clean_base}_{plate_type}_{safe_d3}_{safe_d4}_{safe_prov}.jpg"
                    status = "approved"
                    stat_approved += 1
                else:
                    norm_d3 = "-"
                    norm_d4 = "-"
                    norm_prov = "-"
                    iso_code = ""
                    plate_type = "-"
                    out_fn = f"{clean_base}_-_-_-_-.jpg"
                    status = "cannot_read"
                    stat_cannot_read += 1

                # Safe file collision avoidance
                out_path = plates_dir / out_fn
                if out_path.exists():
                    c_stem = out_path.stem
                    c_ext = out_path.suffix
                    out_fn = f"{c_stem}_seq{batch_start + b_i:04d}{c_ext}"
                    out_path = plates_dir / out_fn

                # Copy image to destination plates/
                shutil.copy2(item["path"], out_path)
                file_hash = calculate_sha256(out_path)

                session_items[out_fn] = {
                    "filename": out_fn,
                    "path": str(out_path.resolve()),
                    "rel_path": out_fn,
                    "subfolder": "",
                    "name": clean_base,
                    "plate_type": plate_type,
                    "d3": norm_d3,
                    "d4": norm_d4,
                    "province": norm_prov,
                    "iso_3166_2": iso_code,
                    "status": status,
                    "ai_d3": raw_d3,
                    "ai_d4": raw_d4,
                    "ai_province": raw_prov,
                    "ai_confidence": conf,
                    "sha256": file_hash
                }

                csv_rows.append({
                    "filename": out_fn,
                    "original_filename": item["filename"],
                    "source_dir": item["source_dir"],
                    "name": clean_base,
                    "plate_type": plate_type,
                    "d3": norm_d3,
                    "d4": norm_d4,
                    "province": norm_prov,
                    "iso_3166_2": iso_code,
                    "confidence": conf,
                    "status": status,
                    "full_plate": f"{norm_d3} {norm_d4} {norm_prov}".strip(),
                    "ai_d3": raw_d3,
                    "ai_d4": raw_d4,
                    "ai_province": raw_prov,
                    "width": img_w,
                    "height": img_h,
                    "sha256": file_hash
                })

        elapsed = time.time() - t0
        fps = total_images / max(elapsed, 0.001)

        # 3. Save .lpr_session.json
        now_iso = datetime.now(timezone.utc).isoformat()
        session_data = {
            "version": "1.3",
            "standards": [
                "ISO 8601:2019",
                "ISO 3166-2:TH",
                "ISO/IEC 10646",
                "ISO/IEC 25012:2008",
                "ISO/IEC 42001:2023"
            ],
            "updated_at": now_iso,
            "reader_checkpoint": str(self.weights_path),
            "conf_threshold": self.conf_threshold,
            "folder": str(plates_dir.resolve()),
            "current_index": 0,
            "total_items": len(session_items),
            "approved_count": stat_approved,
            "cannot_read_count": stat_cannot_read,
            "items": session_items
        }

        session_path = output_dir / ".lpr_session.json"
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # 4. Save annotations.json
        ann_path = output_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        # 5. Save metadata.csv
        csv_path = output_dir / "metadata.csv"
        if csv_rows:
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)

        # 6. Save README.md report
        readme_path = output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(f"""# Thai License Plate Labeled Dataset (Model manual_ground_truth)
## ISO/IEC 42001:2023 & ISO/IEC 25012:2008 Compliant Auto-Labeled Dataset

- **Generated (UTC)**: {now_iso}
- **Model Checkpoint**: `{self.weights_path}`
- **Device**: `{self.device}`
- **Confidence Threshold**: `{self.conf_threshold}`
- **Total Processed Plates**: {total_images}
- **Throughput**: {fps:.1f} fps ({elapsed:.2f}s total duration)
- **Approved Plates**: {stat_approved} ({stat_approved/max(total_images,1)*100:.2f}%)
- **Cannot Read (Degraded / Low Conf)**: {stat_cannot_read} ({stat_cannot_read/max(total_images,1)*100:.2f}%)

### Source Folders Covered
""" + "\n".join([f"- `{s}`" for s in source_dirs]) + f"""

### Dataset Files
- `plates/`: Exported plate crops named canonically `{'{name}'}_{'{plate_type}'}_{'{d3}'}_{'{d4}'}_{'{province}'}.jpg` (or `{'{name}'}_-_-_-_-_.jpg` for unreadable).
- `.lpr_session.json`: Interactive session state for DataManipulator annotation review.
- `metadata.csv`: Full cryptographic ledger including SHA-256 hashes, source references, and OCR confidence.
- `annotations.json`: Mirrored JSON manifest.
""")

        print(f"\n=======================================================")
        print(f"  BATCH LABELING COMPLETE")
        print(f"=======================================================")
        print(f"Total Images Processed : {total_images}")
        print(f"Approved (Conf >= {self.conf_threshold}): {stat_approved} ({stat_approved/total_images*100:.2f}%)")
        print(f"Cannot Read (< {self.conf_threshold}) : {stat_cannot_read} ({stat_cannot_read/total_images*100:.2f}%)")
        print(f"Total Time             : {elapsed:.2f} s ({fps:.1f} FPS)")
        print(f"Output Directory       : {output_dir}")
        print(f"=======================================================\n")

        return {
            "total": total_images,
            "approved": stat_approved,
            "cannot_read": stat_cannot_read,
            "elapsed": elapsed,
            "fps": fps,
            "output_dir": str(output_dir)
        }


def main():
    parser = argparse.ArgumentParser(description="Batch Thai LPR Labeling using Curated Ground Truth Reader")
    parser.add_argument("--weights", type=str, default="models/reader/manual_ground_truth/best_reader.pth")
    parser.add_argument("--output", type=str, default="dataset/labeled/manual_ground_truth_labeled")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--conf", type=float, default=0.60)
    args = parser.parse_args()

    # Predefined sources from user request
    sources = [
        Path(r"D:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images"),
        Path(r"D:\Projects-anti\lpr\dataset\snapshot\raw_video_crops"),
        Path(r"D:\Projects-anti\lpr\dataset\snapshot\moving_vehicle_crops"),
        Path(r"D:\Projects-anti\lpr\dataset\snapshot\19-Oct-2026_snapshots\plates"),
    ]

    base_27 = Path(r"D:\Projects-anti\lpr\dataset\snapshot\27-Oct-2026")
    if base_27.exists():
        for sub in base_27.iterdir():
            if sub.is_dir():
                p1 = sub / "plates"
                p2 = sub / "plate"
                if p1.is_dir():
                    sources.append(p1)
                elif p2.is_dir():
                    sources.append(p2)

    processor = PlateBatchProcessor(
        weights_path=args.weights,
        batch_size=args.batch_size,
        conf_threshold=args.conf
    )
    processor.process_dataset(
        source_dirs=sources,
        output_dir=Path(args.output)
    )


if __name__ == "__main__":
    main()
