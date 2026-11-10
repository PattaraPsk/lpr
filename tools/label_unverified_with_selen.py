#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Unverified Images with SelenLPR 14-Timestep Model & DLT Length Clamping.
Complies with:
- ISO 8601:2019 (UTC Timestamps)
- ISO/IEC 42001:2023 (Annex A.7 / A.9.3 Operational Event Logging)
- ISO/IEC 25012:2008 (Data Quality & SHA-256 Lineage)
- ISO 3166-2:TH (Thailand Administrative Subdivisions)
- Thailand DLT Specifications: d3 <= 3 chars, d4 <= 4 digits
"""

import os
import sys
import re
import json
import csv
import math
import hashlib
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from PIL import Image

# Ensure repo root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.train_selen_lpr import (
    SelenLPR,
    CTC_CHARS,
    CTC_IDX_TO_CHAR,
    decode_and_split_ctc,
    PLATE_TYPES,
    PROVINCE_NAMES,
    VOCAB_SIZE,
)
from tools.DataManipulator.app.provinces import get_iso_code, normalize_thai_script

# DLT standard limits
MAX_D3_LEN = 3
MAX_D4_LEN = 4


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()


def sanitize_component(text: str) -> str:
    if not text:
        return "-"
    cleaned = re.sub(r'[\\/*?:"<>|]', "", text).strip()
    return cleaned if cleaned else "-"


def clamp_d3_d4(d3: str, d4: str) -> Tuple[str, str]:
    """
    Enforce DLT specifications and strict length clamping:
    1. Consonant normalization/remapping (ฃ->ข, ฅ->ค, ฏ->ฎ).
    2. Filter d4 to digits only (0-9).
    3. Strict length clamping: d3 <= 3 characters, d4 <= 4 digits.
    """
    d3_remapped = d3.replace("ฃ", "ข").replace("ฅ", "ค").replace("ฏ", "ฎ")
    d3_clamped = d3_remapped[:MAX_D3_LEN]

    d4_digits = "".join(c for c in d4 if c.isdigit())
    d4_clamped = d4_digits[:MAX_D4_LEN]

    return d3_clamped, d4_clamped


def preprocess_plate_image(im: Image.Image, img_h: int = 64, img_w: int = 384) -> torch.Tensor:
    """
    Standard aspect-ratio preserving resize and border padding for SelenLPR (64x384).
    Matches MultiTaskPlateDataset normalization: (arr - 0.5) / 0.5.
    """
    w, h = im.size
    ratio = w / float(max(h, 1))
    resized_w = min(int(math.ceil(img_h * ratio)), img_w)
    resized_crop = im.resize((resized_w, img_h), Image.BICUBIC)

    arr = np.array(resized_crop, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    t = torch.from_numpy(arr).permute(2, 0, 1)

    pad = torch.zeros(3, img_h, img_w, dtype=torch.float32)
    pad[:, :, :resized_w] = t
    if img_w != resized_w:
        pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(3, img_h, img_w - resized_w)
    return pad.unsqueeze(0)


def main():
    parser = argparse.ArgumentParser(description="Label unverified plates with SelenLPR 14-timestep model and d3/d4 clamping")
    parser.add_argument("--limit", type=int, default=100, help="Number of unverified images to label (default: 100)")
    parser.add_argument("--src-dir", type=str, default=str(ROOT_DIR / "dataset" / "snapshot" / "raw_video_crops"), help="Source crop directory")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "dataset" / "labeled" / "unverified_labeled_100"), help="Output directory")
    parser.add_argument("--weights", type=str, default=str(ROOT_DIR / "models" / "reader" / "selen-lpr-0.0.1" / "best_reader.pth"), help="Model weights path")
    parser.add_argument("--time-steps", type=int, default=14, help="CTC sequence time steps (default: 14)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device)
    now_utc = datetime.now(timezone.utc).isoformat()
    print(f"[{now_utc}] [AIMS:A.9.3] Loading SelenLPR 14-timestep reader on {device}...")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    model_hash = compute_sha256(weights_path)
    print(f"Model Checkpoint: {weights_path} (SHA-256: {model_hash})")
    print(f"CTC Sequence Time Steps: {args.time_steps} | Length Clamping: d3<={MAX_D3_LEN}, d4<={MAX_D4_LEN}")

    model = SelenLPR(vocab_size=len(CTC_CHARS), time_steps=args.time_steps)
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and any("ctc_head" in k for k in ckpt.keys()):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    # Source unverified images
    src_dir = Path(args.src_dir)
    output_dir = Path(args.output_dir)
    plates_output_dir = output_dir / "plates"
    
    # Reset/ensure output directory
    if plates_output_dir.exists():
        for old_f in plates_output_dir.iterdir():
            if old_f.is_file():
                old_f.unlink()
    plates_output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted([f for f in src_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    selected_files = all_files[:args.limit]
    print(f"Found {len(all_files)} total images in {src_dir}. Selecting exactly {len(selected_files)} images for labeling.")

    items: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    print("\n--- Running Inference & Enforcing d3/d4 Length Clamped Labels ---")
    for idx, img_path in enumerate(selected_files, 1):
        orig_sha = compute_sha256(img_path)
        img = Image.open(img_path).convert("RGB")
        tensor = preprocess_plate_image(img).to(device)

        with torch.no_grad():
            preds = model(tensor)
            ctc_logits = preds["ctc_logits"]
            type_logits = preds["type_logits"]
            prov_logits = preds["prov_logits"]

            # CTC decode
            pairs = decode_and_split_ctc(ctc_logits, CTC_IDX_TO_CHAR)
            raw_d3, raw_d4 = pairs[0]

            # Classification
            type_idx = type_logits.argmax(dim=-1).item()
            prov_idx = prov_logits.argmax(dim=-1).item()

            pred_type = PLATE_TYPES[type_idx]
            pred_prov = PROVINCE_NAMES[prov_idx]

            # Confidence estimation
            ctc_probs = torch.softmax(ctc_logits, dim=-1)
            max_ctc_prob = ctc_probs.max(dim=-1).values.mean().item()
            type_prob = torch.softmax(type_logits, dim=-1)[0, type_idx].item()
            prov_prob = torch.softmax(prov_logits, dim=-1)[0, prov_idx].item()
            overall_conf = round(float((max_ctc_prob + type_prob + prov_prob) / 3.0), 4)

        # Enforce DLT length clamping & character sanitation
        d3_pred, d4_pred = clamp_d3_d4(raw_d3, raw_d4)

        iso_prov = get_iso_code(pred_prov) or "TH-UNKNOWN"

        clean_d3 = sanitize_component(d3_pred)
        clean_d4 = sanitize_component(d4_pred)
        clean_prov = sanitize_component(pred_prov)
        clean_type = sanitize_component(pred_type)

        base_stem = img_path.stem
        new_filename = f"{base_stem}_{clean_type}_{clean_d3}_{clean_d4}_{clean_prov}{img_path.suffix.lower()}"
        dest_path = plates_output_dir / new_filename

        shutil.copy2(img_path, dest_path)
        dest_sha = compute_sha256(dest_path)

        full_number = f"{d3_pred} {d4_pred}".strip() if (d3_pred or d4_pred) else "-"

        item_data = {
            "source_file": img_path.name,
            "labeled_filename": new_filename,
            "sha256": dest_sha,
            "plate_type": pred_type,
            "d3": d3_pred,
            "d4": d4_pred,
            "raw_d3": raw_d3,
            "raw_d4": raw_d4,
            "d3_clamped_length": len(d3_pred),
            "d4_clamped_length": len(d4_pred),
            "full_plate_number": full_number,
            "province": pred_prov,
            "iso_3166_2": iso_prov,
            "confidence": overall_conf,
            "ctc_confidence": round(max_ctc_prob, 4),
            "type_confidence": round(type_prob, 4),
            "prov_confidence": round(prov_prob, 4),
            "verification_status": "ai_labeled_unverified",
            "model_version": "selen-lpr-0.0.1",
            "model_time_steps": args.time_steps,
            "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        items[new_filename] = item_data

        csv_rows.append({
            "index": idx,
            "source_file": img_path.name,
            "labeled_filename": new_filename,
            "plate_type": pred_type,
            "d3": d3_pred,
            "d4": d4_pred,
            "full_number": full_number,
            "province": pred_prov,
            "iso_3166_2": iso_prov,
            "confidence": overall_conf,
            "sha256": dest_sha,
        })

        if idx <= 5 or idx % 20 == 0 or idx == len(selected_files):
            print(f"[{idx:3d}/{len(selected_files)}] {img_path.name:<25} -> Type: {pred_type:<15} Plate: {clean_d3} {clean_d4} {clean_prov} (Conf: {overall_conf:.2f})")

    # Save annotations.json
    annotations_doc = {
        "version": "1.1",
        "standards": [
            "ISO 8601:2019",
            "ISO 3166-2:TH",
            "ISO/IEC 10646",
            "ISO/IEC 25012:2008",
            "ISO/IEC 42001:2023",
            "Thailand DLT Specification"
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_used": "selen-lpr-0.0.1",
        "model_time_steps": args.time_steps,
        "model_sha256": model_hash,
        "clamping_policy": {
            "max_d3_length": MAX_D3_LEN,
            "max_d4_length": MAX_D4_LEN,
            "d4_numeric_only": True,
            "consonant_remapping": {"ฃ": "ข", "ฅ": "ค", "ฏ": "ฎ"}
        },
        "total_images": len(items),
        "items": items,
    }

    with open(output_dir / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations_doc, f, ensure_ascii=False, indent=2)

    # Save metadata.csv
    csv_fields = [
        "index", "source_file", "labeled_filename", "plate_type",
        "d3", "d4", "full_number", "province", "iso_3166_2", "confidence", "sha256"
    ]
    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n[SUCCESS] Completed labeling {len(selected_files)} images with 14-timestep model & d3/d4 length clamping.")
    print(f"Export directory: {output_dir}")
    print(f"Annotations JSON: {output_dir / 'annotations.json'}")
    print(f"Metadata CSV:     {output_dir / 'metadata.csv'}")


if __name__ == "__main__":
    main()
