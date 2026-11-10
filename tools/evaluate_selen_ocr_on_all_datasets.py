#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Dataset Empirical Evaluation Suite for SelenOCR v0.1.0
Evaluates SelenOCR v0.1.0 across all previous project datasets:
1. 202610261140 (Anchor Real Verified CCTV Plates)
2. 202611030335 (Expanded Real Verified CCTV/Mobile Plates)
3. dataset-cctv-synthetic (CCTV Profile Synthetic Benchmark)
4. dataset-mobile-synthetic (Mobile Optical Profile Synthetic Benchmark)
5. dataset-dirty10-blur5-angle5 (Degraded Environmental Benchmark: 10% dirty, 5% blur, 5% tilt)
6. dataset-v3.0-synthetic-clean (Clean Synthetic Target Benchmark)

Complies with ISO/IEC 42001:2023 (Annex A.6.7 AI Change Management & Verification)
"""

import sys
import time
import json
import csv
import random
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

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

import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm

from tools.selen_token_codec import (
    VOCAB_SIZE,
    PAD_ID,
    decode_tokens_to_plate,
    PROV_TO_IDX,
    PLATE_TYPE_TO_IDX
)
from tools.selen_ocr_model import SelenOCR

TYPE_NORMALIZATION = {
    "YellowHeavy": "YellowBlack",
    "WhiteHeavy": "WhiteBlackCar",
    "YellowBlackMotor": "YellowBlack",
    "RedBlackCar": "WhiteBlackCar",
    "GreenWhite": "WhiteGreen"
}


def load_dataset_records(dataset_name: str, base_dir: Path, max_samples: int = 2000) -> List[Dict[str, Any]]:
    """Loads and standardizes records from any project dataset version."""
    v_dir = base_dir / dataset_name
    meta_path = v_dir / "metadata.csv"
    if not meta_path.exists():
        return []

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    records = []
    plates_dir = v_dir / "plates"

    for r in raw_rows:
        fn = r.get("filename", "")
        fp_str = r.get("filepath", "")
        if fp_str and Path(fp_str).exists():
            img_path = Path(fp_str)
        else:
            img_path = plates_dir / fn

        if not img_path.exists():
            continue

        raw_type = r.get("plate_type", "WhiteBlackCar").strip()
        p_type = TYPE_NORMALIZATION.get(raw_type, raw_type)

        d3 = r.get("d3", "").strip()
        d4 = r.get("d4", "").strip()
        prov = unicodedata.normalize("NFC", r.get("province", "กรุงเทพมหานคร").strip())

        # Construct full ground truth line
        gt_line = f"{p_type} {d3} {d4} {prov}".strip()

        records.append({
            "filepath": str(img_path),
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov,
            "full_text_line": gt_line,
            "dataset": dataset_name
        })

    if max_samples and len(records) > max_samples:
        random.seed(42)
        records = random.sample(records, max_samples)

    return records


def evaluate_selen_on_records(
    model: nn.Module,
    records: List[Dict[str, Any]],
    device: str = "cuda",
    batch_size: int = 128
) -> Dict[str, Any]:
    """Runs batched inference and calculates accuracy metrics."""
    model.eval()
    total = len(records)
    if total == 0:
        return {}

    exact_matches = 0
    reg_matches = 0 # d3 + d4 match
    type_matches = 0
    d3_matches = 0
    d4_matches = 0
    prov_matches = 0

    mean_rgb = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_rgb = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0_eval = time.perf_counter()

    for i in range(0, total, batch_size):
        batch = records[i:min(i + batch_size, total)]
        tensors = []

        for r in batch:
            try:
                with open(r["filepath"], "rb") as f:
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                    im_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if im_bgr is not None:
                    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (320, 64), interpolation=cv2.INTER_LINEAR)
                else:
                    im = np.full((64, 320, 3), 128, dtype=np.uint8)
            except Exception:
                im = np.full((64, 320, 3), 128, dtype=np.uint8)
            tensors.append(im)

        raw = torch.from_numpy(np.stack(tensors, axis=0)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        batch_tensor = (raw - mean_rgb) / std_rgb

        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = model(batch_tensor)
            ctc_logits = outputs["ctc_logits"]

        pred_seqs = ctc_logits.argmax(dim=-1).cpu().tolist()

        for j, seq in enumerate(pred_seqs):
            r = batch[j]
            gt_type = r["plate_type"]
            gt_d3 = r["d3"]
            gt_d4 = r["d4"]
            gt_prov = r["province"]
            gt_line = r["full_text_line"]

            collapsed = [k for idx, k in enumerate(seq) if k != PAD_ID and (idx == 0 or k != seq[idx - 1])]
            pred_type, pred_d3, pred_d4, pred_prov, pred_line = decode_tokens_to_plate(collapsed)

            # Match checks
            is_type_match = (pred_type == gt_type)
            is_d3_match = (pred_d3 == gt_d3)
            is_d4_match = (pred_d4 == gt_d4)
            is_prov_match = (pred_prov == gt_prov)
            is_reg_match = (is_d3_match and is_d4_match)
            is_exact_match = (is_type_match and is_reg_match and is_prov_match)

            if is_exact_match:
                exact_matches += 1
            if is_reg_match:
                reg_matches += 1
            if is_type_match:
                type_matches += 1
            if is_d3_match:
                d3_matches += 1
            if is_d4_match:
                d4_matches += 1
            if is_prov_match:
                prov_matches += 1

    total_time = time.perf_counter() - t0_eval
    fps = total / total_time if total_time > 0 else 0
    ms_per_img = (total_time / total) * 1000.0 if total > 0 else 0

    return {
        "total_samples": total,
        "exact_match_pct": round((exact_matches / total) * 100.0, 2),
        "registration_pct": round((reg_matches / total) * 100.0, 2),
        "d3_pct": round((d3_matches / total) * 100.0, 2),
        "d4_pct": round((d4_matches / total) * 100.0, 2),
        "province_pct": round((prov_matches / total) * 100.0, 2),
        "type_pct": round((type_matches / total) * 100.0, 2),
        "avg_latency_ms": round(ms_per_img, 2),
        "fps": round(fps, 1)
    }


def run_cross_dataset_eval(chkpt_path: Optional[Path] = None, version_tag: str = "v0.1.1", out_json_path: Optional[Path] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if chkpt_path is None:
        chkpt_path = ROOT_DIR / "models" / "reader" / f"selen_ocr_{version_tag.replace('.', '')}" / "best_reader.pth"
        if not chkpt_path.exists():
            chkpt_path = ROOT_DIR / "models" / "reader" / "selen_ocr_v011" / "best_reader.pth"

    if not chkpt_path.exists():
        print(f"Error: SelenOCR checkpoint not found at {chkpt_path}")
        sys.exit(1)

    print("=" * 110)
    print(f"  CROSS-DATASET EMPIRICAL EVALUATION SUITE: SELENOCR {version_tag.upper()}")
    print(f"  Checkpoint: {chkpt_path}")
    print(f"  Device: {device.upper()} | Precision: FP16 AMP")
    print("=" * 110)

    model = SelenOCR(d_model=384, nhead=6, num_encoder_layers=4, lstm_hidden=256, vocab_size=VOCAB_SIZE)
    chkpt = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device)

    versions_dir = ROOT_DIR / "dataset" / "reading" / "versions"
    datasets_to_test = [
        ("202610261140", "Real Anchor Verified CCTV Plates (1,183 images)", 1183),
        ("202611030335", "Expanded Real Verified CCTV/Mobile (2,000 sampled)", 2000),
        ("dataset-cctv-synthetic", "CCTV Profile Synthetic Benchmark (1,500 images)", 1500),
        ("dataset-mobile-synthetic", "Mobile Camera Synthetic Benchmark (1,500 images)", 1500),
        ("dataset-dirty10-blur5-angle5", "Degraded Environmental Benchmark (2,000 sampled)", 2000),
        ("dataset-v3.0-synthetic-clean", "Clean Synthetic Target Benchmark (2,000 stratified)", 2000),
        ("dataset-v3.1-cctv-addon", "CCTV & Low-Resolution Add-on (2,000 sampled)", 2000),
    ]

    all_results = {}

    for d_name, desc, max_n in datasets_to_test:
        print(f"\n>>> Evaluating on: {d_name} ({desc})...")
        recs = load_dataset_records(d_name, versions_dir, max_samples=max_n)
        if not recs:
            print(f"    [Warning] No records found for {d_name}. Skipping.")
            continue

        res = evaluate_selen_on_records(model, recs, device=device)
        res["description"] = desc
        all_results[d_name] = res

        print(f"    Total: {res['total_samples']:,} | Exact: {res['exact_match_pct']}% | "
              f"Reg (d3+d4): {res['registration_pct']}% | Type: {res['type_pct']}% | Prov: {res['province_pct']}% | Latency: {res['avg_latency_ms']} ms")

    # Output consolidated report table
    print("\n" + "=" * 110)
    print(f"  SELENOCR {version_tag.upper()} CROSS-DATASET BENCHMARK SUMMARY")
    print("=" * 110)
    print(f"{'Dataset Name':<30} | {'Samples':<8} | {'Exact Acc':<10} | {'Reg (d3+d4)':<12} | {'d3 Acc':<8} | {'d4 Acc':<8} | {'Prov Acc':<9} | {'Type Acc':<9} | {'FPS':<6}")
    print("-" * 110)
    for name, r in all_results.items():
        print(f"{name:<30} | {r['total_samples']:>7,} | {r['exact_match_pct']:>8.2f}% | {r['registration_pct']:>10.2f}% | {r['d3_pct']:>6.2f}% | {r['d4_pct']:>6.2f}% | {r['province_pct']:>7.2f}% | {r['type_pct']:>7.2f}% | {r['fps']:>5.1f}")
    print("=" * 110)

    # Save JSON report
    if out_json_path is None:
        out_json_path = ROOT_DIR / "runs" / f"selen_ocr_{version_tag.replace('.', '')}_cross_dataset_benchmark.json"
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved complete cross-dataset audit report to: {out_json_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SelenOCR on all project datasets")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument("--version", type=str, default="v0.1.1", help="Version tag (e.g. v0.1.0 or v0.1.1)")
    parser.add_argument("--out-json", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    chk = Path(args.checkpoint) if args.checkpoint else None
    out_p = Path(args.out_json) if args.out_json else None
    run_cross_dataset_eval(chkpt_path=chk, version_tag=args.version, out_json_path=out_p)
