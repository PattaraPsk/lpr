#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified LPR Benchmark Suite for Dataset v3.0 (Synthetic Clean)
Evaluates EasyOCR Baseline, PaddleOCR Baseline, and SelenOCR v0.1.0 side-by-side.
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Representations)

Metrics Tracked:
1. Exact Full-Text Line Accuracy: "plateType d3 d4 province"
2. Plate Type Classification Accuracy
3. Prefix Sequence (d3) Accuracy
4. Digit Sequence (d4) Accuracy
5. Closed-Set Province Accuracy
6. Negative Condition Accuracy ('+' for not-plate, '-' for unreadable blur)
7. Inference Latency (ms/frame) & Throughput (FPS) on GPU & CPU
"""

import os
import sys
import time
import json
import csv
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.selen_token_codec import (
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    decode_tokens_to_plate
)
from tools.train_easyocr_v3 import EasyOCRModelV3, ThaiPlateDatasetV3
from tools.train_paddleocr_v3 import PaddleOCRBaselineModel, PaddleDatasetV3
from tools.selen_ocr_model import SelenOCR
from tools.train_selen_ocr_v010 import SelenOCRDataset, TYPE_TO_HEAD_IDX, PROV_TO_HEAD_IDX


import cv2


def evaluate_model_on_dataset(
    model: nn.Module,
    model_type: str,
    records: List[Dict[str, Any]],
    device: str = "cuda",
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    High-throughput batched evaluation of a model over test records.
    Computes exact accuracy, sub-sequence accuracy, negative rejection, and GPU/CPU latency.
    """
    model.eval()
    model.to(device)

    total = len(records)
    exact_matches = 0
    type_matches = 0
    d3_matches = 0
    d4_matches = 0
    prov_matches = 0
    negative_matches = 0
    total_negatives = 0

    mean_rgb = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_rgb = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    for i in range(0, total, batch_size):
        batch_recs = records[i:min(i + batch_size, total)]
        cur_bs = len(batch_recs)

        if model_type == "easyocr":
            # Grayscale padded to (64, 600)
            tensors = []
            for r in batch_recs:
                im = cv2.imread(r["filepath"], cv2.IMREAD_GRAYSCALE)
                if im is None:
                    im = np.full((64, 600), 128, dtype=np.uint8)
                else:
                    h, w = im.shape[:2]
                    ratio = w / float(h)
                    rw = min(600, max(16, int(math.ceil(64 * ratio))))
                    r_im = cv2.resize(im, (rw, 64), interpolation=cv2.INTER_LINEAR)
                    p_im = np.full((64, 600), 128, dtype=np.uint8)
                    p_im[:, :rw] = r_im
                    im = p_im
                arr = (im.astype(np.float32) / 255.0 - 0.5) / 0.5
                tensors.append(arr)
            batch_tensor = torch.from_numpy(np.stack(tensors, axis=0)).unsqueeze(1).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                preds = model(batch_tensor) # [B, T, Vocab]

            pred_seqs = preds.argmax(dim=-1).cpu().tolist()

        elif model_type == "paddleocr":
            # RGB padded to (48, 320)
            tensors = []
            for r in batch_recs:
                im = cv2.imread(r["filepath"])
                if im is None:
                    im = np.full((48, 320, 3), 128, dtype=np.uint8)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    h, w = im.shape[:2]
                    ratio = w / float(h)
                    rw = min(320, max(16, int(48 * ratio)))
                    r_im = cv2.resize(im, (rw, 48), interpolation=cv2.INTER_LINEAR)
                    p_im = np.full((48, 320, 3), 128, dtype=np.uint8)
                    p_im[:, :rw] = r_im
                    im = p_im
                tensors.append(im)
            raw = torch.from_numpy(np.stack(tensors, axis=0)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            batch_tensor = (raw - mean_rgb) / std_rgb

            with torch.no_grad(), torch.amp.autocast("cuda"):
                preds = model(batch_tensor)

            pred_seqs = preds.argmax(dim=-1).cpu().tolist()

        else: # selen_ocr
            # Direct RGB resize to (64, 320)
            tensors = []
            for r in batch_recs:
                im = cv2.imread(r["filepath"])
                if im is None:
                    im = np.full((64, 320, 3), 128, dtype=np.uint8)
                else:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, (320, 64), interpolation=cv2.INTER_LINEAR)
                tensors.append(im)
            raw = torch.from_numpy(np.stack(tensors, axis=0)).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            batch_tensor = (raw - mean_rgb) / std_rgb

            with torch.no_grad(), torch.amp.autocast("cuda"):
                outputs = model(batch_tensor)
                ctc_logits = outputs["ctc_logits"]

            pred_seqs = ctc_logits.argmax(dim=-1).cpu().tolist()

        for j, seq in enumerate(pred_seqs):
            r = batch_recs[j]
            gt_type = r["plate_type"]
            gt_d3 = r["d3"]
            gt_d4 = r["d4"]
            gt_prov = r["province"]
            gt_line = r["full_text_line"].strip()
            is_neg = (gt_type in ("+", "-"))
            if is_neg:
                total_negatives += 1

            collapsed = [k for idx, k in enumerate(seq) if k != PAD_ID and (idx == 0 or k != seq[idx - 1])]
            pred_type, pred_d3, pred_d4, pred_prov, pred_line = decode_tokens_to_plate(collapsed)

            if pred_line.strip() == gt_line:
                exact_matches += 1
            if pred_type == gt_type:
                type_matches += 1
            if pred_d3 == gt_d3:
                d3_matches += 1
            if pred_d4 == gt_d4:
                d4_matches += 1
            if pred_prov == gt_prov:
                prov_matches += 1
            if is_neg and pred_type == gt_type:
                negative_matches += 1

    # Measure exact single-frame inference latency (10 warmup, 100 timed)
    latencies = []
    if model_type == "easyocr":
        dummy = torch.randn(1, 1, 64, 600, device=device)
    elif model_type == "paddleocr":
        dummy = torch.randn(1, 3, 48, 320, device=device)
    else:
        dummy = torch.randn(1, 3, 64, 320, device=device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(10):
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        for _ in range(25):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    return {
        "model_type": model_type,
        "total_samples": total,
        "exact_accuracy_pct": round((exact_matches / total) * 100.0, 2),
        "plate_type_accuracy_pct": round((type_matches / total) * 100.0, 2),
        "d3_accuracy_pct": round((d3_matches / total) * 100.0, 2),
        "d4_accuracy_pct": round((d4_matches / total) * 100.0, 2),
        "province_accuracy_pct": round((prov_matches / total) * 100.0, 2),
        "negative_rejection_pct": round((negative_matches / total_negatives) * 100.0, 2) if total_negatives > 0 else 100.0,
        "avg_latency_ms": round(avg_latency, 2),
        "fps": round(fps, 1)
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark EasyOCR, PaddleOCR, and SelenOCR v0.1.0 on Dataset v3.0")
    parser.add_argument("--samples", type=int, default=2000, help="Number of test samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default="runs/benchmark_v3_results.json")
    args = parser.parse_args()

    data_dir = ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.0-synthetic-clean"
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"Error: {meta_path} not found.")
        sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    import random
    random.seed(42)
    # Stratified benchmark split: 80% clean plates, 10% blur plates, 10% non-plate crops
    clean_recs = [r for r in all_records if r["plate_type"] not in ("+", "-")]
    blur_recs = [r for r in all_records if r["plate_type"] == "-"]
    nonplate_recs = [r for r in all_records if r["plate_type"] == "+"]

    n_clean = int(args.samples * 0.80)
    n_blur = int(args.samples * 0.10)
    n_nonplate = args.samples - n_clean - n_blur

    test_records = (
        random.sample(clean_recs, min(n_clean, len(clean_recs))) +
        random.sample(blur_recs, min(n_blur, len(blur_recs))) +
        random.sample(nonplate_recs, min(n_nonplate, len(nonplate_recs)))
    )
    random.shuffle(test_records)

    print(f"Loaded {len(test_records):,} stratified evaluation samples (Clean: {n_clean}, Blur: {n_blur}, Non-plate: {n_nonplate})")

    results = {}

    # 1. EasyOCR
    easy_path = ROOT_DIR / "models" / "reader" / "easyocr_v3" / "best_reader.pth"
    if easy_path.exists():
        print(f"\n--- Evaluating EasyOCR Baseline ({easy_path}) ---")
        easy_model = EasyOCRModelV3(input_channel=1, output_channel=512, hidden_size=256, num_class=VOCAB_SIZE)
        chkpt = torch.load(easy_path, map_location=args.device)
        easy_model.load_state_dict(chkpt["model_state_dict"])
        results["EasyOCR Baseline"] = evaluate_model_on_dataset(easy_model, "easyocr", test_records, device=args.device)

    # 2. PaddleOCR
    paddle_path = ROOT_DIR / "models" / "reader" / "paddleocr_v3" / "best_reader.pth"
    if paddle_path.exists():
        print(f"\n--- Evaluating PaddleOCR Baseline ({paddle_path}) ---")
        paddle_model = PaddleOCRBaselineModel(d_model=256, hidden_size=128, num_class=VOCAB_SIZE)
        chkpt = torch.load(paddle_path, map_location=args.device)
        paddle_model.load_state_dict(chkpt["model_state_dict"])
        results["PaddleOCR Baseline"] = evaluate_model_on_dataset(paddle_model, "paddleocr", test_records, device=args.device)

    # 3. SelenOCR v0.1.0
    selen_path = ROOT_DIR / "models" / "reader" / "selen_ocr_v010" / "best_reader.pth"
    if selen_path.exists():
        print(f"\n--- Evaluating SelenOCR v0.1.0 ({selen_path}) ---")
        selen_model = SelenOCR(d_model=384, nhead=6, num_encoder_layers=4, lstm_hidden=256, vocab_size=VOCAB_SIZE)
        chkpt = torch.load(selen_path, map_location=args.device)
        selen_model.load_state_dict(chkpt["model_state_dict"])
        results["SelenOCR v0.1.0"] = evaluate_model_on_dataset(selen_model, "selen_ocr", test_records, device=args.device)

    out_file = ROOT_DIR / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_test_samples": len(test_records),
        "device": args.device,
        "models": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 115)
    print(f"  BENCHMARK SUMMARY REPORT ({len(test_records):,} TEST SAMPLES | DEVICE: {args.device.upper()})")
    print("=" * 115)
    print(f"{'Model Architecture':<22} | {'Exact Acc':<10} | {'Type Acc':<10} | {'d3 Acc':<10} | {'d4 Acc':<10} | {'Prov Acc':<10} | {'Neg Acc':<10} | {'Latency':<10} | {'FPS':<8}")
    print("-" * 115)
    for name, m in results.items():
        print(f"{name:<22} | {m['exact_accuracy_pct']:>8.2f}% | {m['plate_type_accuracy_pct']:>8.2f}% | {m['d3_accuracy_pct']:>8.2f}% | {m['d4_accuracy_pct']:>8.2f}% | {m['province_accuracy_pct']:>8.2f}% | {m['negative_rejection_pct']:>8.2f}% | {m['avg_latency_ms']:>6.2f} ms | {m['fps']:>6.1f}")
    print("=" * 115)
    print(f"Saved benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
