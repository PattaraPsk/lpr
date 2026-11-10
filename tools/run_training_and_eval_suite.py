#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Orchestrator for Dataset v3.0, Baselines & SelenOCR v0.1.0 Training
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Representations)
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.train_easyocr_v3 import train_easyocr_baseline
from tools.train_paddleocr_v3 import train_paddleocr_baseline
from tools.train_selen_ocr_v010 import train_selen_ocr
from tools.benchmark_v3_readers import evaluate_model_on_dataset
from tools.train_easyocr_v3 import EasyOCRModelV3
from tools.train_paddleocr_v3 import PaddleOCRBaselineModel
from tools.selen_ocr_model import SelenOCR
from tools.selen_token_codec import VOCAB_SIZE

DATASET_DIR = ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.0-synthetic-clean"
OUTPUT_DIR = ROOT_DIR / "runs" / "train" / "suite_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_full_suite():
    print("=" * 80)
    print("  LAUNCHING FULL OCR RETRAINING & BENCHMARK SUITE")
    print("  Models: EasyOCR Baseline, PaddleOCR Baseline, SelenOCR v0.1.0")
    print("  Dataset: v3.0 - Synthetic Clean (120,000 images)")
    print("=" * 80)

    # 1. Train EasyOCR Baseline
    print("\n>>> Phase 1: Training EasyOCR Baseline...")
    t0 = time.time()
    easy_res = train_easyocr_baseline(
        data_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR / "easyocr",
        epochs=15,
        batch_size=128,
        lr=5e-4
    )
    easy_time = time.time() - t0
    print(f"EasyOCR Training completed in {easy_time:.1f}s | Best Val Acc: {easy_res['best_val_acc']:.2f}%")

    # 2. Train PaddleOCR Baseline
    print("\n>>> Phase 2: Training PaddleOCR Baseline...")
    t0 = time.time()
    paddle_res = train_paddleocr_baseline(
        data_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR / "paddleocr",
        epochs=15,
        batch_size=128,
        lr=5e-4
    )
    paddle_time = time.time() - t0
    print(f"PaddleOCR Training completed in {paddle_time:.1f}s | Best Val Acc: {paddle_res['best_val_acc']:.2f}%")

    # 3. Train SelenOCR v0.1.0 (Enlarged Multi-Task Reader)
    print("\n>>> Phase 3: Training SelenOCR v0.1.0...")
    t0 = time.time()
    selen_res = train_selen_ocr(
        data_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR / "selen_ocr_v010",
        epochs=25,
        batch_size=128,
        lr=5e-4
    )
    selen_time = time.time() - t0
    print(f"SelenOCR v0.1.0 Training completed in {selen_time:.1f}s | Best Val Acc: {selen_res['best_val_acc']:.2f}%")

    # 4. Comparative Evaluation on Held-Out Test Split (5,000 samples)
    print("\n>>> Phase 4: Comparative Benchmarking on Test Samples...")
    with open(DATASET_DIR / "metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    test_records = all_records[-5000:]

    # Evaluate EasyOCR
    easy_model = EasyOCRModelV3(input_channel=1, output_channel=512, hidden_size=256, num_class=VOCAB_SIZE)
    easy_chkpt = torch.load(ROOT_DIR / "models" / "reader" / "easyocr_v3" / "best_reader.pth", map_location="cuda")
    easy_model.load_state_dict(easy_chkpt["model_state_dict"])
    easy_bench = evaluate_model_on_dataset(easy_model, "easyocr", test_records, device="cuda")

    # Evaluate PaddleOCR
    paddle_model = PaddleOCRBaselineModel(d_model=256, hidden_size=128, num_class=VOCAB_SIZE)
    paddle_chkpt = torch.load(ROOT_DIR / "models" / "reader" / "paddleocr_v3" / "best_reader.pth", map_location="cuda")
    paddle_model.load_state_dict(paddle_chkpt["model_state_dict"])
    paddle_bench = evaluate_model_on_dataset(paddle_model, "paddleocr", test_records, device="cuda")

    # Evaluate SelenOCR v0.1.0
    selen_model = SelenOCR(d_model=384, nhead=6, num_encoder_layers=4, lstm_hidden=256, vocab_size=VOCAB_SIZE)
    selen_chkpt = torch.load(ROOT_DIR / "models" / "reader" / "selen_ocr_v010" / "best_reader.pth", map_location="cuda")
    selen_model.load_state_dict(selen_chkpt["model_state_dict"])
    selen_bench = evaluate_model_on_dataset(selen_model, "selen_ocr", test_records, device="cuda")

    summary_report = {
        "benchmark_title": "Thai LPR v3.0 Reader Comparison (EasyOCR vs PaddleOCR vs SelenOCR v0.1.0)",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": {
            "easyocr": easy_bench,
            "paddleocr": paddle_bench,
            "selen_ocr_v010": selen_bench
        }
    }

    report_path = OUTPUT_DIR / "benchmark_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print("  FINAL COMPARATIVE BENCHMARK SUMMARY (5,000 TEST SAMPLES)")
    print("=" * 80)
    print(f"{'Model Architecture':<25} | {'Exact Acc':<10} | {'Type Acc':<10} | {'d3 Acc':<10} | {'d4 Acc':<10} | {'Prov Acc':<10} | {'Neg Acc':<10} | {'Latency':<10} | {'FPS':<8}")
    print("-" * 115)
    for k, m in summary_report["models"].items():
        print(f"{m['model_type']:<25} | {m['exact_accuracy_pct']:>8.2f}% | {m['plate_type_accuracy_pct']:>8.2f}% | {m['d3_accuracy_pct']:>8.2f}% | {m['d4_accuracy_pct']:>8.2f}% | {m['province_accuracy_pct']:>8.2f}% | {m['negative_rejection_pct']:>8.2f}% | {m['avg_latency_ms']:>6.2f} ms | {m['fps']:>6.1f}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_suite()
