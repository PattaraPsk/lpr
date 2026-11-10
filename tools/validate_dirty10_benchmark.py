#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation Benchmark: SelenLPR vs. Custom PaddleOCR vs. thai-lpr-reader
Dataset: dataset-dirty10-blur5-angle5 (20,000 synthetic CCTV/edge plates)

Standards Compliance:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality and Integrity)
- ISO 8601:2019 (Date and Time Representation in UTC)
- Thailand DLT Vehicle & License Plate Specifications
"""

import os
import sys
import io
import time
import math
import csv
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure repo root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure UTF-8 output on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import (
    VisionAttentionOCR,
    decode_tokens_to_plate,
    match_province_value
)
from tools.train_selen_lpr import (
    SelenLPR,
    CTC_IDX_TO_CHAR as SELEN_CTC_IDX_TO_CHAR,
    decode_and_split_ctc,
    PROVINCE_NAMES as SELEN_PROVINCES
)
from tools.train_paddleocr_thai import (
    SelenPaddleOCR,
    decode_and_split_ctc_single_char_prov,
    AUX_IDX_TO_PROV
)
from tools.label_unverified_with_selen import clamp_d3_d4


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def cer_score(pred: str, gt: str) -> float:
    if not gt:
        return 0.0 if not pred else 1.0
    dist = levenshtein_distance(pred, gt)
    return dist / len(gt)


def load_single_image(filepath: str) -> Optional[Image.Image]:
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except Exception as e:
        return None


def prep_selen_paddle_batch(images: List[Image.Image], img_h: int = 64, img_w: int = 384) -> torch.Tensor:
    B = len(images)
    batch_tensor = torch.zeros((B, 3, img_h, img_w), dtype=torch.float32)
    for i, im in enumerate(images):
        if im is None:
            continue
        rgb = im.convert("RGB")
        w, h = rgb.size
        ratio = w / float(max(h, 1))
        rw = min(int(math.ceil(img_h * ratio)), img_w)
        rc = rgb.resize((rw, img_h), Image.BICUBIC)
        arr = np.array(rc, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        batch_tensor[i, :, :, :rw] = torch.from_numpy(arr).permute(2, 0, 1)
    return batch_tensor


def prep_thai_lpr_batch(images: List[Image.Image], img_h: int = 64, img_w: int = 600) -> torch.Tensor:
    B = len(images)
    batch_tensor = torch.zeros((B, 1, img_h, img_w), dtype=torch.float32)
    for i, im in enumerate(images):
        if im is None:
            continue
        gray = im.convert("L")
        w, h = gray.size
        ratio = w / float(max(h, 1))
        rw = min(int(math.ceil(img_h * ratio)), img_w)
        rc = gray.resize((rw, img_h), Image.BICUBIC)
        arr = np.array(rc, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).unsqueeze(0)
        batch_tensor[i, 0, :, :rw] = t
        if rw < img_w:
            batch_tensor[i, 0, :, rw:] = t[:, :, rw - 1:rw].expand(1, img_h, img_w - rw)
    return batch_tensor


class ModelEvaluator:
    def __init__(self, model_name: str, checkpoint_path: Path, device: torch.device):
        self.name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        print(f"[{get_utc_timestamp()}] Loading {self.name} from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        sd = {k.replace("module.", ""): v for k, v in ckpt.items()}

        if self.name == "SelenLPR":
            m = SelenLPR(vocab_size=len(SELEN_CTC_IDX_TO_CHAR), time_steps=14).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            return m
        elif self.name == "Custom PaddleOCR":
            from tools.train_paddleocr_thai import CTC_CHARS as PADDLE_CHARS
            m = SelenPaddleOCR(vocab_size=len(PADDLE_CHARS), time_steps=14).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            return m
        elif self.name == "thai-lpr-reader":
            m = VisionAttentionOCR(num_classes=187, batch_max_length=25).to(self.device)
            m.load_state_dict(sd)
            m.eval()
            return m
        else:
            raise ValueError(f"Unknown model name: {self.name}")

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        if self.name == "SelenLPR":
            tensor = prep_selen_paddle_batch(images).to(self.device, non_blocking=True)
            with torch.no_grad():
                preds = self.model(tensor)
                pairs = decode_and_split_ctc(preds["ctc_logits"], SELEN_CTC_IDX_TO_CHAR)
                prov_indices = preds["prov_logits"].argmax(dim=-1).cpu().tolist()
                for (raw_d3, raw_d4), p_idx in zip(pairs, prov_indices):
                    d3, d4 = clamp_d3_d4(raw_d3, raw_d4)
                    prov = SELEN_PROVINCES[p_idx] if p_idx < len(SELEN_PROVINCES) else ""
                    full = f"{d3} {d4} {prov}".strip()
                    results.append({"d3": d3, "d4": d4, "province": prov, "full_plate": full})

        elif self.name == "Custom PaddleOCR":
            tensor = prep_selen_paddle_batch(images).to(self.device, non_blocking=True)
            with torch.no_grad():
                preds = self.model(tensor)
                tuples = decode_and_split_ctc_single_char_prov(preds["ctc_logits"])
                prov_indices = preds["prov_logits"].argmax(dim=-1).cpu().tolist()
                for (raw_d3, raw_d4, ctc_prov), p_idx in zip(tuples, prov_indices):
                    d3, d4 = clamp_d3_d4(raw_d3, raw_d4)
                    prov = ctc_prov if ctc_prov else AUX_IDX_TO_PROV.get(p_idx, "")
                    full = f"{d3} {d4} {prov}".strip()
                    results.append({"d3": d3, "d4": d4, "province": prov, "full_plate": full})

        elif self.name == "thai-lpr-reader":
            tensor = prep_thai_lpr_batch(images).to(self.device, non_blocking=True)
            with torch.no_grad():
                preds = self.model(tensor)
                tokens = preds.argmax(dim=-1).cpu().tolist()
                for tok in tokens:
                    d3, d4, prov, full = decode_tokens_to_plate(tok)
                    results.append({"d3": d3, "d4": d4, "province": prov, "full_plate": full})

        return results


def run_benchmark(metadata_csv: Path, plates_dir: Path, batch_size: int = 128, limit: Optional[int] = None) -> Dict[str, Any]:
    print(f"[{get_utc_timestamp()}] Loading metadata from {metadata_csv}...")
    with open(metadata_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit is not None and limit > 0:
        rows = rows[:limit]
    total_samples = len(rows)
    print(f"[{get_utc_timestamp()}] Dataset contains {total_samples} plates to evaluate.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{get_utc_timestamp()}] Execution device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model_configs = [
        ("SelenLPR", REPO_ROOT / "models/reader/selen-lpr-0.0.1/best_reader.pth"),
        ("Custom PaddleOCR", REPO_ROOT / "models/reader/paddleocr_thai/best_reader.pth"),
        ("thai-lpr-reader", REPO_ROOT / "models/reader/thai_plate_reader.pth"),
    ]

    benchmark_results: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": get_utc_timestamp(),
            "dataset_path": str(plates_dir),
            "total_plates": total_samples,
            "batch_size": batch_size,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
        "models": {}
    }

    # Pre-cache image paths and GT
    img_paths = [str(plates_dir / r["filename"]) for r in rows]
    gt_d3 = [r["d3"] for r in rows]
    gt_d4 = [r["d4"] for r in rows]
    gt_prov = [r["province"] for r in rows]
    gt_full = [r["full_plate"] for r in rows]
    plate_types = [r.get("plate_type", "Unknown") for r in rows]

    num_batches = math.ceil(total_samples / batch_size)

    for model_name, ckpt_path in model_configs:
        print("\n" + "=" * 80)
        print(f"[{get_utc_timestamp()}] STARTING EVALUATION: {model_name}")
        print("=" * 80)

        evaluator = ModelEvaluator(model_name, ckpt_path, device)

        correct_d3 = 0
        correct_d4 = 0
        correct_core = 0
        correct_prov = 0
        correct_full = 0

        cer_d3_sum = 0.0
        cer_d4_sum = 0.0
        cer_full_sum = 0.0

        type_stats: Dict[str, Dict[str, int]] = {}

        t_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=8) as pool:
            for b_idx in range(num_batches):
                start_i = b_idx * batch_size
                end_i = min(start_i + batch_size, total_samples)
                batch_paths = img_paths[start_i:end_i]

                # Load images in parallel
                images = list(pool.map(load_single_image, batch_paths))

                # Run inference
                preds = evaluator.predict_batch(images)

                # Evaluate batch
                for offset, pred in enumerate(preds):
                    idx = start_i + offset
                    g_d3 = gt_d3[idx]
                    g_d4 = gt_d4[idx]
                    g_pr = gt_prov[idx]
                    g_fl = gt_full[idx]
                    pt = plate_types[idx]

                    if pt not in type_stats:
                        type_stats[pt] = {"total": 0, "full_match": 0, "core_match": 0}
                    type_stats[pt]["total"] += 1

                    is_d3 = (pred["d3"] == g_d3)
                    is_d4 = (pred["d4"] == g_d4)
                    is_core = (is_d3 and is_d4)
                    is_pr = (pred["province"] == g_pr)
                    is_full = (is_core and is_pr)

                    if is_d3: correct_d3 += 1
                    if is_d4: correct_d4 += 1
                    if is_core:
                        correct_core += 1
                        type_stats[pt]["core_match"] += 1
                    if is_pr: correct_prov += 1
                    if is_full:
                        correct_full += 1
                        type_stats[pt]["full_match"] += 1

                    cer_d3_sum += cer_score(pred["d3"], g_d3)
                    cer_d4_sum += cer_score(pred["d4"], g_d4)
                    cer_full_sum += cer_score(pred["full_plate"], g_fl)

                if (b_idx + 1) % max(1, (num_batches // 10)) == 0 or (b_idx + 1) == num_batches:
                    curr_done = end_i
                    elapsed = time.perf_counter() - t_start
                    fps = curr_done / max(elapsed, 0.001)
                    print(f"[{get_utc_timestamp()}] {model_name}: {curr_done}/{total_samples} ({(curr_done/total_samples)*100:.1f}%) | "
                          f"FPS: {fps:.1f} | Full Match: {(correct_full/curr_done)*100:.2f}% | Core: {(correct_core/curr_done)*100:.2f}%")

        total_time = time.perf_counter() - t_start
        fps = total_samples / max(total_time, 0.001)
        latency_ms = (total_time / total_samples) * 1000.0

        model_metrics = {
            "checkpoint": str(ckpt_path),
            "total_evaluated": total_samples,
            "total_time_seconds": round(total_time, 2),
            "latency_ms_per_plate": round(latency_ms, 2),
            "throughput_fps": round(fps, 1),
            "accuracy": {
                "d3_acc_percent": round((correct_d3 / total_samples) * 100.0, 2),
                "d4_acc_percent": round((correct_d4 / total_samples) * 100.0, 2),
                "core_registration_acc_percent": round((correct_core / total_samples) * 100.0, 2),
                "province_acc_percent": round((correct_prov / total_samples) * 100.0, 2),
                "full_plate_exact_acc_percent": round((correct_full / total_samples) * 100.0, 2),
            },
            "exact_counts": {
                "d3_correct": correct_d3,
                "d4_correct": correct_d4,
                "core_correct": correct_core,
                "province_correct": correct_prov,
                "full_plate_correct": correct_full,
            },
            "cer": {
                "d3_cer": round(cer_d3_sum / total_samples, 4),
                "d4_cer": round(cer_d4_sum / total_samples, 4),
                "full_plate_cer": round(cer_full_sum / total_samples, 4),
            },
            "plate_type_breakdown": {
                pt: {
                    "total": st["total"],
                    "full_match_percent": round((st["full_match"] / st["total"]) * 100.0, 2) if st["total"] else 0.0,
                    "core_match_percent": round((st["core_match"] / st["total"]) * 100.0, 2) if st["total"] else 0.0,
                }
                for pt, st in type_stats.items()
            }
        }

        benchmark_results["models"][model_name] = model_metrics

        # Clean up GPU memory before next model
        del evaluator
        torch.cuda.empty_cache()

    return benchmark_results


def save_summary_markdown(results: Dict[str, Any], output_path: Path):
    meta = results["metadata"]
    models = results["models"]

    md = []
    md.append("# Thai License Plate Recognition Benchmark Report")
    md.append(f"**Date (UTC):** {meta['timestamp_utc']}  ")
    md.append(f"**Dataset:** `{meta['dataset_path']}`  ")
    md.append(f"**Total Plates Evaluated:** {meta['total_plates']:,}  ")
    md.append(f"**Hardware:** {meta['gpu_name']} (Batch Size: {meta['batch_size']})  \n")
    md.append("---")
    md.append("## 1. Overall Accuracy Comparison\n")
    md.append("| Model | d3 Prefix Acc | d4 Number Acc | Core (d3+d4) Acc | Province Acc | Full Plate Match | Full CER | Latency (ms) | FPS |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, m in models.items():
        acc = m["accuracy"]
        cer = m["cer"]
        md.append(f"| **{name}** | {acc['d3_acc_percent']:.2f}% | {acc['d4_acc_percent']:.2f}% | **{acc['core_registration_acc_percent']:.2f}%** | {acc['province_acc_percent']:.2f}% | **{acc['full_plate_exact_acc_percent']:.2f}%** | {cer['full_plate_cer']:.4f} | {m['latency_ms_per_plate']:.2f} ms | {m['throughput_fps']:.1f} |")

    md.append("\n---\n")
    md.append("## 2. Plate Type Breakdown (Full Plate Exact Match)\n")

    # Collect all plate types
    all_types = sorted(list(next(iter(models.values()))["plate_type_breakdown"].keys()))
    header = "| Plate Type | " + " | ".join([f"**{m}**" for m in models.keys()]) + " |"
    md.append(header)
    md.append("| :--- | " + " | ".join([":---:" for _ in models.keys()]) + " |")

    for pt in all_types:
        row = [f"`{pt}`"]
        for m_name in models.keys():
            st = models[m_name]["plate_type_breakdown"].get(pt, {})
            row.append(f"{st.get('full_match_percent', 0.0):.2f}% ({st.get('total', 0)})")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n---\n")
    md.append("## 3. ISO/IEC 42001:2023 Verification Summary\n")
    md.append("- **Annex A.7 (Data Quality):** Dataset `dataset-dirty10-blur5-angle5` evaluated across standard DLT plate specifications (10% dirt, 5% blur, ±5° angle).")
    md.append("- **Annex A.6.7 (AI Change & Model Governance):** Comparative lineage verification between SelenLPR, Custom PaddleOCR, and production baseline thai-lpr-reader.")
    md.append("- **Verification Outcome:** Full lineage captured in `runs/val/benchmark_dirty10_blur5_angle5.json`.")

    content = "\n".join(md)
    output_path.write_text(content, encoding="utf-8")
    print(f"[{get_utc_timestamp()}] Summary Markdown saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Thai LPR 3-Model Validation Benchmark on dirty10-blur5-angle5")
    parser.add_argument("--metadata", type=str, default="dataset/reading/versions/dataset-dirty10-blur5-angle5/metadata.csv")
    parser.add_argument("--plates_dir", type=str, default="dataset/reading/versions/dataset-dirty10-blur5-angle5/plates")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke test")
    parser.add_argument("--output_json", type=str, default="runs/val/benchmark_dirty10_blur5_angle5.json")
    parser.add_argument("--output_summary", type=str, default="runs/val/benchmark_dirty10_blur5_angle5_summary.md")
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    plates_path = Path(args.plates_dir)
    if not meta_path.is_absolute():
        meta_path = REPO_ROOT / meta_path
    if not plates_path.is_absolute():
        plates_path = REPO_ROOT / plates_path

    results = run_benchmark(meta_path, plates_path, batch_size=args.batch_size, limit=args.limit)

    # Save JSON
    out_json = REPO_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[{get_utc_timestamp()}] Results JSON written to {out_json}")

    # Save Markdown
    out_md = REPO_ROOT / args.output_summary
    save_summary_markdown(results, out_md)


if __name__ == "__main__":
    main()
