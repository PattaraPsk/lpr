#!/usr/bin/env python3
"""
SELEN-LPR: OCR Standalone Model Evaluation Suite.
Evaluates pure PyTorch Vision-Attention OCR against authentic Thai labeled datasets.
Computes:
1. Full Plate Exact Match Accuracy (%)
2. Category Prefix (d3) Exact Match Accuracy (%)
3. Plate Digits (d4) Exact Match Accuracy (%)
4. Province Exact Match Accuracy (%)
5. Normalized Edit Distance (NED)
6. Throughput (FPS) and Average Latency (ms)
"""

import os
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.vision_attention_ocr import VisionAttentionOCR
from src.models.vocab import EASYOCR_DECODING_LIST, decode_raw_prediction, REMAP_DICT, CHAR_LIST


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_edit_distance(pred: str, gt: str) -> float:
    """ICDAR 2019 standard Normalized Edit Distance."""
    if len(gt) == 0 and len(pred) == 0:
        return 1.0
    if len(gt) == 0 or len(pred) == 0:
        return 0.0
    dist = levenshtein_distance(pred, gt)
    return max(0.0, 1.0 - (dist / max(len(pred), len(gt))))


class OCREvaluator:
    def __init__(
        self,
        weights_path: str = str(ROOT_DIR / "models" / "ocr" / "thai_plate_attn_ocr.pt"),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        img_h: int = 64,
        img_w: int = 600
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.img_h = img_h
        self.img_w = img_w
        print(f"[INFO] Initializing OCREvaluator on device: {self.device}")
        print(f"[INFO] Loading OCR weights from: {weights_path}")

        self.model = VisionAttentionOCR(num_classes=187, batch_max_length=25)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.to_tensor = transforms.ToTensor()

    def preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """Authentic NormalizePAD transform."""
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        h, w = gray.shape
        ratio = w / float(max(h, 1))
        resized_w = min(int(np.ceil(self.img_h * ratio)), self.img_w)
        pil_img = Image.fromarray(gray, "L")
        resized_crop = pil_img.resize((resized_w, self.img_h), Image.BICUBIC)

        t = self.to_tensor(resized_crop)
        t.sub_(0.5).div_(0.5)

        pad_tensor = torch.zeros(1, self.img_h, self.img_w, dtype=torch.float32)
        pad_tensor[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad_tensor[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(1, self.img_h, self.img_w - resized_w)
        return pad_tensor

    def evaluate_batch(self, tensors: torch.Tensor) -> List[Tuple[str, str, str, str]]:
        """Batched forward pass + decoding."""
        tensors = tensors.to(self.device)
        with torch.no_grad():
            preds = self.model(tensors)
            indices = preds.argmax(dim=-1).cpu().numpy().tolist()

        results = []
        for ind in indices:
            d3, d4, prov, full = decode_raw_prediction(ind)
            results.append((d3, d4, prov, full))
        return results

    def run_evaluation(
        self,
        dataset_csv: str,
        image_dir: str,
        max_samples: int = 0,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Runs evaluation on labeled dataset and returns comprehensive statistics.
        Supports both test_decoded.csv format and raw labels.csv format.
        """
        print("=" * 75)
        print(f"  SELEN-LPR: EVALUATING PURE PYTORCH OCR MODEL")
        print("=" * 75)
        print(f"Dataset CSV   : {dataset_csv}")
        print(f"Image Directory: {image_dir}")
        print(f"Batch Size    : {batch_size}")
        print(f"Max Samples   : {'ALL' if max_samples <= 0 else max_samples}")
        print("=" * 75)

        with open(dataset_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]
            all_rows = list(reader)

        if max_samples > 0:
            all_rows = all_rows[:max_samples]

        total_samples = len(all_rows)
        print(f"[INFO] Loaded {total_samples} labeled ground-truth records.")

        is_decoded_format = "3DIGIT" in fieldnames or "charactor" in fieldnames
        is_raw_format = "words" in fieldnames

        # Metrics accumulators
        correct_full = 0
        correct_d3 = 0
        correct_d4 = 0
        correct_prov = 0
        ned_scores = []
        char_levenshtein = []

        total_infer_time = 0.0
        processed_count = 0
        missing_count = 0
        errors_sample: List[Dict[str, Any]] = []

        # Batch loop
        for b_idx in range(0, total_samples, batch_size):
            batch_slice = all_rows[b_idx : b_idx + batch_size]
            batch_tensors = []
            batch_meta = []

            for row in batch_slice:
                if is_decoded_format:
                    fname = row.get("IMAGE_NAME") or row.get("filename")
                    gt_d3 = str(row.get("3DIGIT") or row.get("charactor") or "").strip()
                    gt_d4 = str(row.get("4DIGIT") or row.get("digits") or "").strip()
                    gt_prov = str(row.get("PROVINCE") or row.get("province") or "").strip()
                    gt_full = f"{gt_d3} {gt_d4} {gt_prov}".strip()
                elif is_raw_format:
                    fname = row.get("filename")
                    raw_words = row.get("words", "")
                    # Raw format is like "g>_4293_¢"
                    parts = raw_words.split("_")
                    if len(parts) >= 3:
                        raw_d3, raw_d4, raw_p = parts[0], parts[1], parts[2]
                        gt_d3 = "".join([REMAP_DICT.get(c, c) for c in raw_d3])
                        gt_d4 = raw_d4
                        gt_prov = REMAP_DICT.get(raw_p, raw_p)
                    else:
                        gt_d3, gt_d4, gt_prov = "", "", ""
                    gt_full = f"{gt_d3} {gt_d4} {gt_prov}".strip()
                else:
                    fname = list(row.values())[0]
                    gt_full, gt_d3, gt_d4, gt_prov = "", "", "", ""

                img_path = Path(image_dir) / fname
                if not img_path.exists():
                    missing_count += 1
                    continue

                try:
                    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                if img is None:
                    missing_count += 1
                    continue

                t = self.preprocess(img)
                batch_tensors.append(t)
                batch_meta.append({
                    "filename": fname,
                    "gt_d3": gt_d3,
                    "gt_d4": gt_d4,
                    "gt_prov": gt_prov,
                    "gt_full": gt_full
                })

            if not batch_tensors:
                continue

            t_batch = torch.stack(batch_tensors, dim=0)

            t0 = time.time()
            predictions = self.evaluate_batch(t_batch)
            infer_dur = time.time() - t0
            total_infer_time += infer_dur

            for meta, pred in zip(batch_meta, predictions):
                pred_d3, pred_d4, pred_prov, pred_full = pred
                gt_d3 = meta["gt_d3"]
                gt_d4 = meta["gt_d4"]
                gt_prov = meta["gt_prov"]
                gt_full = meta["gt_full"]

                match_d3 = (pred_d3 == gt_d3)
                match_d4 = (pred_d4 == gt_d4)
                match_prov = (pred_prov == gt_prov)
                match_full = (pred_full == gt_full)

                if match_d3:
                    correct_d3 += 1
                if match_d4:
                    correct_d4 += 1
                if match_prov:
                    correct_prov += 1
                if match_full:
                    correct_full += 1

                ned = normalized_edit_distance(pred_full, gt_full)
                ned_scores.append(ned)

                dist = levenshtein_distance(pred_full, gt_full)
                char_levenshtein.append(dist)

                processed_count += 1

                if not match_full and len(errors_sample) < 10:
                    errors_sample.append({
                        "file": meta["filename"],
                        "gt": gt_full,
                        "pred": pred_full,
                        "d3_match": match_d3,
                        "d4_match": match_d4,
                        "prov_match": match_prov
                    })

            if (b_idx + batch_size) % 160 == 0 or b_idx + batch_size >= total_samples:
                cur_acc = (correct_full / max(processed_count, 1)) * 100
                print(f"  Processed {processed_count}/{total_samples} samples | Batch Acc: {cur_acc:.2f}% | Latency: {(infer_dur/len(batch_meta))*1000:.1f}ms/img")

        # Summary calculations
        full_acc = (correct_full / max(processed_count, 1)) * 100
        d3_acc = (correct_d3 / max(processed_count, 1)) * 100
        d4_acc = (correct_d4 / max(processed_count, 1)) * 100
        prov_acc = (correct_prov / max(processed_count, 1)) * 100
        avg_ned = (sum(ned_scores) / max(len(ned_scores), 1)) * 100
        fps = processed_count / max(total_infer_time, 0.001)
        avg_latency = (total_infer_time / max(processed_count, 1)) * 1000

        print("\n" + "=" * 75)
        print("          SELEN-LPR: OCR BENCHMARK EVALUATION RESULTS")
        print("=" * 75)
        print(f"Total Evaluated Images   : {processed_count} (Missing/Skipped: {missing_count})")
        print(f"Exact Full Match Acc     : {full_acc:.2f}% ({correct_full}/{processed_count})")
        print(f"Category Prefix (d3) Acc : {d3_acc:.2f}% ({correct_d3}/{processed_count})")
        print(f"Plate Digits (d4) Acc    : {d4_acc:.2f}% ({correct_d4}/{processed_count})")
        print(f"Province Name Acc        : {prov_acc:.2f}% ({correct_prov}/{processed_count})")
        print(f"Normalized Edit Dist(NED): {avg_ned:.2f}%")
        print(f"Pure PyTorch Throughput  : {fps:.1f} FPS ({avg_latency:.2f} ms/image)")
        print("=" * 75)

        if errors_sample:
            print("\n[SAMPLE ERROR ANALYSIS]")
            for err in errors_sample[:5]:
                print(f"  * {err['file']}")
                print(f"    GT  : {err['gt']}")
                print(f"    Pred: {err['pred']}")
                print(f"    Detail: d3={err['d3_match']}, d4={err['d4_match']}, prov={err['prov_match']}")

        return {
            "processed": processed_count,
            "full_acc": full_acc,
            "d3_acc": d3_acc,
            "d4_acc": d4_acc,
            "prov_acc": prov_acc,
            "avg_ned": avg_ned,
            "fps": fps,
            "avg_latency_ms": avg_latency,
            "errors": errors_sample
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate pure PyTorch OCR on labeled dataset.")
    parser.add_argument("--csv", type=str,
                        default=r"D:\data\LicensePlateRecognition\all_data\valid\250106\1v_v\test_decoded.csv",
                        help="Path to dataset CSV file.")
    parser.add_argument("--img-dir", type=str,
                        default=r"D:\data\LicensePlateRecognition\all_data\valid\250106\1v_v",
                        help="Path to folder containing image crops.")
    parser.add_argument("--weights", type=str,
                        default=str(ROOT_DIR / "models" / "ocr" / "thai_plate_attn_ocr.pt"),
                        help="OCR model weights.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit evaluation samples (0 for all).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Inference batch size.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device ('cuda' or 'cpu').")
    args = parser.parse_args()

    evaluator = OCREvaluator(weights_path=args.weights, device=args.device)
    evaluator.run_evaluation(
        dataset_csv=args.csv,
        image_dir=args.img_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
