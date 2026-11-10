#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR / MobileNetV3 Baseline Training Pipeline on Dataset v3.0 (Synthetic Clean)
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Representations)

Architecture:
- MobileNetV3 Large Backbone (with adapted horizontal strides for sequence preservation)
- BiLSTM Sequence Modeling (2-Layer Bidirectional, 128 hidden size)
- Linear CTC Head (145 classes, PAD_ID=0 blank)
- Single text line output: "plateType d3 d4 province" (max 11 tokens)
- High-throughput in-memory caching and GPU batch normalization for ultra-fast convergence.
"""

import os
import sys
import time
import json
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
    VOCABULARY,
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    encode_plate_to_tokens,
    decode_tokens_to_plate
)


class PaddleOCRBaselineModel(nn.Module):
    """
    PaddleOCR PP-OCR style architecture:
    MobileNetV3 Large Backbone + BiLSTM Neck + CTC Head.
    Preserves horizontal spatial resolution (W=40 sequence steps) to accommodate
    up to 11 tokens with CTC blank separations.
    """
    def __init__(self, d_model: int = 256, hidden_size: int = 128, num_class: int = VOCAB_SIZE):
        super().__init__()
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # Adapt horizontal strides at Stage 3 and Stage 4 to preserve sequence length for CTC alignment
        for m in mobilenet.features[7].modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (2, 1)
        for m in mobilenet.features[13].modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (2, 1)

        self.backbone = mobilenet.features  # Out: [B, 960, 2, 40] for input [B, 3, 48, 320]
        self.proj = nn.Conv2d(960, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

        self.neck = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size * 2, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W]
        feat = self.proj(self.backbone(x)) # [B, d_model, H', W'=40]
        feat = feat.mean(dim=2).permute(0, 2, 1) # [B, W'=40, d_model]
        feat = self.norm(feat)
        seq_out, _ = self.neck(feat) # [B, W', hidden_size * 2]
        logits = self.classifier(seq_out) # [B, W', num_class]
        return logits


class PaddleDatasetV3:
    """Dataset wrapper for compatibility with benchmark suites."""
    def __init__(self, samples: List[Dict[str, Any]], img_h: int = 48, img_w: int = 320):
        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.samples)


def preload_dataset_to_ram(
    records: List[Dict[str, Any]],
    img_h: int = 48,
    img_w: int = 320,
    max_workers: int = 16
) -> Tuple[np.ndarray, List[List[int]], List[str]]:
    """
    High-speed parallel ingestion of image dataset into a compact RAM buffer (uint8 RGB).
    """
    total = len(records)
    print(f"Preloading {total:,} images into system RAM with {max_workers} threads...")
    t0 = time.time()

    img_buffer = np.full((total, img_h, img_w, 3), 128, dtype=np.uint8)
    token_list: List[List[int]] = [None] * total
    line_list: List[str] = [None] * total

    chunk_size = 1000

    def worker_chunk(start_idx: int, end_idx: int):
        for idx in range(start_idx, end_idx):
            rec = records[idx]
            p = rec["filepath"]
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                ratio = w / float(h)
                rw = min(img_w, max(16, int(img_h * ratio)))
                r_img = cv2.resize(img, (rw, img_h), interpolation=cv2.INTER_LINEAR)
                img_buffer[idx, :, :rw] = cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB)
            tokens = json.loads(rec["token_ids"]) if isinstance(rec["token_ids"], str) else rec["token_ids"]
            token_list[idx] = tokens
            line_list[idx] = rec["full_text_line"]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(worker_chunk, i, min(i + chunk_size, total))
            for i in range(0, total, chunk_size)
        ]
        for f in futures:
            f.result()

    elapsed = time.time() - t0
    mem_gb = img_buffer.nbytes / (1024 ** 3)
    print(f"Preloaded {total:,} images in {elapsed:.2f}s ({total/elapsed:.1f} imgs/s) | RAM: {mem_gb:.2f} GB")
    return img_buffer, token_list, line_list


def train_paddleocr_baseline(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 1e-3,
    target_acc: float = 95.0,
    device: str = "cuda"
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir = ROOT_DIR / "models" / "reader" / "paddleocr_v3"
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    import csv
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    random.seed(42)
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.95)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    print(f"PaddleOCR Train: {len(train_records):,} | Val: {len(val_records):,}")

    # Ingest datasets into RAM buffers
    train_imgs, train_tokens, train_lines = preload_dataset_to_ram(train_records)
    val_imgs, val_tokens, val_lines = preload_dataset_to_ram(val_records)

    model = PaddleOCRBaselineModel(d_model=256, hidden_size=128, num_class=VOCAB_SIZE)
    model = model.to(device)

    criterion = nn.CTCLoss(blank=PAD_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ImageNet normalization tensors on GPU
    gpu_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    gpu_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    best_val_acc = 0.0
    history = []
    start_epoch = 1

    # Resume from checkpoint if present
    best_path = chkpt_dir / "best_reader.pth"
    if best_path.exists():
        try:
            chk = torch.load(best_path, map_location=device)
            model.load_state_dict(chk["model_state_dict"])
            optimizer.load_state_dict(chk["optimizer_state_dict"])
            best_val_acc = float(chk.get("val_acc", 0.0))
            saved_epoch = int(chk.get("epoch", 0))
            start_epoch = saved_epoch + 1
            # Step scheduler to match saved epoch
            for _ in range(saved_epoch):
                scheduler.step()
            print(f"[RESUME] Loaded best checkpoint from {best_path} (Epoch {saved_epoch}, Val Acc: {best_val_acc:.2f}%)")
        except Exception as e:
            print(f"[WARN] Could not resume from {best_path}: {e}")

    rec_path = out_dir / "training_record.json"
    if rec_path.exists():
        try:
            with open(rec_path, "r", encoding="utf-8") as f:
                rec_data = json.load(f)
                history = rec_data.get("history", [])
        except Exception:
            pass

    num_train = len(train_records)
    num_val = len(val_records)

    print("=" * 80)
    print("  TRAINING PADDLEOCR / MOBILENETV3 BASELINE - DATASET v3.0")
    print(f"  Vocabulary: {VOCAB_SIZE} tokens | Device: {device} | Mixed Precision FP16")
    print(f"  Batch Size: {batch_size} | LR: {lr} | Target Acc: {target_acc}%")
    print("=" * 80)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        # Shuffle training indices
        indices = np.random.permutation(num_train)

        for step in range(0, num_train, batch_size):
            batch_idx = indices[step:min(step + batch_size, num_train)]
            b_size = len(batch_idx)

            # Batch images
            batch_u8 = torch.from_numpy(train_imgs[batch_idx])
            batch_gpu = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
            imgs = (batch_gpu - gpu_mean) / gpu_std

            # Targets & target lengths
            batch_toks = [train_tokens[i] for i in batch_idx]
            max_len = max(len(t) for t in batch_toks)
            targets = torch.full((b_size, max_len), PAD_ID, dtype=torch.long, device=device)
            target_lens = torch.zeros(b_size, dtype=torch.long, device=device)
            for j, t in enumerate(batch_toks):
                targets[j, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                target_lens[j] = len(t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                preds = model(imgs) # [B, T=40, Vocab]
                log_probs = preds.log_softmax(2).permute(1, 0, 2) # [T, B, Vocab]
                input_lengths = torch.full((b_size,), preds.size(1), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * b_size

        train_loss /= num_train
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        exact_matches = 0

        with torch.no_grad():
            for v_step in range(0, num_val, batch_size):
                v_end = min(v_step + batch_size, num_val)
                b_size = v_end - v_step

                batch_u8 = torch.from_numpy(val_imgs[v_step:v_end])
                batch_gpu = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
                imgs = (batch_gpu - gpu_mean) / gpu_std

                batch_toks = val_tokens[v_step:v_end]
                max_len = max(len(t) for t in batch_toks)
                targets = torch.full((b_size, max_len), PAD_ID, dtype=torch.long, device=device)
                target_lens = torch.zeros(b_size, dtype=torch.long, device=device)
                for j, t in enumerate(batch_toks):
                    targets[j, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
                    target_lens[j] = len(t)

                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
                    log_probs = preds.log_softmax(2).permute(1, 0, 2)
                    input_lengths = torch.full((b_size,), preds.size(1), dtype=torch.long, device=device)
                    loss = criterion(log_probs, targets, input_lengths, target_lens)

                val_loss += loss.item() * b_size

                pred_ids = preds.argmax(dim=-1).cpu().tolist()
                for i, seq in enumerate(pred_ids):
                    collapsed = [k for j, k in enumerate(seq) if k != PAD_ID and (j == 0 or k != seq[j - 1])]
                    _, _, _, _, pred_line = decode_tokens_to_plate(collapsed)
                    gt_line = val_lines[v_step + i]
                    if pred_line.strip() == gt_line.strip():
                        exact_matches += 1

        val_loss /= num_val
        val_acc = (exact_matches / num_val) * 100.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Exact Acc: {val_acc:5.2f}% ({exact_matches}/{num_val}) | "
              f"Time: {elapsed:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = chkpt_dir / "best_reader.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "vocab_size": VOCAB_SIZE,
                "architecture": "PaddleOCR-MobileNetV3-BiLSTM-CTC"
            }, best_path)
            print(f"  --> Saved new best PaddleOCR checkpoint: {best_path} (Acc: {val_acc:.2f}%)")

        rec_path = out_dir / "training_record.json"
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": "PaddleOCR Baseline v3",
                "best_val_acc": best_val_acc,
                "history": history,
                "last_epoch": epoch,
                "updated_at_utc": datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

        if best_val_acc >= target_acc:
            print(f"\n[SUCCESS] Achieved target accuracy: {best_val_acc:.2f}% >= {target_acc}% at epoch {epoch}!")
            break

    print(f"\nTraining completed. Best Val Exact Acc: {best_val_acc:.2f}%")
    return {"best_val_acc": best_val_acc, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.0-synthetic-clean"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "runs" / "train" / "paddleocr_v3"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--target-acc", type=float, default=95.0)
    args = parser.parse_args()

    train_paddleocr_baseline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_acc=args.target_acc
    )
