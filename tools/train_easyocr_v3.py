#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyOCR Baseline Training Pipeline on Dataset v3.0 (Synthetic Clean)
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Representations)

Architecture:
- ResNet Feature Extractor + 2-Layer BiLSTM Sequence Modeling + CTC Linear Head
- Single text line output: "plateType d3 d4 province"
- Max tokens: start + plateType + 3 + 4 + 1 + stop = 11 tokens
"""

import os
import sys
import time
import json
import math
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
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

from easyocr_modules import ResNet_FeatureExtractor, BidirectionalLSTM
from tools.selen_token_codec import (
    VOCABULARY,
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    encode_plate_to_tokens,
    decode_tokens_to_plate
)


class EasyOCRModelV3(nn.Module):
    """
    JaidedAI EasyOCR CRNN architecture adapted for 11-token Thai LPR output.
    """
    def __init__(self, input_channel: int = 1, output_channel: int = 512, hidden_size: int = 256, num_class: int = VOCAB_SIZE):
        super().__init__()
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 64, W]
        feat = self.FeatureExtraction(x)
        feat = self.AdaptiveAvgPool(feat.permute(0, 3, 1, 2)).squeeze(3) # [B, W', C]
        context = self.SequenceModeling(feat) # [B, W', hidden_size]
        logits = self.Prediction(context) # [B, W', num_class]
        return logits


class ThaiPlateDatasetV3(Dataset):
    """Dataset loader reading from metadata.csv or plates folder."""
    def __init__(self, samples: List[Dict[str, Any]], img_h: int = 64, img_w: int = 600):
        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        record = self.samples[idx]
        img_path = record["filepath"]
        
        # Load grayscale
        img = Image.open(img_path).convert("L")
        
        # NormalizePAD: scale preserving aspect ratio, pad to (img_h, img_w)
        w, h = img.size
        ratio = w / float(h)
        if math.ceil(self.img_h * ratio) > self.img_w:
            resized_w = self.img_w
        else:
            resized_w = math.ceil(self.img_h * ratio)
        resized_img = img.resize((resized_w, self.img_h), Image.Resampling.BILINEAR)

        # Pad to img_w with 128 (neutral gray)
        pad_img = Image.new("L", (self.img_w, self.img_h), 128)
        pad_img.paste(resized_img, (0, 0))

        img_arr = np.array(pad_img, dtype=np.float32) / 255.0
        img_arr = (img_arr - 0.5) / 0.5  # Normalize to [-1, 1]
        tensor = torch.from_numpy(img_arr).unsqueeze(0) # [1, H, W]

        # Target token IDs
        token_ids = json.loads(record["token_ids"]) if isinstance(record["token_ids"], str) else record["token_ids"]
        target = torch.tensor(token_ids, dtype=torch.long)
        target_len = len(token_ids)

        return tensor, target, target_len, record


def collate_fn(batch):
    images, targets, target_lengths, records = zip(*batch)
    images = torch.stack(images, dim=0)

    max_len = max(target_lengths)
    padded_targets = torch.full((len(targets), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, padded_targets, target_lengths, records


def train_easyocr_baseline(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 5e-4,
    device: str = "cuda"
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir = ROOT_DIR / "models" / "reader" / "easyocr_v3"
    chkpt_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    import csv
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_records = list(reader)

    print(f"Loaded {len(all_records):,} total samples from {metadata_path}")

    # 95 / 5 train / val split
    random.seed(42)
    random.shuffle(all_records)
    split_idx = int(len(all_records) * 0.95)
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]
    print(f"Train samples: {len(train_records):,} | Val samples: {len(val_records):,}")

    train_ds = ThaiPlateDatasetV3(train_records)
    val_ds = ThaiPlateDatasetV3(val_records)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn, pin_memory=True
    )

    model = EasyOCRModelV3(input_channel=1, output_channel=512, hidden_size=256, num_class=VOCAB_SIZE)
    model = model.to(device)

    criterion = nn.CTCLoss(blank=PAD_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0
    history = []

    print("=" * 80)
    print("  TRAINING EASYOCR BASELINE (CRNN) - DATASET v3.0")
    print(f"  Vocabulary: {VOCAB_SIZE} tokens | Device: {device} | Mixed Precision FP16")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, targets, target_lens, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                preds = model(imgs) # [B, T, Vocab]
                log_probs = preds.log_softmax(2).permute(1, 0, 2) # [T, B, Vocab]
                input_lengths = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lens)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        exact_matches = 0
        total_val = len(val_ds)

        with torch.no_grad():
            for imgs, targets, target_lens, records in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lens = target_lens.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
                    log_probs = preds.log_softmax(2).permute(1, 0, 2)
                    input_lengths = torch.full((imgs.size(0),), preds.size(1), dtype=torch.long, device=device)
                    loss = criterion(log_probs, targets, input_lengths, target_lens)

                val_loss += loss.item() * imgs.size(0)

                # Greedy decode
                pred_ids = preds.argmax(dim=-1).cpu().tolist()
                for i, seq in enumerate(pred_ids):
                    collapsed = [k for j, k in enumerate(seq) if k != PAD_ID and (j == 0 or k != seq[j - 1])]
                    _, _, _, _, pred_line = decode_tokens_to_plate(collapsed)
                    gt_line = records[i]["full_text_line"]
                    if pred_line.strip() == gt_line.strip():
                        exact_matches += 1

        val_loss /= total_val
        val_acc = (exact_matches / total_val) * 100.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Exact Acc: {val_acc:5.2f}% ({exact_matches}/{total_val}) | "
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
                "architecture": "EasyOCR-ResNet-BiLSTM-CTC"
            }, best_path)
            print(f"  --> Saved new best EasyOCR checkpoint: {best_path} (Acc: {val_acc:.2f}%)")

    # Save training record
    rec_path = out_dir / "training_record.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "EasyOCR Baseline v3",
            "best_val_acc": best_val_acc,
            "history": history,
            "completed_at_utc": datetime.now(timezone.utc).isoformat()
        }, f, indent=2)

    return {"best_val_acc": best_val_acc, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-v3.0-synthetic-clean"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "runs" / "train" / "easyocr_v3"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    train_easyocr_baseline(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
