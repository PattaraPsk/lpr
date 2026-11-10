#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Pipeline for SelenOCR v0.1.1 (Combined Dataset v3.0 + Dataset v3.1 CCTV Add-on)
Warm-starts from SelenOCR v0.1.0 checkpoint to rapidly adapt to:
- Low-resolution downscaling (20px - 48px)
- Morphological stroke variations (dilation/erosion)
- CCTV sensor noise & low-bitrate compression
- Real CCTV anchor plates (202610261140)

Complies with ISO/IEC 42001:2023 (Annex A.6.7 AI System Change Management)
"""

import os
import sys
import time
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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

from tools.selen_token_codec import (
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    PLATE_TYPES,
    PROVINCES_78,
    decode_tokens_to_plate
)
from tools.selen_ocr_model import SelenOCR

TYPE_TO_HEAD_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
TYPE_TO_HEAD_IDX["+"] = len(TYPE_TO_HEAD_IDX)
TYPE_TO_HEAD_IDX["-"] = len(TYPE_TO_HEAD_IDX)

PROV_TO_HEAD_IDX = {p: i for i, p in enumerate(PROVINCES_78)}
PROV_TO_HEAD_IDX["<unknown>"] = len(PROV_TO_HEAD_IDX)
PROV_TO_HEAD_IDX["+"] = len(PROV_TO_HEAD_IDX)
PROV_TO_HEAD_IDX["-"] = len(PROV_TO_HEAD_IDX)


def load_combined_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads Dataset v3.0 (120k) and Dataset v3.1 (21k) and creates a balanced train/val split."""
    v30_path = ROOT_DIR / "dataset/reading/versions/dataset-v3.0-synthetic-clean/metadata.csv"
    v31_path = ROOT_DIR / "dataset/reading/versions/dataset-v3.1-cctv-addon/metadata.csv"

    records_v30 = []
    if v30_path.exists():
        with open(v30_path, "r", encoding="utf-8") as f:
            records_v30 = list(csv.DictReader(f))

    records_v31 = []
    if v31_path.exists():
        with open(v31_path, "r", encoding="utf-8") as f:
            records_v31 = list(csv.DictReader(f))

    print(f"Loaded records: Dataset v3.0 = {len(records_v30):,} | Dataset v3.1 = {len(records_v31):,}")

    # Shuffle v3.0 and v3.1 independently to ensure balanced splits
    random.seed(42)
    random.shuffle(records_v30)
    random.shuffle(records_v31)

    # 95% train / 5% val split from each dataset
    split_30 = int(len(records_v30) * 0.95)
    split_31 = int(len(records_v31) * 0.95)

    train_records = records_v30[:split_30] + records_v31[:split_31]
    val_records = records_v30[split_30:] + records_v31[split_31:]

    random.shuffle(train_records)
    random.shuffle(val_records)

    print(f"Combined Training Set: {len(train_records):,} | Combined Validation Set: {len(val_records):,}")
    return train_records, val_records


class SelenOCRDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], img_h: int = 64, img_w: int = 320):
        self.records = records
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, Dict[str, Any]]:
        record = self.records[idx]
        img_path = record["filepath"]

        try:
            with open(img_path, "rb") as f:
                buf = np.frombuffer(f.read(), dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
            else:
                rgb = np.full((self.img_h, self.img_w, 3), 128, dtype=np.uint8)
        except Exception:
            rgb = np.full((self.img_h, self.img_w, 3), 128, dtype=np.uint8)

        img_arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_arr = (img_arr - mean) / std
        tensor = torch.from_numpy(img_arr.transpose(2, 0, 1))

        token_ids = json.loads(record["token_ids"]) if isinstance(record["token_ids"], str) else record["token_ids"]
        target = torch.tensor(token_ids, dtype=torch.long)
        target_len = len(token_ids)

        p_type = record["plate_type"]
        type_idx = TYPE_TO_HEAD_IDX.get(p_type, 0)

        prov = record["province"]
        prov_idx = PROV_TO_HEAD_IDX.get(prov, PROV_TO_HEAD_IDX["<unknown>"])

        return tensor, target, target_len, type_idx, prov_idx, record


def collate_fn(batch):
    images, targets, target_lengths, type_indices, prov_indices, records = zip(*batch)
    images = torch.stack(images, dim=0)

    max_len = max(target_lengths)
    padded_targets = torch.full((len(targets), max_len), PAD_ID, dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t

    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    type_targets = torch.tensor(type_indices, dtype=torch.long)
    prov_targets = torch.tensor(prov_indices, dtype=torch.long)

    return images, padded_targets, target_lengths, type_targets, prov_targets, records


def train_selen_ocr_v011(
    output_dir: Path = ROOT_DIR / "models" / "reader" / "selen_ocr_v011",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 2.5e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    warm_start_chkpt: Optional[Path] = ROOT_DIR / "models/reader/selen_ocr_v010/best_reader.pth"
):
    output_dir.mkdir(parents=True, exist_ok=True)
    chkpt_dir = output_dir

    train_records, val_records = load_combined_records()

    train_ds = SelenOCRDataset(train_records)
    val_ds = SelenOCRDataset(val_records)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=8,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=8,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True
    )

    model = SelenOCR(
        d_model=384,
        nhead=6,
        num_encoder_layers=4,
        lstm_hidden=256,
        vocab_size=VOCAB_SIZE,
        num_plate_types=len(TYPE_TO_HEAD_IDX),
        num_provinces=len(PROV_TO_HEAD_IDX)
    )
    model = model.to(device)

    # Warm-start weights from SelenOCR v0.1.0
    if warm_start_chkpt and Path(warm_start_chkpt).exists():
        print(f"Warm-starting SelenOCR v0.1.1 from converged v0.1.0 checkpoint: {warm_start_chkpt}")
        chkpt = torch.load(warm_start_chkpt, map_location=device)
        model.load_state_dict(chkpt["model_state_dict"], strict=True)
        print("Successfully loaded pre-trained backbone, FPN, Transformer, and multi-task heads.")

    ctc_criterion = nn.CTCLoss(blank=PAD_ID, zero_infinity=True)
    type_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    prov_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    w_ctc = 1.0
    w_type = 0.25
    w_prov = 0.25

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best_val_acc = 0.0
    best_real_acc = 0.0

    print("=" * 80)
    print("  TRAINING SELENOCR v0.1.1 (DATASET v3.0 + v3.1 COMBINED CURRICULUM)")
    print(f"  Total Images: {len(train_records) + len(val_records):,} | Train: {len(train_records):,} | Val: {len(val_records):,}")
    print(f"  Target: Resilient CCTV Degradation & Low-Res Stamping Invariance")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_ctc_loss = 0.0
        t0 = time.time()

        for batch_i, (imgs, targets, target_lens, type_targets, prov_targets, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)
            type_targets = type_targets.to(device, non_blocking=True)
            prov_targets = prov_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(imgs)
                ctc_logits = outputs["ctc_logits"]
                type_logits = outputs["type_logits"]
                prov_logits = outputs["prov_logits"]

            loss_type = type_criterion(type_logits.float(), type_targets)
            loss_prov = prov_criterion(prov_logits.float(), prov_targets)

            log_probs = ctc_logits.float().log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((imgs.size(0),), ctc_logits.size(1), dtype=torch.long, device=device)
            loss_ctc = ctc_criterion(log_probs, targets, input_lengths, target_lens)

            if torch.isnan(loss_ctc) or torch.isinf(loss_ctc) or \
               torch.isnan(loss_type) or torch.isinf(loss_type) or \
               torch.isnan(loss_prov) or torch.isinf(loss_prov):
                optimizer.zero_grad()
                continue

            total_loss = (w_ctc * loss_ctc) + (w_type * loss_type) + (w_prov * loss_prov)
            if torch.isnan(total_loss) or torch.isinf(total_loss) or not total_loss.requires_grad:
                optimizer.zero_grad()
                continue

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * imgs.size(0)
            train_ctc_loss += loss_ctc.item() * imgs.size(0)

            if (batch_i + 1) % 200 == 0 or (batch_i + 1) == len(train_loader):
                cur_total = total_loss.item()
                cur_ctc = loss_ctc.item()
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  [Epoch {epoch:2d}/{epochs:2d}] Batch [{batch_i+1:4d}/{len(train_loader):4d}] "
                      f"Loss: {cur_total:.4f} (CTC: {cur_ctc:.4f}, Type: {loss_type.item():.4f}, Prov: {loss_prov.item():.4f}) "
                      f"LR: {cur_lr:.2e}", flush=True)

        train_loss /= len(train_ds)
        train_ctc_loss /= len(train_ds)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        exact_matches = 0
        real_matches = 0
        real_total = 0
        total_val = len(val_ds)

        with torch.no_grad():
            for imgs, targets, target_lens, type_targets, prov_targets, records in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lens = target_lens.to(device, non_blocking=True)
                type_targets = type_targets.to(device, non_blocking=True)
                prov_targets = prov_targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = model(imgs)
                    ctc_logits = outputs["ctc_logits"]
                    type_logits = outputs["type_logits"]
                    prov_logits = outputs["prov_logits"]

                loss_type = type_criterion(type_logits.float(), type_targets)
                loss_prov = prov_criterion(prov_logits.float(), prov_targets)
                log_probs = ctc_logits.float().log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((imgs.size(0),), ctc_logits.size(1), dtype=torch.long, device=device)
                loss_ctc = ctc_criterion(log_probs, targets, input_lengths, target_lens)

                total_loss = (w_ctc * loss_ctc) + (w_type * loss_type) + (w_prov * loss_prov)
                val_loss += total_loss.item() * imgs.size(0)

                pred_ids = ctc_logits.argmax(dim=-1).cpu().tolist()
                for i, seq in enumerate(pred_ids):
                    collapsed = [k for j, k in enumerate(seq) if k != PAD_ID and (j == 0 or k != seq[j - 1])]
                    _, _, _, _, pred_line = decode_tokens_to_plate(collapsed)
                    gt_line = records[i]["full_text_line"]
                    is_match = (pred_line.strip() == gt_line.strip())
                    if is_match:
                        exact_matches += 1

                    if records[i].get("is_real_cctv") is True or records[i].get("is_real_cctv") == "True":
                        real_total += 1
                        if is_match:
                            real_matches += 1

        val_loss /= total_val
        val_acc = (exact_matches / total_val) * 100.0
        real_acc = (real_matches / real_total * 100.0) if real_total > 0 else 0.0
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train: {train_loss:.4f} (CTC: {train_ctc_loss:.4f}) | "
              f"Val: {val_loss:.4f} | Overall Val Exact: {val_acc:5.2f}% | "
              f"Real CCTV Val Exact: {real_acc:5.2f}% ({real_matches}/{real_total}) | Time: {elapsed:.1f}s", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = chkpt_dir / "best_reader.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "real_cctv_acc": real_acc,
                "vocab_size": VOCAB_SIZE,
                "architecture": "SelenOCR-v0.1.1-MultiTask-LowRes-CCTV-Curriculum"
            }, best_path)
            print(f"  --> Saved new best SelenOCR v0.1.1 checkpoint: {best_path} (Val Acc: {val_acc:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    args = parser.parse_args()
    train_selen_ocr_v011(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
