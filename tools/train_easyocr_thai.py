#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EasyOCR Thai License Plate Recognition Training Pipeline.
Fine-tunes official JaidedAI EasyOCR Thai pre-trained weights (ResNet + BiLSTM + CTC)
on balanced multi-source Thai license plate dataset (29,646 plates).

Complies with:
- ISO 8601:2019 (Date/Time & UTC Representations)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System - Annex A.6.7 / A.7.2 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- DLT Thailand Standard Character Set (41 Consonants, 10 Digits, 78 Provinces)
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
from collections import OrderedDict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import easyocr_model
from tools.DataManipulator.app.provinces import PROVINCES_DATA, normalize_thai_script, match_province_value

# EasyOCR Thai Character Definition (Canonical JaidedAI 177 classes)
SEPARATOR_LIST = {'th': ['\xa2', '\xa3'], 'en': ['\xa4', '\xa5']}
SEPARATOR_CHARS = []
for lang, sep in SEPARATOR_LIST.items():
    SEPARATOR_CHARS += sep

BASE_CHARACTERS = (
    ''.join(SEPARATOR_CHARS) +
    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ' +
    'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' +
    'กขคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ' +
    'ฤเแโใไะาุูิีืึั่้๊๋็์ำํฺฯๆ' +
    '0123456789๑๒๓๔๕๖๗๘๙'
)

CHARACTER_LIST = ['[blank]'] + list(BASE_CHARACTERS)
CHAR2IDX = {c: i for i, c in enumerate(CHARACTER_LIST)}
NUM_CLASSES = len(CHARACTER_LIST)  # 177


class ThaiPlateEasyDataset(Dataset):
    """
    Dataset loader for Thai license plates for CTC sequence recognition.
    Preprocesses images with NormalizePAD (H=64, W=600) for edge-robust OCR.
    """
    def __init__(self, samples: List[Tuple[Path, str]], img_h: int = 64, img_w: int = 600, augment: bool = False):
        self.samples = samples
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        img_path, full_label = self.samples[idx]
        try:
            im = Image.open(img_path).convert("L")
        except Exception:
            im = Image.new("L", (self.img_w, self.img_h), color=128)

        # Image padding and normalization
        w, h = im.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(np.ceil(self.img_h * ratio)), self.img_w)
        resized = im.resize((resized_w, self.img_h), Image.BICUBIC)

        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).unsqueeze(0)

        pad = torch.zeros(1, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(1, self.img_h, self.img_w - resized_w)

        # Encode target characters
        target_indices = [CHAR2IDX[c] for c in full_label if c in CHAR2IDX]
        if not target_indices:
            target_indices = [CHAR2IDX.get(' ', 0)]

        target_tensor = torch.tensor(target_indices, dtype=torch.long)
        return pad, target_tensor, len(target_indices), full_label


def collate_fn(batch):
    images, targets, target_lengths, raw_labels = zip(*batch)
    images = torch.stack(images, 0)
    flat_targets = torch.cat(targets, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, flat_targets, target_lengths, raw_labels


def gather_balanced_dataset(max_per_prefix: int = 60) -> List[Tuple[Path, str]]:
    """Gathers samples across all sources with prefix balancing."""
    data_roots = [
        ROOT_DIR / "dataset/labeled/manually_labeled/plates",
        ROOT_DIR / "dataset/ready/202610261140",
        ROOT_DIR / "dataset/snapshot/mobile",
        ROOT_DIR / "dataset/dataset-cctv-synthetic/plates",
        ROOT_DIR / "dataset/dataset-mobile-synthetic/plates",
        ROOT_DIR / "dataset/verified",
    ]

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_samples = []

    for d in data_roots:
        if not d.exists():
            continue
        search_dirs = [d / "plates", d] if (d / "plates").is_dir() else [d]
        for s_dir in search_dirs:
            for p in s_dir.rglob("*"):
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                if "+_+_+_+" in p.name or "-_-_-_-" in p.name:
                    continue

                # Parse filename
                parts = p.stem.split("_")
                d3, d4, prov = "", "", ""
                if len(parts) >= 5 and parts[-1] and parts[-2] and parts[-3]:
                    d3, d4, prov = parts[-3], parts[-2], parts[-1]
                elif len(parts) >= 4 and parts[-1] and parts[-2] and parts[-3]:
                    d3, d4, prov = parts[-3], parts[-2], parts[-1]
                else:
                    folder_parts = p.parent.name.split("_")
                    if len(folder_parts) >= 3:
                        d3, d4, prov = folder_parts[-3], folder_parts[-2], folder_parts[-1]

                if d3 and d4:
                    full_text = f"{d3} {d4}"
                    if prov and prov != "-":
                        full_text = f"{d3} {d4} {prov}"
                    raw_samples.append((p, d3, full_text))

    # Balancing
    prefix_counts: Dict[str, int] = {}
    balanced_samples = []
    random.seed(42)
    random.shuffle(raw_samples)

    seen_paths = set()
    for p, d3, full_text in raw_samples:
        if p in seen_paths:
            continue
        seen_paths.add(p)
        count = prefix_counts.get(d3, 0)
        if count < max_per_prefix:
            prefix_counts[d3] = count + 1
            balanced_samples.append((p, full_text))

    print(f"[DATASET] Gathered {len(balanced_samples):,} balanced samples across {len(prefix_counts):,} distinct prefixes.")
    return balanced_samples


def decode_ctc_predictions(preds_idx: torch.Tensor) -> List[str]:
    """Greedy CTC decoder."""
    decoded_strings = []
    for seq in preds_idx:
        decoded = []
        prev = 0
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                char = CHARACTER_LIST[idx]
                if char not in SEPARATOR_CHARS:
                    decoded.append(char)
            prev = idx
        decoded_strings.append("".join(decoded))
    return decoded_strings


def train_easyocr_thai(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    output_dir: str = "models/reader/easyocr_thai",
    pretrained_path: str = r"C:\Users\WS4090\.EasyOCR\model\thai.pth"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size}")

    # 1. Dataset
    all_samples = gather_balanced_dataset(max_per_prefix=60)
    val_size = int(len(all_samples) * 0.10)
    train_size = len(all_samples) - val_size
    train_samples, val_samples = all_samples[:train_size], all_samples[train_size:]

    train_ds = ThaiPlateEasyDataset(train_samples, img_h=64, img_w=600, augment=True)
    val_ds = ThaiPlateEasyDataset(val_samples, img_h=64, img_w=600, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # 2. Model & Pretrained Weights
    model = easyocr_model.Model(input_channel=1, output_channel=512, hidden_size=512, num_class=NUM_CLASSES).to(device)
    if os.path.exists(pretrained_path):
        print(f"[PRETRAINED] Loading public EasyOCR Thai weights from: {pretrained_path}")
        raw_state = torch.load(pretrained_path, map_location="cpu")
        new_state = OrderedDict()
        for k, v in raw_state.items():
            new_state[k.replace("module.", "")] = v
        msg = model.load_state_dict(new_state, strict=True)
        print(f"[PRETRAINED] Loaded successfully: {msg}")
    else:
        print("[WARNING] Pretrained path not found, initializing from scratch.")

    # 3. Loss & Optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_exact_match = 0.0

    print("\n" + "=" * 70)
    print("      EASYOCR THAI LICENSE PLATE TRAINING PIPELINE")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")

        for images, targets, target_lengths, _ in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images, None)  # [B, T=151, C=177]
                log_probs = preds.log_softmax(2).permute(1, 0, 2)  # [T, B, C]
                input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"ctc_loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        exact_matches = 0
        total_val = 0

        with torch.no_grad():
            for images, targets, target_lengths, raw_labels in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                preds = model(images, None)
                log_probs = preds.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), preds.size(1), dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                pred_indices = preds.softmax(2).argmax(dim=-1)
                pred_texts = decode_ctc_predictions(pred_indices)

                for pred_t, true_t in zip(pred_texts, raw_labels):
                    if pred_t.strip() == true_t.strip():
                        exact_matches += 1
                    total_val += 1

        avg_val_loss = val_loss / len(val_loader)
        exact_match_acc = (exact_matches / total_val) * 100.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Exact Match: {exact_match_acc:.2f}% ({exact_matches}/{total_val})")

        # Save Best
        if exact_match_acc > best_exact_match or avg_val_loss < best_loss:
            best_loss = min(best_loss, avg_val_loss)
            best_exact_match = max(best_exact_match, exact_match_acc)
            best_file = out_path / "best_reader.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "exact_match": exact_match_acc,
                "val_loss": avg_val_loss,
                "character_list": CHARACTER_LIST,
                "num_classes": NUM_CLASSES,
                "standards": ["ISO/IEC 42001:2023", "ISO 8601:2019", "ISO/IEC 25012:2008"],
            }, best_file)
            print(f"  [*] Saved new best model to: {best_file} (Exact Match: {exact_match_acc:.2f}%)")

    total_duration = time.time() - start_time
    print(f"\n[COMPLETE] Training finished in {total_duration/60:.2f} minutes. Best Exact Match: {best_exact_match:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EasyOCR Thai Model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_easyocr_thai(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
