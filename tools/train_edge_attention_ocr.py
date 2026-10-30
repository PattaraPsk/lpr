#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EdgeAttentionOCR Training Pipeline (Model Generation selen-lpr-0.0.1)
Lightweight Pretrained Attention-Based Sequence OCR for Edge Thai License Plate Recognition.

Complies with:
- ISO 8601:2019 (Date/Time & UTC Representations)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.7.2 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- DLT Thailand Standard Character Set (41 Consonants, 10 Digits, 77+1 Provinces)
"""

import os
import sys
import math
import time
import json
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
import torchvision.models as models
from tqdm import tqdm

# Add repository root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 stdout on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import (
    EASYOCR_DECODING_LIST,
    decode_tokens_to_plate
)
from tools.DataManipulator.app.provinces import normalize_thai_script, match_province_value

CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(EASYOCR_DECODING_LIST)}


def calculate_sha256(file_path: Path) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def encode_plate_to_tokens(d3: str, d4: str, province: str, max_length: int = 25) -> List[int]:
    """Encodes d3, d4, province into authentic 187-class attention tokens."""
    tokens = []
    # 1. Prefix letters (d3)
    for c in normalize_thai_script(d3):
        if c in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[c])
    tokens.append(60)  # delimiter '_'

    # 2. Digits (d4)
    for c in d4:
        if c in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[c])
    tokens.append(60)  # delimiter '_'

    # 3. Province name
    clean_prov = normalize_thai_script(province).strip()
    if clean_prov in CHAR_TO_IDX:
        tokens.append(CHAR_TO_IDX[clean_prov])

    tokens.append(1)  # EOS token [s]

    target_len = max_length + 1
    if len(tokens) < target_len:
        tokens.extend([0] * (target_len - len(tokens)))
    else:
        tokens = tokens[:target_len]

    return tokens


# =============================================================================
# Architecture: EdgeAttentionOCR (Pretrained MobileNetV3 + Transformer Attention)
# =============================================================================

class EdgeAttentionOCR(nn.Module):
    """
    Lightweight, attention-based sequence recognition model for edge Thai LPR.
    - Pretrained MobileNetV3-Small vision backbone (ImageNet-1K).
    - Transformer Encoder for 2D spatial feature contextualization without bounding boxes.
    - Transformer Cross-Attention Decoder for sequence prediction.
    - Size: ~3.8M parameters (~14.6 MB float32), 75% smaller than legacy baseline (57.9 MB).
    """
    def __init__(self, num_classes: int = 187, max_length: int = 26, d_model: int = 256, nhead: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.max_length = max_length
        self.d_model = d_model

        # 1. Pretrained MobileNetV3-Small backbone (ImageNet)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features  # Downsamples 16x -> (B, 576, H/16, W/16)

        # 2. 1x1 Conv Projection to d_model
        self.proj = nn.Conv2d(576, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

        # 3. Learnable Visual Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 120, d_model) * 0.02)

        # 4. Multi-Head Self-Attention Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 5. Token Embeddings & Learnable Query Positional Embeddings
        self.char_embed = nn.Embedding(num_classes, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

        # 6. Multi-Head Cross-Attention Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # 7. Output Classification Head
        self.fc_out = nn.Linear(d_model, num_classes)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts and contextualizes visual sequence features."""
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        # Visual backbone
        feats = self.backbone(images)       # (B, 576, H', W')
        feats = self.proj(feats)             # (B, d_model, H', W')
        
        # Pool vertically across plate height (H' -> 1)
        feats = feats.mean(dim=2)            # (B, d_model, W')
        visual_seq = feats.permute(0, 2, 1)  # (B, W', d_model)
        visual_seq = self.norm(visual_seq)

        w_seq = visual_seq.shape[1]
        visual_seq = visual_seq + self.pos_encoder[:, :w_seq, :]
        memory = self.encoder(visual_seq)    # (B, W', d_model)
        return memory

    def forward(self, images: torch.Tensor, text_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass:
        - In training (text_tokens provided): Teacher forcing with causal attention mask.
        - In inference (text_tokens is None): Parallel cross-attention querying.
        """
        memory = self.extract_features(images)
        b_size = images.shape[0]

        if text_tokens is not None:
            # Teacher forcing training
            tgt_emb = self.char_embed(text_tokens) + self.dec_pos[:, :text_tokens.shape[1], :]
            seq_len = text_tokens.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=images.device)
            out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=causal_mask)
            logits = self.fc_out(out)  # (B, seq_len, num_classes)
            return logits
        else:
            # High-speed parallel cross-attention query inference
            tgt_emb = self.dec_pos[:, :self.max_length, :].expand(b_size, -1, -1)
            out = self.decoder(tgt=tgt_emb, memory=memory)
            logits = self.fc_out(out)  # (B, max_length, num_classes)
            return logits


# =============================================================================
# Dataset Loader
# =============================================================================

class MultiSourcePlateDataset(Dataset):
    """
    Dataset loader for combined multi-source license plate images.
    Parses {name}_{plate_type}_{d3}_{d4}_{province}.jpg ground truth.
    """
    def __init__(self, data_dirs: List[Path], img_h: int = 64, img_w: int = 384, max_length: int = 25):
        self.img_h = img_h
        self.img_w = img_w
        self.max_length = max_length
        self.samples: List[Tuple[Path, str, str, str]] = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for d in data_dirs:
            d = Path(d).resolve()
            if not d.exists():
                print(f"[WARN] Directory not found: {d}")
                continue

            search_paths = [d / "plates", d] if (d / "plates").is_dir() else [d]
            found_count = 0
            for s_dir in search_paths:
                for p in s_dir.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    p_str = str(p)
                    # Filter out placeholders
                    if "+_+_+_+" in p_str or "-_-_-_-" in p_str:
                        continue

                    d3, d4, prov = self.parse_ground_truth(p)
                    if d3 or d4:
                        self.samples.append((p, d3, d4, prov))
                        found_count += 1

            print(f"[DATASET] Loaded {found_count:,} valid plates from: {d.name}")

        # Deduplicate
        seen = set()
        deduped = []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0])
                deduped.append(s)
        self.samples = deduped
        print(f"[TOTAL] Unique valid training plates: {len(self.samples):,}")

    @staticmethod
    def parse_ground_truth(p: Path) -> Tuple[str, str, str]:
        parts = p.stem.split("_")
        if len(parts) >= 5 and parts[-1] and parts[-2] and parts[-3]:
            d3 = parts[-3]
            d4 = parts[-2]
            prov = parts[-1]
        elif len(parts) >= 4 and parts[-1] and parts[-2] and parts[-3]:
            d3 = parts[-3]
            d4 = parts[-2]
            prov = parts[-1]
        else:
            folder_parts = p.parent.name.split("_")
            if len(folder_parts) >= 4:
                d3 = folder_parts[-3]
                d4 = folder_parts[-2]
                prov = folder_parts[-1]
            elif len(folder_parts) == 3:
                d3, d4, prov = folder_parts[0], folder_parts[1], folder_parts[2]
            else:
                d3 = parts[-3] if len(parts) >= 3 else ""
                d4 = parts[-2] if len(parts) >= 2 else ""
                prov = parts[-1] if len(parts) >= 1 else ""

        if d3 in ["+", "-"]: d3 = ""
        if d4 in ["+", "-"]: d4 = ""
        if prov in ["+", "-"]: prov = ""

        # Consonant clean
        d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค")
        return d3, d4, prov

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p, d3, d4, prov = self.samples[idx]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (self.img_w, self.img_h), color=(128, 128, 128))

        # Aspect ratio preserving resize with padding
        w, h = im.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(math.ceil(self.img_h * ratio)), self.img_w)
        resized_crop = im.resize((resized_w, self.img_h), Image.BICUBIC)

        # Normalized float tensor [-1, 1]
        arr = np.array(resized_crop, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

        pad = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(3, self.img_h, self.img_w - resized_w)

        token_list = encode_plate_to_tokens(d3, d4, prov, max_length=self.max_length)
        targets = torch.tensor(token_list, dtype=torch.long)
        return pad, targets


# =============================================================================
# Training Pipeline
# =============================================================================

def train_edge_attention_ocr(
    data_dirs: List[Path],
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.0005,
    device_str: str = "cuda:0",
    output_dir: str = "runs/train/edge_attention_ocr_a2",
    model_export_path: str = "models/reader/selen-lpr-0.0.1/best_reader.pth"
):
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(model_export_path).resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      EDGE ATTENTION OCR (selen-lpr-0.0.1) TRAINING PIPELINE")
    print("=" * 75)
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Output Dir    : {out_dir}")
    print(f"Export Checkpt: {export_path}")
    print("=" * 75)

    # 1. Dataset & DataLoader (95% Train / 5% Validation Split)
    full_dataset = MultiSourcePlateDataset(data_dirs=data_dirs, img_h=64, img_w=384, max_length=25)
    total_len = len(full_dataset)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    print(f"[DATA SPLIT] Train: {train_len:,} samples | Val: {val_len:,} samples")

    # 2. Model Initialization (Pretrained MobileNetV3 + Transformer Attention)
    model = EdgeAttentionOCR(num_classes=187, max_length=26, d_model=256, nhead=4).to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] EdgeAttentionOCR Initialized. Parameters: {params_count:,} ({params_count*4/1024/1024:.2f} MB float32)")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_exact_acc = 0.0
    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # In teacher forcing, input tokens to decoder is targets[:, :-1]
                dec_in = targets[:, :-1]
                dec_out_target = targets[:, 1:]
                logits = model(images, dec_in)
                loss = criterion(logits.reshape(-1, 187), dec_out_target.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_plates = 0
        total_eval_samples = 0

        with torch.no_grad():
            for val_img, val_tgt in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                val_img = val_img.to(device, non_blocking=True)
                val_tgt = val_tgt.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    # Evaluation loss with teacher forcing
                    dec_in = val_tgt[:, :-1]
                    dec_out_target = val_tgt[:, 1:]
                    logits = model(val_img, dec_in)
                    v_loss = criterion(logits.reshape(-1, 187), dec_out_target.reshape(-1))

                val_loss += v_loss.item()
                val_batches += 1

                # Parallel inference accuracy check
                pred_logits = model(val_img)
                preds = pred_logits.argmax(dim=-1)

                for b in range(val_img.shape[0]):
                    gt_tokens = val_tgt[b].cpu().numpy().tolist()
                    pred_tokens = preds[b].cpu().numpy().tolist()

                    _, _, _, gt_full = decode_tokens_to_plate(gt_tokens)
                    _, _, _, pr_full = decode_tokens_to_plate(pred_tokens)

                    if gt_full and pr_full == gt_full:
                        correct_plates += 1
                    total_eval_samples += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        exact_acc = (correct_plates / max(total_eval_samples, 1)) * 100.0

        print(f"\n[EPOCH {epoch:02d}/{epochs:02d}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Exact Match: {exact_acc:.2f}% ({correct_plates}/{total_eval_samples})")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_loss": round(avg_val_loss, 5),
            "val_exact_acc": round(exact_acc, 2),
            "lr": round(scheduler.get_last_lr()[0], 7)
        }
        history.append(epoch_record)

        # Save Checkpoints
        last_ckpt_path = out_dir / "last_reader.pth"
        torch.save(model.state_dict(), last_ckpt_path)

        if avg_val_loss < best_val_loss or exact_acc > best_exact_acc:
            best_val_loss = min(avg_val_loss, best_val_loss)
            best_exact_acc = max(exact_acc, best_exact_acc)
            best_ckpt_path = out_dir / "best_reader.pth"
            torch.save(model.state_dict(), best_ckpt_path)
            torch.save(model.state_dict(), export_path)
            print(f"[SAVE] New best checkpoint saved to: {best_ckpt_path} and {export_path}")

    total_time = time.time() - t_start
    best_hash = calculate_sha256(export_path)

    # Save training report & history JSON
    report = {
        "model_architecture": "EdgeAttentionOCR (selen-lpr-0.0.1)",
        "backbone": "MobileNetV3-Small (Pretrained ImageNet-1K)",
        "attention_type": "Transformer Multi-Head Self & Cross Attention",
        "parameters": params_count,
        "model_size_mb": round(params_count * 4 / (1024 * 1024), 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "total_training_time_sec": round(total_time, 2),
        "best_val_loss": round(best_val_loss, 5),
        "best_exact_acc": round(best_exact_acc, 2),
        "checkpoint_path": str(export_path),
        "checkpoint_sha256": best_hash,
        "history": history
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("      TRAINING COMPLETE")
    print("=" * 75)
    print(f"Total Duration : {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Best Val Loss  : {best_val_loss:.4f}")
    print(f"Best Exact Acc : {best_exact_acc:.2f}%")
    print(f"Model Checkpoint: {export_path}")
    print(f"SHA-256 Hash   : {best_hash}")
    print("=" * 75 + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EdgeAttentionOCR selen-lpr-0.0.1 on Multi-Source LPR Dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    data_directories = [
        ROOT_DIR / "dataset" / "labeled" / "manually_labeled" / "plates",
        ROOT_DIR / "dataset" / "dataset-cctv-synthetic" / "plates",
        ROOT_DIR / "dataset" / "dataset-mobile-synthetic" / "plates",
        ROOT_DIR / "dataset" / "ready" / "202610261140",
        ROOT_DIR / "dataset" / "ready" / "mobile",
        ROOT_DIR / "dataset" / "verified" / "dataset-staging-approved",
        ROOT_DIR / "dataset" / "verified" / "curated_lpr_dataset",
    ]

    train_edge_attention_ocr(
        data_dirs=data_directories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
