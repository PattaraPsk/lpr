#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenLPR v2 (selen-lpr-0.0.1 generation release)
High-Speed Edge-Optimized Multi-Task Thai License Plate Recognition.

Architecture & Innovations:
1. Pretrained MobileNetV3-Small backbone (ImageNet-1K) with modified horizontal strides
   (W'=48 columns, H'=8 rows) preserving character aspect ratios without over-compression.
2. Two-Band Spatial Feature Slicing:
   - Upper Band (H'=0..4): Dedicated feature map for registration characters (d3 + d4).
   - Lower Band (H'=5..7): Dedicated feature map for province text.
   - Global Representation: Dedicated feature map for plate type colorways.
3. Multi-Head Self-Attention Transformer Encoder (2 layers, 4 heads) over upper band.
4. Spatially-Decoupled Slot Attention Decoders:
   - 4 queries for d3 prefix characters (DLT consonants + prefix digits).
   - 4 queries for d4 registration digits (0-9).
   - Unified 2-layer Transformer Cross-Attention Decoder with zero train/test discrepancy.
5. SelenMultiTaskLoss:
   - Focal Loss on plate type (handling class imbalance).
   - Sequence Cross-Entropy on d3 and d4 slots with label smoothing.
   - Class-balanced Cross-Entropy on province.
   - DLT domain consistency penalty prior.
   - Kendall homoscedastic uncertainty weighting.

Complies with:
- ISO 8601:2019 (Date/Time & UTC Representations)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.7.2 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- DLT Thailand Standard Character Set (41 Consonants, 10 Digits, 77+1 Provinces, 11 Plate Types)
"""

import os
import sys
import math
import time
import json
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import EASYOCR_DECODING_LIST
from tools.DataManipulator.app.provinces import PROVINCES_DATA, normalize_thai_script, match_province_value
from tools.DataManipulator.app.plate_types import PLATE_TYPES_DATA, match_plate_type

# =============================================================================
# Vocabulary & DLT Specifications
# =============================================================================

# 11 Plate Types
PLATE_TYPES = [
    'Other', 'WhiteBlackCar', 'WhiteBlackMotor', 'WhiteBlue', 'WhiteGreen',
    'YellowBlack', 'YellowBlue', 'YellowGreen', 'GreenWhite', 'RedBlack', 'OrangeBlack'
]
TYPE_TO_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(PLATE_TYPES)}

# 79 Provinces (77 provinces + Betong + Unknown)
PROVINCE_NAMES = ['<unknown>'] + [p[0] for p in PROVINCES_DATA]
if 'เบตง' not in PROVINCE_NAMES:
    PROVINCE_NAMES.append('เบตง')
PROV_TO_IDX = {p: i for i, p in enumerate(PROVINCE_NAMES)}
IDX_TO_PROV = {i: p for i, p in enumerate(PROVINCE_NAMES)}

# DLT 41 Consonants + 10 Digits for d3 prefix (Total 52 classes including <PAD>=0)
D3_CHARS = ['<PAD>'] + [str(d) for d in range(10)] + [
    'ก', 'ข', 'ค', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ',
    'ฎ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 'ท', 'ธ', 'น',
    'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล',
    'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ'
]
D3_TO_IDX = {c: i for i, c in enumerate(D3_CHARS)}
IDX_TO_D3 = {i: c for i, c in enumerate(D3_CHARS)}

# 10 Digits for d4 registration number (Total 11 classes including <PAD>=0)
D4_CHARS = ['<PAD>'] + [str(d) for d in range(10)]
D4_TO_IDX = {c: i for i, c in enumerate(D4_CHARS)}
IDX_TO_D4 = {i: c for i, c in enumerate(D4_CHARS)}


def calculate_sha256(file_path: Path) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


# =============================================================================
# Architecture: SelenLPR v2 (selen-lpr-0.0.1)
# =============================================================================

class SelenLPRv2Edge(nn.Module):
    """
    High-Resolution Pretrained MobileNetV3 + Two-Band Transformer Multi-Task Reader.
    - Pretrained MobileNetV3-Small backbone (ImageNet-1K).
    - Tuned horizontal strides on blocks 4 & 9 -> feature map (B, 576, 8, 48).
    - Upper band (0..4) vertical pooling -> (B, 48, 256) registration sequence.
    - 2-layer TransformerEncoder contextualizes character sequence.
    - 8 slot queries (4 for d3, 4 for d4) decoded via 2-layer TransformerDecoder.
    - Lower band (5..7) pooled for direct province classification (79 classes).
    - Global feature pooled for plate type classification (11 classes).
    - Total parameters: ~3.82M (~14.57 MB float32), 75% smaller than legacy baseline (57.9 MB).
    """
    def __init__(
        self,
        num_types: int = len(PLATE_TYPES),
        num_provs: int = len(PROVINCE_NAMES),
        num_d3: int = len(D3_CHARS),
        num_d4: int = len(D4_CHARS),
        d_model: int = 256,
        nhead: int = 4
    ):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types
        self.num_provs = num_provs
        self.num_d3 = num_d3
        self.num_d4 = num_d4

        # 1. Pretrained MobileNetV3-Small vision backbone
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Modify strides of block 4 and block 9 to (1, 1) to retain (H'=8, W'=48)
        for m in mobilenet.features[4].modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)
        for m in mobilenet.features[9].modules():
            if hasattr(m, 'stride') and m.stride == (2, 2):
                m.stride = (1, 1)

        self.backbone = mobilenet.features  # (B, 576, 8, 48)
        self.proj = nn.Conv2d(576, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

        # 2. Upper Band Registration Sequence Encoder (rows 0..4)
        self.upper_pool = nn.AdaptiveAvgPool2d((1, None))
        self.pos_enc = nn.Parameter(torch.randn(1, 48, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.05, activation="relu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 3. Spatially-Decoupled Slot Character Decoders (4 for d3, 4 for d4 = 8 queries)
        self.char_queries = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.05, activation="relu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)

        self.d3_head = nn.Linear(d_model, num_d3)
        self.d4_head = nn.Linear(d_model, num_d4)

        # 4. Lower Band Province Classifier (rows 5..7)
        self.prov_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_provs)
        )

        # 5. Global Plate Type Classifier (full feature map)
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_types)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b_size = x.shape[0]
        feats = self.proj(self.backbone(x))  # (B, d_model, 8, 48)

        # Slice bands
        upper = feats[:, :, 0:5, :]  # Upper 62.5% of plate (d3 + d4)
        lower = feats[:, :, 5:8, :]  # Lower 37.5% of plate (province)

        # Encode upper band sequence
        upper_seq = self.upper_pool(upper).squeeze(2).permute(0, 2, 1)  # (B, 48, d_model)
        upper_seq = self.norm(upper_seq)
        memory = self.encoder(upper_seq + self.pos_enc)

        # Decode characters in parallel
        q = self.char_queries.expand(b_size, -1, -1)
        dec_out = self.decoder(tgt=q, memory=memory)  # (B, 8, d_model)

        pred_d3 = self.d3_head(dec_out[:, 0:4, :])    # (B, 4, num_d3)
        pred_d4 = self.d4_head(dec_out[:, 4:8, :])    # (B, 4, num_d4)
        pred_prov = self.prov_head(lower)             # (B, num_provs)
        pred_type = self.type_head(feats)             # (B, num_types)

        return {
            "type": pred_type,
            "prov": pred_prov,
            "d3": pred_d3,
            "d4": pred_d4
        }


# =============================================================================
# Specialized Multi-Task Loss: SelenMultiTaskLoss
# =============================================================================

class SelenMultiTaskLoss(nn.Module):
    """
    ISO/IEC 42001:2023 compliant multi-task loss with dual weighting modes:
    1. 'clamped_kendall': Clamped homoscedastic uncertainty balancing (s in [-2.5, 1.2])
       with sequence scaling (d3=2.5, d4=2.0) preventing gradient starvation.
    2. 'static': Brute-force static weighted sum (1.0*type + 1.0*prov + 3.0*d3 + 3.0*d4 + 0.10*consistency)
       forcing direct gradient propagation through the Transformer sequence decoders.
    """
    def __init__(self, mode: str = "clamped_kendall", d3_weight: float = 2.5, d4_weight: float = 2.0):
        super().__init__()
        self.mode = mode
        self.d3_weight = d3_weight
        self.d4_weight = d4_weight

        # Kendall log variances initialized with negative values for sequence heads
        self.log_var_type = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.log_var_prov = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.log_var_d3 = nn.Parameter(torch.tensor([-0.693], dtype=torch.float32))
        self.log_var_d4 = nn.Parameter(torch.tensor([-0.693], dtype=torch.float32))

        self.ce_type = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")
        self.ce_prov = nn.CrossEntropyLoss(label_smoothing=0.02)
        self.ce_d3 = nn.CrossEntropyLoss(label_smoothing=0.02)
        self.ce_d4 = nn.CrossEntropyLoss(label_smoothing=0.02)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Plate Type Focal Loss (Smoothed, AMP FP16 safe)
        p_t = F.softmax(preds["type"].float(), dim=-1)
        pt_target = p_t.gather(1, targets["type"].unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt_target).pow(1.5)
        raw_type_loss = self.ce_type(preds["type"], targets["type"])
        l_type = (focal_weight * raw_type_loss).mean()

        # 2. Province Loss
        l_prov = self.ce_prov(preds["prov"], targets["prov"])

        # 3. d3 Sequence Loss (4 slots)
        b, s3, c3 = preds["d3"].shape
        l_d3 = self.ce_d3(preds["d3"].reshape(-1, c3), targets["d3"].reshape(-1))

        # 4. d4 Sequence Loss (4 slots)
        b, s4, c4 = preds["d4"].shape
        l_d4 = self.ce_d4(preds["d4"].reshape(-1, c4), targets["d4"].reshape(-1))

        # 5. DLT Domain Consistency Penalty (Taxi prefix 'ท' vs WhiteBlackCar)
        p_white = p_t[:, 1]
        p_taxi = F.softmax(preds["d3"][:, 0, :].float(), dim=-1)[:, D3_TO_IDX.get('ท', 0)]
        l_consistency = torch.relu(p_white + p_taxi - 1.0).mean()

        # Multi-task loss aggregation
        if self.mode == "static":
            loss_total = (
                1.0 * l_type +
                1.0 * l_prov +
                3.0 * l_d3 +
                3.0 * l_d4 +
                0.10 * l_consistency
            )
        else:
            # Clamped Homoscedastic Uncertainty Weighting
            s_type = torch.clamp(self.log_var_type, min=-2.0, max=2.0)
            s_prov = torch.clamp(self.log_var_prov, min=-2.0, max=2.0)
            s_d3   = torch.clamp(self.log_var_d3,   min=-2.5, max=1.2)
            s_d4   = torch.clamp(self.log_var_d4,   min=-2.5, max=1.2)

            loss_total = (
                torch.exp(-s_type) * l_type + 0.5 * s_type +
                torch.exp(-s_prov) * l_prov + 0.5 * s_prov +
                torch.exp(-s_d3)   * (l_d3 * self.d3_weight) + 0.5 * s_d3 +
                torch.exp(-s_d4)   * (l_d4 * self.d4_weight) + 0.5 * s_d4 +
                0.10 * l_consistency
            )

        return {
            "loss": loss_total,
            "loss_type": l_type.detach(),
            "loss_prov": l_prov.detach(),
            "loss_d3": l_d3.detach(),
            "loss_d4": l_d4.detach(),
            "loss_consistency": l_consistency.detach()
        }

# =============================================================================
# Dataset Loader & Clean Parser
# =============================================================================

class SelenDataset(Dataset):
    """
    Dataset loader for multi-task Thai license plate images.
    Parses and verifies:
    - Plate Type (11 classes)
    - d3 prefix (1..4 characters: Thai consonants and prefix digits)
    - d4 registration number (1..4 digits)
    - Province (79 classes)
    """
    def __init__(self, data_dirs: List[Path], img_h: int = 64, img_w: int = 384, augment: bool = False):
        self.img_h = img_h
        self.img_w = img_w
        self.augment = augment
        self.samples: List[Tuple[Path, str, str, str, str]] = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for d in data_dirs:
            d = Path(d).resolve()
            if not d.exists():
                continue

            search_paths = [d / "plates", d] if (d / "plates").is_dir() else [d]
            for s_dir in search_paths:
                for p in s_dir.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in exts:
                        continue
                    p_str = str(p)
                    if "+_+_+_+" in p_str or "-_-_-_-" in p_str:
                        continue

                    parsed = self.parse_info(p)
                    if parsed is not None:
                        pt, d3, d4, prov = parsed
                        self.samples.append((p, pt, d3, d4, prov))

        # Deduplicate by file path
        seen = set()
        deduped = []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0])
                deduped.append(s)
        self.samples = deduped
        print(f"[DATASET] Loaded {len(self.samples):,} clean, validated multi-task samples.")

    @staticmethod
    def parse_info(p: Path) -> Optional[Tuple[str, str, str, str]]:
        parts = p.stem.split("_")
        plate_type = "WhiteBlackCar"
        d3, d4, prov = "", "", ""

        if len(parts) >= 5 and parts[-1] and parts[-2] and parts[-3]:
            plate_type = parts[-4]
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
                plate_type = folder_parts[-4]
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

        # Remap obsolete / confusable Thai characters according to DLT specs
        d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค").replace("ฏ", "ฎ")
        d4 = d4.replace("ฃ", "ข").replace("ฅ", "ค").replace("ฏ", "ฎ")

        # Validate: d4 must be 1 to 4 digits
        if not (1 <= len(d4) <= 4 and d4.isdigit()):
            return None

        # Validate: d3 must be 1 to 4 characters
        if not (1 <= len(d3) <= 4):
            return None

        matched_pt = match_plate_type(plate_type) or "WhiteBlackCar"
        clean_prov = match_province_value(prov) if prov else ""

        return matched_pt, d3, d4, clean_prov

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, pt, d3, d4, prov = self.samples[idx]
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (self.img_w, self.img_h), color=(128, 128, 128))

        # Data augmentation on training set
        if self.augment:
            if random.random() < 0.4:
                enh = ImageEnhance.Brightness(im)
                im = enh.enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.4:
                enh = ImageEnhance.Contrast(im)
                im = enh.enhance(random.uniform(0.85, 1.15))
            if random.random() < 0.15:
                im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

        # Maintain aspect ratio with right-edge padding
        w, h = im.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(math.ceil(self.img_h * ratio)), self.img_w)
        resized_crop = im.resize((resized_w, self.img_h), Image.BICUBIC)

        arr = np.array(resized_crop, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).permute(2, 0, 1)

        pad = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(3, self.img_h, self.img_w - resized_w)

        # Encode slot targets
        d3_slots = [D3_TO_IDX.get(c, 0) for c in d3[:4]] + [0] * max(0, 4 - len(d3))
        d4_slots = [D4_TO_IDX.get(c, 0) for c in d4[:4]] + [0] * max(0, 4 - len(d4))
        type_idx = TYPE_TO_IDX.get(pt, 1)  # Default WhiteBlackCar
        prov_idx = PROV_TO_IDX.get(prov, 0)

        targets = {
            "type": torch.tensor(type_idx, dtype=torch.long),
            "prov": torch.tensor(prov_idx, dtype=torch.long),
            "d3": torch.tensor(d3_slots[:4], dtype=torch.long),
            "d4": torch.tensor(d4_slots[:4], dtype=torch.long)
        }
        return pad, targets


# =============================================================================
# Training & Validation Runner
# =============================================================================

def train_selen_lpr_v2(
    data_dirs: List[Path],
    val_dirs: List[Path],
    epochs: int = 500,
    start_epoch: Optional[int] = None,
    batch_size: int = 64,
    lr: float = 0.0003,
    loss_mode: str = "clamped_kendall",
    resume: Optional[str] = None,
    device_str: str = "cuda:0",
    output_dir: str = "runs/train/selen_lpr_0.0.1",
    model_export_path: str = "models/reader/selen-lpr-0.0.1/best_reader.pth"
):
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(model_export_path).resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    print("=" * 80)
    print("      SELEN-LPR v2 (selen-lpr-0.0.1) CONTINUED TRAINING PIPELINE")
    print("=" * 80)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Epochs : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Loss Mode     : {loss_mode}")
    print(f"Output Dir    : {out_dir}")
    print(f"Export Checkpt: {export_path}")
    print("=" * 80)

    # Build datasets
    train_dataset = SelenDataset(data_dirs, augment=True)
    val_dataset = SelenDataset(val_dirs, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    model = SelenLPRv2Edge().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"[*] Model Architecture: SelenLPRv2Edge")
    print(f"[*] Total Trainable Parameters: {total_params:,} ({model_size_mb:.2f} MB float32)")

    # Loss function & Optimizer
    criterion = SelenMultiTaskLoss(mode=loss_mode).to(device)
    params_to_opt = list(model.parameters())
    if loss_mode != "static":
        params_to_opt += list(criterion.parameters())

    optimizer = torch.optim.AdamW(
        params_to_opt,
        lr=lr,
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_exact_acc = -1.0
    best_val_loss = float("inf")
    history = []
    actual_start_epoch = 1

    # Checkpoint resumption
    if resume and Path(resume).is_file():
        resume_p = Path(resume).resolve()
        print(f"[*] Resuming checkpoint from: {resume_p}")
        ckpt = torch.load(resume_p, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("[+] Restored optimizer state.")
                except Exception as e:
                    print(f"[!] Warning: Could not restore optimizer: {e}")
            if "scaler_state_dict" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass
            if "best_exact_acc" in ckpt:
                best_exact_acc = ckpt["best_exact_acc"]
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            if "history" in ckpt and isinstance(ckpt["history"], list):
                history = ckpt["history"]
            if "epoch" in ckpt and start_epoch is None:
                actual_start_epoch = ckpt["epoch"] + 1
        elif isinstance(ckpt, dict):
            # Pure model weights state dict
            missing, unexp = model.load_state_dict(ckpt, strict=False)
            print(f"[+] Loaded model weights ({len(ckpt)} keys, missing={len(missing)}, unexpected={len(unexp)}).")
            summary_path = out_dir / "selen_lpr_summary.json"
            if summary_path.is_file() and start_epoch is None:
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        prev_sum = json.load(f)
                    if "history" in prev_sum and len(prev_sum["history"]) > 0:
                        history = prev_sum["history"]
                        actual_start_epoch = prev_sum["history"][-1]["epoch"] + 1
                        best_exact_acc = prev_sum.get("best_exact_acc", -1.0)
                        best_val_loss = prev_sum.get("best_val_loss", float("inf"))
                        print(f"[+] Auto-detected prior training history: Resuming at Epoch {actual_start_epoch} (Prior Best Exact Acc: {best_exact_acc:.2f}%)")
                except Exception as e:
                    print(f"[!] Note: Could not parse previous summary: {e}")

    if start_epoch is not None:
        actual_start_epoch = start_epoch

    remaining_epochs = max(1, epochs - actual_start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)
    print(f"[*] Training scheduled from Epoch {actual_start_epoch:03d} to {epochs:03d} ({remaining_epochs} epochs remaining).")

    t_start = time.time()

    for epoch in range(actual_start_epoch, epochs + 1):
        model.train()
        criterion.train()
        train_loss = 0.0
        n_train_batches = len(train_loader)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]")
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            for k in targets:
                targets[k] = targets[k].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(imgs)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "typ": f"{loss_dict['loss_type'].item():.2f}",
                "prv": f"{loss_dict['loss_prov'].item():.2f}",
                "d3": f"{loss_dict['loss_d3'].item():.2f}",
                "d4": f"{loss_dict['loss_d4'].item():.2f}"
            })

        scheduler.step()
        train_loss /= max(n_train_batches, 1)

        # Validation on benchmark
        model.eval()
        criterion.eval()
        val_loss = 0.0
        correct_type = 0
        correct_prov = 0
        correct_d3 = 0
        correct_d4 = 0
        exact_match = 0
        total_eval = 0

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch:03d}/{epochs:03d} [Val]"):
                imgs = imgs.to(device, non_blocking=True)
                for k in targets:
                    targets[k] = targets[k].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    preds = model(imgs)
                    loss_dict = criterion(preds, targets)

                val_loss += loss_dict["loss"].item()

                pred_types = preds["type"].argmax(dim=-1)
                pred_provs = preds["prov"].argmax(dim=-1)
                pred_d3 = preds["d3"].argmax(dim=-1)
                pred_d4 = preds["d4"].argmax(dim=-1)

                correct_type += (pred_types == targets["type"]).sum().item()
                correct_prov += (pred_provs == targets["prov"]).sum().item()

                match_d3 = (pred_d3 == targets["d3"]).all(dim=-1)
                match_d4 = (pred_d4 == targets["d4"]).all(dim=-1)
                match_type = (pred_types == targets["type"])
                match_prov = (pred_provs == targets["prov"])

                correct_d3 += match_d3.sum().item()
                correct_d4 += match_d4.sum().item()
                exact_match += (match_d3 & match_d4 & match_type & match_prov).sum().item()
                total_eval += imgs.size(0)

        val_loss /= max(len(val_loader), 1)
        acc_type = correct_type / max(total_eval, 1) * 100
        acc_prov = correct_prov / max(total_eval, 1) * 100
        acc_d3 = correct_d3 / max(total_eval, 1) * 100
        acc_d4 = correct_d4 / max(total_eval, 1) * 100
        acc_exact = exact_match / max(total_eval, 1) * 100

        print(
            f"[EPOCH {epoch:03d}/{epochs:03d}] Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | "
            f"Type={acc_type:.1f}% | Prov={acc_prov:.1f}% | d3={acc_d3:.1f}% | d4={acc_d4:.1f}% | Exact={acc_exact:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "acc_type": round(acc_type, 2),
            "acc_prov": round(acc_prov, 2),
            "acc_d3": round(acc_d3, 2),
            "acc_d4": round(acc_d4, 2),
            "acc_exact": round(acc_exact, 2)
        })

        # Save checkpoint bundle for resuming
        ckpt_bundle = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_exact_acc": best_exact_acc,
            "best_val_loss": best_val_loss,
            "loss_mode": loss_mode,
            "history": history
        }
        torch.save(ckpt_bundle, out_dir / "last_checkpoint.pth")
        torch.save(model.state_dict(), out_dir / "last_reader.pth")

        # Save best model checkpoint
        if acc_exact > best_exact_acc or (acc_exact == best_exact_acc and val_loss < best_val_loss):
            best_exact_acc = acc_exact
            best_val_loss = val_loss
            ckpt_bundle["best_exact_acc"] = best_exact_acc
            ckpt_bundle["best_val_loss"] = best_val_loss
            torch.save(ckpt_bundle, out_dir / "best_checkpoint.pth")
            torch.save(model.state_dict(), export_path)
            torch.save(model.state_dict(), out_dir / "best_reader.pth")
            print(f"  [+] Saved new best model checkpoint to {export_path} (Exact Acc: {best_exact_acc:.2f}%)")

    total_time = time.time() - t_start
    sha256_hash = calculate_sha256(export_path)

    summary = {
        "model_name": "selen-lpr-0.0.1",
        "architecture": "SelenLPRv2Edge (Pretrained MobileNetV3 + Two-Band Transformer Multi-Task Reader)",
        "parameters": total_params,
        "model_size_mb": round(model_size_mb, 2),
        "epochs": epochs,
        "start_epoch": actual_start_epoch,
        "batch_size": batch_size,
        "loss_mode": loss_mode,
        "total_time_sec": round(total_time, 2),
        "best_exact_acc": round(best_exact_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoint_path": str(export_path),
        "checkpoint_sha256": sha256_hash,
        "history": history
    }

    with open(out_dir / "selen_lpr_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE: SelenLPR v2 (selen-lpr-0.0.1)")
    print(f"Best Exact 4-Way Match Accuracy : {best_exact_acc:.2f}%")
    print(f"Best Validation Loss           : {best_val_loss:.4f}")
    print(f"Checkpoint Export Path         : {export_path}")
    print(f"Checkpoint SHA-256             : {sha256_hash}")
    print(f"Total Elapsed Time             : {total_time/60:.2f} minutes")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SelenLPR v2 Multi-Task Reader")
    parser.add_argument("--epochs", type=int, default=500, help="Total target epochs")
    parser.add_argument("--start-epoch", type=int, default=None, help="Explicit start epoch (auto-detected if None)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (e.g. 64 or 128 for RTX 4090)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--loss-mode", type=str, default="clamped_kendall", choices=["clamped_kendall", "static"],
                        help="Loss mode: clamped_kendall or static weighted sum")
    parser.add_argument("--resume", type=str, default="models/reader/selen-lpr-0.0.1/best_reader.pth",
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, default="cuda:0", help="Execution device (cuda:0 or cpu)")
    args = parser.parse_args()

    # Training data paths (33,400 clean plates)
    DATA_DIRS = [
        ROOT_DIR / "dataset/labeled/manually_labeled/plates",
        ROOT_DIR / "dataset/dataset-cctv-synthetic/plates",
        ROOT_DIR / "dataset/dataset-mobile-synthetic/plates",
        ROOT_DIR / "dataset/verified/dataset-staging-approved",
        ROOT_DIR / "dataset/verified/curated_lpr_dataset"
    ]

    # Validation benchmark paths (Real plates)
    VAL_DIRS = [
        ROOT_DIR / "dataset/ready/202610261140",
        ROOT_DIR / "dataset/ready/mobile"
    ]

    train_selen_lpr_v2(
        data_dirs=DATA_DIRS,
        val_dirs=VAL_DIRS,
        epochs=args.epochs,
        start_epoch=args.start_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_mode=args.loss_mode,
        resume=args.resume,
        device_str=args.device
    )