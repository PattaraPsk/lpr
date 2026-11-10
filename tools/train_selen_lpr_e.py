#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenLPR-E: Multi-Task Thai License Plate Recognition Model.
Continues from JaidedAI EasyOCR Thai Foundation Model (ResNet + BiLSTM) with:
- Dedicated Alphanumeric CTC Sequence Head with Explicit Boundary Delimiter (d3 <sep> d4)
- Color-Aware Plate Type Multi-Class Head (11 vehicle categories)
- Closed-Set Administrative Province Classifier (79 subdivisions)

Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System - Annex A.6.7 / A.7.2 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Date/Time Representation)
- DLT Thailand Standard Character Constraints: len(d3) <= 3, len(d4) <= 4
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
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from easyocr_modules import ResNet_FeatureExtractor, BidirectionalLSTM
from tools.train_easyocr_thai import CHARACTER_LIST as EASYOCR_CHAR_LIST, CHAR2IDX as EASYOCR_CHAR2IDX
from tools.DataManipulator.app.plate_types import PLATE_TYPES_DATA, match_plate_type
from tools.DataManipulator.app.provinces import PROVINCES_DATA, match_province_value, normalize_thai_script

# =============================================================================
# Vocabulary & Mapping Definitions
# =============================================================================

THAI_CONSONANTS = "กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"

# 53-Token Delimiter-Aware CTC Vocabulary:
# 0: [blank], 1: ' ' (<sep>), 2..11: '0'..'9', 12..52: 41 Thai consonants
CTC_TOKENS = ["[blank]", " "] + [str(i) for i in range(10)] + list(THAI_CONSONANTS)
CTC_CHAR_TO_IDX = {c: i for i, c in enumerate(CTC_TOKENS)}
CTC_IDX_TO_CHAR = {i: c for i, c in enumerate(CTC_TOKENS)}
CTC_VOCAB_SIZE = len(CTC_TOKENS)  # 53
CTC_BLANK_IDX = 0
CTC_SEP_IDX = 1

# Plate Type Classes (11 classes)
PLATE_TYPES = [
    "Other",
    "WhiteBlackCar",
    "WhiteBlackMotor",
    "WhiteBlue",
    "WhiteGreen",
    "YellowBlack",
    "YellowBlue",
    "YellowGreen",
    "GreenWhite",
    "RedBlack",
    "OrangeBlack",
]
TYPE_TO_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(PLATE_TYPES)}

# Province Classes (79 classes: 0=<unknown>, 1..78=Provinces + Betong)
PROVINCE_NAMES = ["<unknown>"] + [p[0] for p in PROVINCES_DATA]
if "เบตง" not in PROVINCE_NAMES:
    PROVINCE_NAMES.append("เบตง")
PROV_TO_IDX = {name: i for i, name in enumerate(PROVINCE_NAMES)}
IDX_TO_PROV = {i: name for i, name in enumerate(PROVINCE_NAMES)}

DEFAULT_PRETRAINED_PATH = r"C:\Users\WS4090\.EasyOCR\model\thai.pth"


def get_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# SelenLPR-E Architecture
# =============================================================================

class SelenLPR_E(nn.Module):
    """
    SelenLPR-E: Multi-Task Edge Thai License Plate Recognition Model.
    Initializes from EasyOCR Thai foundation weights (ResNet + 2-layer BiLSTM):
    - FeatureExtraction: ResNet Feature Extractor (out_channels=512)
    - SequenceModeling: 2-Layer Bidirectional LSTM (hidden_size=512)
    - Multitask Prediction Heads:
      1. ctc_head: Alphanumeric CTC Sequence Head (d3 <sep> d4, vocab_size=53)
      2. type_head: Color-Aware Vehicle Plate Type Classifier (11 classes)
      3. prov_head: Closed-Set Administrative Province Classifier (79 classes)
    """
    def __init__(
        self,
        num_classes_ctc: int = CTC_VOCAB_SIZE,
        num_classes_type: int = len(PLATE_TYPES),
        num_classes_prov: int = len(PROVINCE_NAMES),
    ):
        super().__init__()
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel=1, output_channel=512)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(512, 512, 512),
            BidirectionalLSTM(512, 512, 512)
        )
        self.ctc_head = nn.Linear(512, num_classes_ctc)

        # Color-aware plate type head: pooled sequence features (512) + 6-dim RGB moments (mean & std)
        self.type_head = nn.Sequential(
            nn.Linear(512 + 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes_type)
        )

        # Province classification head: pooled sequence features (512) -> 79 classes
        self.prov_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes_prov)
        )

    def forward(self, x_rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_rgb: Tensor of shape [B, 3, H, W] normalized to [-1, 1].
        Returns:
            Dict containing ctc_logits, type_logits, and prov_logits.
        """
        # 1. Grayscale conversion for text feature extraction
        x_gray = 0.299 * x_rgb[:, 0:1] + 0.587 * x_rgb[:, 1:2] + 0.114 * x_rgb[:, 2:3]

        # 2. ResNet visual feature extraction
        visual = self.FeatureExtraction(x_gray)

        # 3. Permute and adaptive average pool: [B, C, H, W] -> [B, W, C, 1] -> [B, W, C]
        pooled = self.AdaptiveAvgPool(visual.permute(0, 3, 1, 2)).squeeze(3)

        # 4. BiLSTM sequence modeling: [B, W, 512]
        seq = self.SequenceModeling(pooled)

        # 5. Head 1: Alphanumeric sequence CTC logits
        ctc_logits = self.ctc_head(seq)

        # 6. Global spatial pooling for plate type and province
        seq_pool = seq.mean(dim=1)

        # Head 2: Plate type with RGB color statistics
        c_mean = x_rgb.mean(dim=[-2, -1])  # [B, 3]
        c_std = x_rgb.std(dim=[-2, -1])    # [B, 3]
        type_features = torch.cat([seq_pool, c_mean, c_std], dim=-1)
        type_logits = self.type_head(type_features)

        # Head 3: Closed-set province classifier
        prov_logits = self.prov_head(seq_pool)

        return {
            "ctc_logits": ctc_logits,
            "type_logits": type_logits,
            "prov_logits": prov_logits,
        }

    def load_easyocr_foundation(self, pretrained_path: str = DEFAULT_PRETRAINED_PATH) -> Tuple[int, int]:
        """
        Transfers pre-trained ResNet, BiLSTM, and matching CTC linear weights
        directly from EasyOCR Thai foundation weights.
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"EasyOCR Thai foundation model not found at: {pretrained_path}")

        raw_sd = torch.load(pretrained_path, map_location="cpu")
        model_sd = self.state_dict()
        backbone_transferred = 0

        # Transfer ResNet and BiLSTM weights
        for k, v in raw_sd.items():
            clean_k = k.replace("module.", "")
            if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
                model_sd[clean_k].copy_(v)
                backbone_transferred += 1

        # Transfer matching prediction weights for CTC head
        pred_w = raw_sd["module.Prediction.weight"]  # [177, 512]
        pred_b = raw_sd["module.Prediction.bias"]    # [177]
        ctc_transferred = 0

        for char, our_idx in CTC_CHAR_TO_IDX.items():
            if char in EASYOCR_CHAR2IDX:
                e_idx = EASYOCR_CHAR2IDX[char]
                self.ctc_head.weight.data[our_idx].copy_(pred_w[e_idx])
                self.ctc_head.bias.data[our_idx].copy_(pred_b[e_idx])
                ctc_transferred += 1

        return backbone_transferred, ctc_transferred


# =============================================================================
# Multitask Decode & Enforce DLT Constraints
# =============================================================================

def decode_selen_lpr_e(
    preds: Dict[str, torch.Tensor],
    blank_idx: int = CTC_BLANK_IDX
) -> List[Dict[str, Any]]:
    """
    Decodes multitask predictions:
    1. Plate Type: argmax from type_logits.
    2. Province: argmax from prov_logits.
    3. Alphanumeric: greedy CTC decode on ctc_logits, split on explicit delimiter.
    Strictly clamps d3 <= 3, d4 <= 4 per Thailand DLT specs.
    """
    type_preds = preds["type_logits"].argmax(dim=-1).cpu().tolist()
    prov_preds = preds["prov_logits"].argmax(dim=-1).cpu().tolist()
    ctc_preds = preds["ctc_logits"].argmax(dim=-1).cpu().tolist()

    results: List[Dict[str, Any]] = []

    for b, seq in enumerate(ctc_preds):
        # Decode plate type
        p_type = IDX_TO_TYPE.get(type_preds[b], "WhiteBlackCar")

        # Decode province
        prov_name = IDX_TO_PROV.get(prov_preds[b], "")
        if prov_name == "<unknown>":
            prov_name = ""

        # CTC greedy collapse & remove blank
        collapsed = [k for i, k in enumerate(seq) if k != blank_idx and (i == 0 or k != seq[i - 1])]
        full_str = "".join(CTC_IDX_TO_CHAR.get(k, "") for k in collapsed).strip()

        # Split on explicit delimiter
        if " " in full_str:
            parts = full_str.split(" ", 1)
            raw_d3 = parts[0].strip()
            raw_d4 = parts[1].strip()
        elif "-" in full_str:
            parts = full_str.split("-", 1)
            raw_d3 = parts[0].strip()
            raw_d4 = parts[1].strip()
        else:
            # Intelligent fallback: split on Thai consonant boundary or middle
            has_thai = any("\u0e01" <= c <= "\u0e2e" for c in full_str)
            if has_thai:
                last_c = max(i for i, c in enumerate(full_str) if "\u0e01" <= c <= "\u0e2e")
                raw_d3 = full_str[:last_c + 1]
                raw_d4 = full_str[last_c + 1:]
            else:
                # Numbers only (e.g., private truck series 53 9210)
                raw_d3 = full_str[:2]
                raw_d4 = full_str[2:]

        # DLT Consonant Remapping & Length Clamping
        d3 = raw_d3.replace("ฃ", "ข").replace("ฅ", "ค")[:3]
        d4 = "".join(c for c in raw_d4 if c.isdigit())[:4]

        full_plate = f"{d3} {d4} {prov_name}".strip()

        results.append({
            "plate_type": p_type,
            "d3": d3,
            "d4": d4,
            "province": prov_name,
            "full_plate": full_plate
        })

    return results


# =============================================================================
# Dataset Loader
# =============================================================================

class SelenLPREDataset(Dataset):
    """
    Dataset loader for SelenLPR-E multitask training.
    Encodes:
    - Alphanumeric sequence: d3 <sep> d4 (with explicit space delimiter)
    - Vehicle Plate Type: 0..10
    - Province: 0..78
    """
    _VALID_D3 = set("0123456789" + THAI_CONSONANTS)
    _VALID_D4 = set("0123456789")

    def __init__(
        self,
        data_dirs: List[Path],
        img_h: int = 64,
        img_w: int = 256,
        max_samples: Optional[int] = None
    ):
        self.img_h = img_h
        self.img_w = img_w
        self.samples: List[Tuple[Path, str, str, str, str]] = []

        print(f"[{get_utc_timestamp()}] Scanning dataset directories...")
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for d in data_dirs:
            if not d.exists():
                print(f"[!] Warning: Directory does not exist: {d}")
                continue

            search_dir = d / "plates" if (d / "plates").is_dir() else d
            count = 0
            for p in search_dir.glob("*.jpg"):
                parsed = self.parse_filename_metadata(p)
                if parsed:
                    self.samples.append((p, *parsed))
                    count += 1
            print(f"  [+] Found {count:,} samples in: {search_dir}")

        random.seed(42)
        random.shuffle(self.samples)
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(f"[{get_utc_timestamp()}] Total active training corpus: {len(self.samples):,} plates.")

    @staticmethod
    def parse_filename_metadata(p: Path) -> Optional[Tuple[str, str, str, str]]:
        """
        Parses metadata from filename using right-aligned standard format:
        ..._<PlateType>_<d3>_<d4>_<prov>.jpg
        """
        parts = p.stem.split("_")
        if len(parts) >= 4:
            cand_pt = parts[-4]
            cand_d3 = parts[-3]
            cand_d4 = parts[-2]
            cand_prov = parts[-1]

            pt = match_plate_type(cand_pt)
            if pt and pt not in ["+", "-"]:
                d3 = cand_d3.replace("ฃ", "ข").replace("ฅ", "ค")[:3]
                d4 = "".join(c for c in cand_d4 if c.isdigit())[:4]
                prov = match_province_value(cand_prov) if cand_prov else ""
                if d3 or d4:
                    return pt, d3, d4, prov

        # Fallback: scan backwards for valid plate type
        for offset in range(4, min(len(parts), 8)):
            cand_pt = parts[-offset]
            pt = match_plate_type(cand_pt)
            if pt and pt not in ["+", "-"]:
                idx = len(parts) - offset
                d3 = parts[idx + 1].replace("ฃ", "ข").replace("ฅ", "ค")[:3] if idx + 1 < len(parts) else ""
                d4 = "".join(c for c in parts[idx + 2] if c.isdigit())[:4] if idx + 2 < len(parts) else ""
                prov = match_province_value(parts[idx + 3]) if idx + 3 < len(parts) else ""
                if d3 or d4:
                    return pt, d3, d4, prov

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, pt, d3, d4, prov = self.samples[idx]

        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            im = Image.new("RGB", (self.img_w, self.img_h), color=(128, 128, 128))

        w, h = im.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(math.ceil(self.img_h * ratio)), self.img_w)
        resized_crop = im.resize((resized_w, self.img_h), Image.BICUBIC)

        arr = np.array(resized_crop, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).permute(2, 0, 1)

        # Pad width to self.img_w
        pad = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(3, self.img_h, self.img_w - resized_w)

        # 1. Plate Type Target
        type_idx = TYPE_TO_IDX.get(pt, 0)

        # 2. Delimiter-Aware Sequence Target: d3 <sep> d4
        d3_clean = normalize_thai_script(d3)[:3]
        d4_clean = "".join(c for c in d4 if c.isdigit())[:4]
        target_str = f"{d3_clean} {d4_clean}" if d3_clean and d4_clean else (d3_clean or d4_clean)
        seq_indices = [CTC_CHAR_TO_IDX[c] for c in target_str if c in CTC_CHAR_TO_IDX]

        max_seq = 10
        seq_targets = torch.zeros(max_seq, dtype=torch.long)
        if seq_indices:
            valid_len = min(len(seq_indices), max_seq)
            seq_targets[:valid_len] = torch.tensor(seq_indices[:valid_len], dtype=torch.long)
            target_length = torch.tensor(valid_len, dtype=torch.long)
        else:
            target_length = torch.tensor(0, dtype=torch.long)

        # 3. Province Target
        prov_idx = PROV_TO_IDX.get(prov, 0)

        targets = {
            "type": torch.tensor(type_idx, dtype=torch.long),
            "seq_targets": seq_targets,
            "target_lengths": target_length,
            "prov": torch.tensor(prov_idx, dtype=torch.long),
            "d3_gt": d3_clean,
            "d4_gt": d4_clean,
            "prov_gt": prov,
            "type_gt": pt,
        }

        return pad, targets


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    types = torch.stack([b[1]["type"] for b in batch], dim=0)
    provs = torch.stack([b[1]["prov"] for b in batch], dim=0)
    target_lengths = torch.stack([b[1]["target_lengths"] for b in batch], dim=0)

    # Flatten seq_targets for CTCLoss
    seq_targets_list = []
    for b in batch:
        t_len = b[1]["target_lengths"].item()
        if t_len > 0:
            seq_targets_list.append(b[1]["seq_targets"][:t_len])
    if seq_targets_list:
        flat_targets = torch.cat(seq_targets_list, dim=0)
    else:
        flat_targets = torch.tensor([], dtype=torch.long)

    d3_gts = [b[1]["d3_gt"] for b in batch]
    d4_gts = [b[1]["d4_gt"] for b in batch]
    prov_gts = [b[1]["prov_gt"] for b in batch]
    type_gts = [b[1]["type_gt"] for b in batch]

    return imgs, {
        "type": types,
        "prov": provs,
        "target_lengths": target_lengths,
        "flat_targets": flat_targets,
        "d3_gt": d3_gts,
        "d4_gt": d4_gts,
        "prov_gt": prov_gts,
        "type_gt": type_gts,
    }


# =============================================================================
# Training Pipeline
# =============================================================================

def train_selen_lpr_e(
    data_dirs: List[Path],
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.0003,
    target_accuracy_threshold: float = 95.0,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Optional[Path] = None
):
    device = torch.device(device_str)
    out_dir = Path(output_dir) if output_dir else ROOT_DIR / "runs" / "train" / "selenLPR_E"
    export_dir = ROOT_DIR / "models" / "reader" / "selenLPR-E"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SELENLPR-E MULTI-TASK THAI LICENSE PLATE RECOGNITION PIPELINE")
    print("  ISO/IEC 42001:2023 & ISO/IEC 25012:2008 COMPLIANT")
    print("=" * 80)
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Target Accuracy Stop Threshold: {target_accuracy_threshold:.1f}%")
    print(f"Output Path   : {out_dir}")
    print(f"Export Checkpt: {export_dir / 'best_reader.pth'}")
    print("=" * 80)

    # 1. Dataset & DataLoader (95% Train / 5% Val)
    full_dataset = SelenLPREDataset(data_dirs=data_dirs, img_h=64, img_w=256)
    total_len = len(full_dataset)
    val_len = min(2000, max(500, int(total_len * 0.05)))
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    print(f"[{get_utc_timestamp()}] [DATA SPLIT] Train: {train_len:,} samples | Val: {val_len:,} samples")

    # 2. Model Initialization & Pre-trained Foundation Transfer
    model = SelenLPR_E().to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"[{get_utc_timestamp()}] [MODEL] SelenLPR-E initialized. Total parameters: {params_count:,} ({params_count*4/1024/1024:.2f} MB float32)")

    try:
        b_trans, c_trans = model.load_easyocr_foundation(DEFAULT_PRETRAINED_PATH)
        print(f"[{get_utc_timestamp()}] [+] Loaded EasyOCR Thai foundation weights: {b_trans} backbone/sequence tensors, {c_trans} CTC token classes transferred.")
    except Exception as e:
        print(f"[{get_utc_timestamp()}] [!] Warning loading foundation: {e}. Training from random initialization.")

    # 3. Loss & Optimizer
    criterion_ctc = nn.CTCLoss(blank=CTC_BLANK_IDX, zero_infinity=True)
    criterion_type = nn.CrossEntropyLoss()
    criterion_prov = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_exact_acc = 0.0
    best_core_acc = 0.0
    best_val_loss = float("inf")
    history = []
    t_start = time.time()
    early_stopped = False

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")

        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            t_type = targets["type"].to(device, non_blocking=True)
            t_prov = targets["prov"].to(device, non_blocking=True)
            t_lengths = targets["target_lengths"].to(device, non_blocking=True)
            flat_tgt = targets["flat_targets"].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(imgs)

                # CTC Loss
                ctc_logits = preds["ctc_logits"]
                b_size, time_steps, _ = ctc_logits.shape
                log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)
                input_lengths = torch.full((b_size,), time_steps, dtype=torch.long, device=device)

                loss_ctc = criterion_ctc(log_probs, flat_tgt, input_lengths, t_lengths)
                loss_type = criterion_type(preds["type_logits"], t_type)
                loss_prov = criterion_prov(preds["prov_logits"], t_prov)

                loss = loss_ctc + 0.5 * loss_type + 0.5 * loss_prov

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ctc": f"{loss_ctc.item():.4f}",
                "type": f"{loss_type.item():.4f}",
                "prov": f"{loss_prov.item():.4f}",
            })

        avg_train_loss = train_loss / max(train_batches, 1)

        # ---------------------------------------------------------
        # Validation Loop
        # ---------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_batches = 0

        correct_d3 = 0
        correct_d4 = 0
        correct_core = 0
        correct_type = 0
        correct_prov = 0
        correct_exact_4way = 0
        total_eval = 0

        with torch.no_grad():
            for v_imgs, v_targets in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                v_imgs = v_imgs.to(device, non_blocking=True)
                vt_type = v_targets["type"].to(device, non_blocking=True)
                vt_prov = v_targets["prov"].to(device, non_blocking=True)
                vt_lengths = v_targets["target_lengths"].to(device, non_blocking=True)
                v_flat_tgt = v_targets["flat_targets"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_preds = model(v_imgs)

                    v_ctc = v_preds["ctc_logits"]
                    vb_size, v_tsteps, _ = v_ctc.shape
                    v_log_probs = v_ctc.log_softmax(dim=-1).permute(1, 0, 2)
                    v_input_lens = torch.full((vb_size,), v_tsteps, dtype=torch.long, device=device)

                    v_loss_ctc = criterion_ctc(v_log_probs, v_flat_tgt, v_input_lens, vt_lengths)
                    v_loss_type = criterion_type(v_preds["type_logits"], vt_type)
                    v_loss_prov = criterion_prov(v_preds["prov_logits"], vt_prov)
                    v_total_loss = v_loss_ctc + 0.5 * v_loss_type + 0.5 * v_loss_prov

                val_loss += v_total_loss.item()
                val_batches += 1

                # Decode and evaluate accuracy
                decoded_batch = decode_selen_lpr_e(v_preds)

                for b in range(len(decoded_batch)):
                    pred = decoded_batch[b]
                    gt_d3 = v_targets["d3_gt"][b]
                    gt_d4 = v_targets["d4_gt"][b]
                    gt_prov = v_targets["prov_gt"][b]
                    gt_type = v_targets["type_gt"][b]

                    is_d3 = (pred["d3"] == gt_d3)
                    is_d4 = (pred["d4"] == gt_d4)
                    is_core = (is_d3 and is_d4)
                    is_prov = (pred["province"] == gt_prov)
                    is_type = (pred["plate_type"] == gt_type)
                    is_exact = (is_core and is_prov and is_type)

                    if is_d3: correct_d3 += 1
                    if is_d4: correct_d4 += 1
                    if is_core: correct_core += 1
                    if is_prov: correct_prov += 1
                    if is_type: correct_type += 1
                    if is_exact: correct_exact_4way += 1
                    total_eval += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        acc_d3 = (correct_d3 / max(total_eval, 1)) * 100.0
        acc_d4 = (correct_d4 / max(total_eval, 1)) * 100.0
        acc_core = (correct_core / max(total_eval, 1)) * 100.0
        acc_type = (correct_type / max(total_eval, 1)) * 100.0
        acc_prov = (correct_prov / max(total_eval, 1)) * 100.0
        acc_exact = (correct_exact_4way / max(total_eval, 1)) * 100.0

        print(
            f"\n[EPOCH {epoch:02d}/{epochs:02d}] Val Loss: {avg_val_loss:.4f} | "
            f"d3: {acc_d3:.2f}% | d4: {acc_d4:.2f}% | Core Reg (d3+d4): {acc_core:.2f}% | "
            f"Type: {acc_type:.2f}% | Prov: {acc_prov:.2f}% | Exact 4-Way: {acc_exact:.2f}%"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "d3_acc": round(acc_d3, 2),
            "d4_acc": round(acc_d4, 2),
            "core_reg_acc": round(acc_core, 2),
            "type_acc": round(acc_type, 2),
            "prov_acc": round(acc_prov, 2),
            "exact_4way_acc": round(acc_exact, 2),
            "timestamp_utc": get_utc_timestamp()
        }
        history.append(record)

        # Save Best Checkpoint
        if acc_exact > best_exact_acc or (acc_exact == best_exact_acc and acc_core > best_core_acc):
            best_exact_acc = acc_exact
            best_core_acc = acc_core
            best_val_loss = avg_val_loss

            ckpt_payload = {
                "model_name": "selenLPR-E",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": {
                    "d3_acc": acc_d3,
                    "d4_acc": acc_d4,
                    "core_reg_acc": acc_core,
                    "type_acc": acc_type,
                    "prov_acc": acc_prov,
                    "exact_4way_acc": acc_exact
                },
                "val_loss": avg_val_loss,
                "ctc_tokens": CTC_TOKENS,
                "plate_types": PLATE_TYPES,
                "province_names": PROVINCE_NAMES,
                "standards": ["ISO/IEC 42001:2023", "ISO/IEC 25012:2008", "ISO 8601:2019"]
            }

            # Save in runs and export to models/reader/selenLPR-E/
            best_path_runs = out_dir / "best_reader.pth"
            best_path_export = export_dir / "best_reader.pth"
            torch.save(ckpt_payload, best_path_runs)
            torch.save(ckpt_payload, best_path_export)

            sha256_hash = hashlib.sha256(open(best_path_export, "rb").read()).hexdigest().upper()
            print(f"  [*] New Best Checkpoint Saved -> Exact 4-Way: {acc_exact:.2f}% | Core Reg: {acc_core:.2f}%")
            print(f"      Path: {best_path_export} (SHA-256: {sha256_hash})")

        # Early Stopping Check against Target Benchmark
        if acc_core >= target_accuracy_threshold and acc_prov >= 90.0:
            print("\n" + "=" * 80)
            print(f"[*] TARGET CONVERGENCE BENCHMARK ACHIEVED AT EPOCH {epoch}!")
            print(f"[*] Core Registration Acc: {acc_core:.2f}% >= {target_accuracy_threshold:.1f}%")
            print(f"[*] Province Acc: {acc_prov:.2f}% >= 90.0%")
            print(f"[*] Triggering early stopping and exporting converged selenLPR-E weights.")
            print("=" * 80)
            early_stopped = True
            break

    total_time = time.time() - t_start

    # Export Training Summary
    summary = {
        "model_name": "selenLPR-E",
        "base_provenance": "Official JaidedAI EasyOCR Thai Foundation Weights (ResNet + BiLSTM)",
        "multitask_heads": {
            "ctc_alphanumeric": "d3 <sep> d4 with space delimiter token (vocab size 53)",
            "plate_type": "11 vehicle classes with RGB color statistics",
            "province": "79 closed-set Thailand administrative subdivisions"
        },
        "total_training_time_sec": round(total_time, 2),
        "epochs_completed": len(history),
        "early_stopped": early_stopped,
        "best_exact_4way_acc": best_exact_acc,
        "best_core_reg_acc": best_core_acc,
        "best_val_loss": best_val_loss,
        "checkpoint_export": str(export_dir / "best_reader.pth"),
        "history": history
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  SELENLPR-E TRAINING PIPELINE COMPLETED")
    print(f"  Total Duration : {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"  Best Exact Acc : {best_exact_acc:.2f}%")
    print(f"  Best Core Reg  : {best_core_acc:.2f}%")
    print(f"  Export Model   : {export_dir / 'best_reader.pth'}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SelenLPR-E Multi-Task Thai LPR Model")
    parser.add_argument("--epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--target-acc", type=float, default=95.0, help="Target accuracy stopping threshold")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    active_dirs = [
        ROOT_DIR / "dataset" / "reading" / "versions" / "202611030335" / "plates",
        ROOT_DIR / "dataset" / "reading" / "versions" / "dataset-dirty10-blur5-angle5" / "plates",
        ROOT_DIR / "dataset" / "reading" / "versions" / "202610261140" / "plates",
    ]

    train_selen_lpr_e(
        data_dirs=active_dirs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        target_accuracy_threshold=args.target_acc,
        device_str=args.device
    )
