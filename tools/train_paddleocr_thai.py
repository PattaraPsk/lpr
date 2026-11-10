#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR-Style / SVTR Multi-Task Thai License Plate Recognition Model
(selen-svtr-singlechar-0.0.1)

Pipeline Highlights:
1. Architecture:
   - MobileNetV3 + Transformer Context + 2-Layer BiLSTM Sequence Neck + CTC Head.
   - Plate Type Head: 11 vehicle classes.
   - Sequence Head (CTC): 130 classes (0=<pad>/blank, 1..10=digits, 11..51=Thai consonants,
     52..129=78 Thai Provinces encoded into single distinct Unicode characters U+E000..U+E04D).
   - Province Multi-Task Classifier Head: 79 classes for joint multi-task representation.
2. Training:
   - 50 Epochs with PyTorch AMP FP16 mixed precision on RTX 4090.
   - Multi-Task Loss: nn.CTCLoss (sequence) + Focal Loss / CrossEntropy (plate_type, prov).
   - Evaluation: Independent separate accuracy tracking for plate_type, d3, d4, provinces, and exact plate.
3. Dataset:
   - dataset/ready/202611030230/ + previous synthetic datasets (dataset-cctv-synthetic & dataset-mobile-synthetic).
   - Deduplicated across sources with robust metadata parsing.

Standards Compliance:
- ISO/IEC 42001:2023 (AI Management System - Clauses 7.5, 8.4, Annex A.6.7, A.9.3)
- ISO 8601:2019 (Date and Time Representation)
- ISO/IEC 25012:2008 (Data Quality and Integrity)
- DLT Thailand Standard Vehicle & License Plate Specifications
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
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.models as models
from tqdm import tqdm

# Add repository root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 standard output on Windows systems
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.provinces import PROVINCES_DATA, normalize_thai_script, match_province_value
from tools.DataManipulator.app.plate_types import PLATE_TYPES_DATA, match_plate_type


def get_utc_timestamp() -> str:
    """Return current ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


# =============================================================================
# Vocabularies & Single-Character Province Mapping
# =============================================================================

# 1. Plate Types (11 classes)
PLATE_TYPES = [
    'Other', 'WhiteBlackCar', 'WhiteBlackMotor', 'WhiteBlue', 'WhiteGreen',
    'YellowBlack', 'YellowBlue', 'YellowGreen', 'GreenWhite', 'RedBlack', 'OrangeBlack'
]
TYPE_TO_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(PLATE_TYPES)}

# 2. 78 Jurisdictions (77 provinces + Betong)
PROVINCES_78 = [p[0] for p in PROVINCES_DATA]
if 'เบตง' not in PROVINCES_78:
    PROVINCES_78.append('เบตง')

# Province names with unknown (79 classes for auxiliary prov classification head)
PROVINCE_NAMES_79 = ['<unknown>'] + PROVINCES_78
PROV_TO_AUX_IDX = {p: i for i, p in enumerate(PROVINCE_NAMES_79)}
AUX_IDX_TO_PROV = {i: p for i, p in enumerate(PROVINCE_NAMES_79)}

# Single-character province mapping using Unicode Private Use Area (U+E000 to U+E04D)
PROV_TO_CHAR = {p: chr(0xE000 + i) for i, p in enumerate(PROVINCES_78)}
CHAR_TO_PROV = {chr(0xE000 + i): p for i, p in enumerate(PROVINCES_78)}
PROV_CHARS = [chr(0xE000 + i) for i in range(len(PROVINCES_78))]

# 3. Unified CTC Vocabulary (130 classes)
# Index 0: <pad> / blank
# Index 1..10: '0'..'9'
# Index 11..51: 41 DLT Thai Consonants
# Index 52..129: 78 Single-Character Province Tokens
CTC_BLANK_IDX = 0
THAI_CONSONANTS = 'กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
CTC_CHARS = ['<pad>'] + [str(d) for d in range(10)] + list(THAI_CONSONANTS) + PROV_CHARS
CTC_CHAR_TO_IDX = {c: i for i, c in enumerate(CTC_CHARS)}
CTC_IDX_TO_CHAR = {i: c for i, c in enumerate(CTC_CHARS)}
VOCAB_SIZE = len(CTC_CHARS)  # 130


# =============================================================================
# Decoding & Post-Processing Functions
# =============================================================================

def decode_and_split_ctc_single_char_prov(
    logits: torch.Tensor,
    blank_idx: int = CTC_BLANK_IDX
) -> List[Tuple[str, str, str]]:
    """
    Greedy CTC decode (Batch, TimeSteps, Vocab) -> list of (d3, d4, province_name) tuples.
    1. Collapses consecutive duplicate indices and strips blank index.
    2. Identifies and extracts single-character province token (0xE000..0xE04D).
    3. Splits remaining alphanumeric string at the last Thai consonant into d3 and d4.
    """
    preds = logits.argmax(dim=-1).cpu().tolist()
    results: List[Tuple[str, str, str]] = []

    for seq in preds:
        collapsed = [k for i, k in enumerate(seq) if k != blank_idx and (i == 0 or k != seq[i - 1])]
        full_str = "".join(CTC_IDX_TO_CHAR.get(k, "") for k in collapsed)

        prov_found = ""
        reg_chars = []
        for ch in full_str:
            if 0xE000 <= ord(ch) < 0xE000 + len(PROVINCES_78):
                prov_found = CHAR_TO_PROV.get(ch, "")
            else:
                reg_chars.append(ch)

        reg_str = "".join(reg_chars)

        last_thai_idx = -1
        for i, ch in enumerate(reg_str):
            if "\u0e01" <= ch <= "\u0e2e":
                last_thai_idx = i

        if last_thai_idx != -1:
            d3 = reg_str[:last_thai_idx + 1]
            d4 = reg_str[last_thai_idx + 1:]
        else:
            d3 = ""
            d4 = reg_str

        results.append((d3, d4, prov_found))

    return results


# =============================================================================
# Model Architecture: LeftToRightCTCHead & SelenPaddleOCR
# =============================================================================

class LeftToRightCTCHead(nn.Module):
    """
    CTC Sequence Recognition Head with 2-layer Bidirectional LSTM Neck.
    Adapts 2D CNN feature map (B, C, H, W) to (B, TimeSteps, VocabSize).
    """
    def __init__(
        self,
        in_channels: int = 256,
        d_model: int = 256,
        hidden_size: Optional[int] = None,
        vocab_size: int = VOCAB_SIZE,
        time_steps: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, time_steps))

        if hidden_size is None:
            hidden_size = d_model // 2  # 128 (128 * 2 = 256)
        self.hidden_size = hidden_size

        self.neck = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2
        self.classifier = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        pooled = self.spatial_pool(feat_2d)
        x = pooled.squeeze(2).permute(0, 2, 1).contiguous()
        if hasattr(self.neck, "flatten_parameters"):
            try:
                self.neck.flatten_parameters()
            except Exception:
                pass
        recurrent, _ = self.neck(x)
        return self.classifier(recurrent)


class SelenPaddleOCR(nn.Module):
    """
    Unified Multi-Task MobileNetV3 + Transformer + 2-Layer BiLSTM CTC Thai LPR Architecture.
    Predicts:
    1. Plate Type (11 classes)
    2. Province Auxiliary Classifier (79 classes)
    3. CTC Sequence Head (130 classes: 0=pad, digits, Thai consonants, 78 single-char provinces)
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        vocab_size: int = VOCAB_SIZE,
        ctc_hidden_size: Optional[int] = None,
        ctc_num_layers: int = 2,
        time_steps: int = 32,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Vision Backbone (MobileNetV3-Small)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features  # (B, 576, H/32, W/32)
        self.proj = nn.Conv2d(576, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

        # 2. Learnable Positional Encoding & Transformer Encoder
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 3. Head 1: Plate Type Classifier (11 classes)
        self.type_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, len(PLATE_TYPES))
        )

        # 4. Head 2: CTC Sequence Head (130 classes)
        self.ctc_head = LeftToRightCTCHead(
            in_channels=d_model,
            d_model=d_model,
            hidden_size=ctc_hidden_size,
            vocab_size=vocab_size,
            time_steps=time_steps,
            num_layers=ctc_num_layers,
        )

        # 5. Head 3: Province Auxiliary Decoder (79 classes)
        self.prov_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        prov_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.prov_decoder = nn.TransformerDecoder(prov_dec_layer, num_layers=1)
        self.prov_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, len(PROVINCE_NAMES_79))
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, _, _, _ = x.shape
        feat_2d = self.proj(self.backbone(x))  # (B, d_model, H', W')
        b, c, h, w = feat_2d.shape

        # Sequence context representation
        feat_seq = feat_2d.flatten(2).permute(0, 2, 1)  # (B, H'*W', d_model)
        feat_seq = self.norm(feat_seq)
        seq_len = feat_seq.shape[1]
        feat_seq = feat_seq + self.pos_encoder[:, :seq_len, :]
        memory = self.encoder(feat_seq)

        # Global average pool for plate type
        type_feat = memory.mean(dim=1)
        type_logits = self.type_head(type_feat)

        # Province auxiliary classification
        prov_q = self.prov_query.expand(b, -1, -1)
        prov_feat = self.prov_decoder(tgt=prov_q, memory=memory).squeeze(1)
        prov_logits = self.prov_head(prov_feat)

        # CTC Sequence recognition
        ctc_logits = self.ctc_head(feat_2d)

        return {
            "type_logits": type_logits,
            "ctc_logits": ctc_logits,
            "prov_logits": prov_logits
        }


# =============================================================================
# Multi-Task Loss Function
# =============================================================================

class MultiTaskCTCLoss(nn.Module):
    """
    Combined Multi-Task Loss optimizing:
    - Sequence CTC Loss (weight 1.0)
    - Plate Type Cross-Entropy (weight 0.2)
    - Province Cross-Entropy (weight 0.2)
    """
    def __init__(self, blank_idx: int = CTC_BLANK_IDX):
        super().__init__()
        self.blank_idx = blank_idx
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
        self.type_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.prov_loss = nn.CrossEntropyLoss(label_smoothing=0.05)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctc_logits = preds["ctc_logits"]  # (B, TimeSteps, Vocab)
        log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)  # (TimeSteps, B, Vocab)

        seq_targets = targets["seq_targets"]
        target_lengths = targets["target_lengths"]
        b, t, v = ctc_logits.shape
        input_lengths = torch.full((b,), t, dtype=torch.long, device=ctc_logits.device)

        loss_ctc = self.ctc_loss(log_probs, seq_targets, input_lengths, target_lengths)
        loss_type = self.type_loss(preds["type_logits"], targets["type"])
        loss_prov = self.prov_loss(preds["prov_logits"], targets["prov"])

        total_loss = loss_ctc + 0.2 * loss_type + 0.2 * loss_prov

        return {
            "loss": total_loss,
            "loss_ctc": loss_ctc,
            "loss_type": loss_type,
            "loss_prov": loss_prov
        }


# =============================================================================
# Dataset Loader
# =============================================================================

class SingleCharProvPlateDataset(Dataset):
    """
    Deduplicated Multi-Task Plate Dataset with Single-Character Province Encoding.
    Target sequence: [d3 characters] + [d4 digits] + [single province character].
    """
    def __init__(self, data_dirs: List[Path], img_h: int = 64, img_w: int = 384, max_seq_len: int = 32):
        self.img_h = img_h
        self.img_w = img_w
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[Path, str, str, str, str]] = []

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        seen_filenames = {}

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

                    # Deduplicate by filename
                    fname = p.name
                    if fname in seen_filenames:
                        continue

                    parsed = self.parse_metadata_robust(p)
                    if parsed:
                        pt, d3, d4, prov = parsed
                        if d3 or d4:
                            seen_filenames[fname] = p
                            self.samples.append((p, pt, d3, d4, prov))

        print(f"[{get_utc_timestamp()}] [DATASET] Loaded {len(self.samples):,} unique multi-task samples.")

    @staticmethod
    def parse_metadata_robust(p: Path) -> Optional[Tuple[str, str, str, str]]:
        cand_strings = [p.parent.name, p.stem]
        for s in cand_strings:
            parts = s.split("_")
            for idx in range(len(parts) - 1, -1, -1):
                prov_match = match_province_value(parts[idx])
                if prov_match:
                    if idx >= 2:
                        d4 = parts[idx - 1]
                        d3 = parts[idx - 2]
                        pt = parts[idx - 3] if idx >= 3 else "WhiteBlackCar"

                        if d3 in ["+", "-"]: d3 = ""
                        if d4 in ["+", "-"]: d4 = ""
                        matched_pt = match_plate_type(pt)
                        if not matched_pt or matched_pt in ["+", "-"]:
                            matched_pt = "WhiteBlackCar"

                        d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค")
                        return matched_pt, d3, d4, prov_match
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

        pad = torch.zeros(3, self.img_h, self.img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if self.img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(3, self.img_h, self.img_w - resized_w)

        # 1. Plate Type Target
        type_idx = TYPE_TO_IDX.get(pt, 0)

        # 2. Unified CTC Sequence Target: d3 + d4 + single-character province
        d3_clean = normalize_thai_script(d3)
        d3_indices = [CTC_CHAR_TO_IDX[c] for c in d3_clean if c in CTC_CHAR_TO_IDX and c != "<pad>"]
        d4_indices = [CTC_CHAR_TO_IDX[c] for c in d4 if c in CTC_CHAR_TO_IDX and c != "<pad>"]

        prov_indices = []
        if prov in PROV_TO_CHAR:
            prov_char = PROV_TO_CHAR[prov]
            if prov_char in CTC_CHAR_TO_IDX:
                prov_indices.append(CTC_CHAR_TO_IDX[prov_char])

        full_indices = d3_indices + d4_indices + prov_indices
        seq_len = len(full_indices)

        seq_targets = torch.zeros(self.max_seq_len, dtype=torch.long)
        if seq_len > 0:
            valid_len = min(seq_len, self.max_seq_len)
            seq_targets[:valid_len] = torch.tensor(full_indices[:valid_len], dtype=torch.long)
            target_length = torch.tensor(valid_len, dtype=torch.long)
        else:
            target_length = torch.tensor(0, dtype=torch.long)

        # 3. Province Auxiliary Classifier Target (79 classes)
        prov_aux_idx = PROV_TO_AUX_IDX.get(prov, 0)

        targets = {
            "type": torch.tensor(type_idx, dtype=torch.long),
            "seq_targets": seq_targets,
            "target_lengths": target_length,
            "prov": torch.tensor(prov_aux_idx, dtype=torch.long),
        }
        return pad, targets


# =============================================================================
# Training & Verification Pipeline
# =============================================================================

def train_paddleocr_thai(
    data_dir: str = "",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.0005,
    device: str = "cuda:0",
    resume: str = "",
    dry_run: bool = False,
    output_dir: str = "models/reader/paddleocr_thai",
    target_exact_acc: float = 0.0
):
    print("=" * 80)
    print(f"[{get_utc_timestamp()}] PADDLEOCR / SVTR SINGLE-CHARACTER PROVINCE LPR PIPELINE")
    print("=" * 80)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / "best_reader.pth"

    if data_dir:
        data_directories = [Path(data_dir).resolve()]
    else:
        # Default to latest 202611* ready dataset
        ready_root = ROOT_DIR / "dataset/ready"
        candidates = sorted([p for p in ready_root.glob("202611*") if p.is_dir()], reverse=True)
        if candidates:
            data_directories = [candidates[0]]
        else:
            data_directories = [ready_root / "202610261140"]

    device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Dry Run Mode  : {dry_run}")
    print(f"Vocab Size    : {VOCAB_SIZE} (Single-Char Encoded Provinces)")
    if target_exact_acc > 0:
        print(f"Early Stop    : Exact Plate > {target_exact_acc:.1f}%")
    print(f"Output Dir    : {out_dir}")
    print("=" * 80)

    # 1. Dataset & DataLoader (95% Train / 5% Validation)
    full_dataset = SingleCharProvPlateDataset(data_dirs=data_directories, img_h=64, img_w=384, max_seq_len=32)
    total_len = len(full_dataset)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    if dry_run:
        print(f"[{get_utc_timestamp()}] [DRY-RUN] Slicing minimal dataset for rapid 1-epoch execution.")
        train_ds = Subset(train_ds, range(min(128, len(train_ds))))
        val_ds = Subset(val_ds, range(min(64, len(val_ds))))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"[{get_utc_timestamp()}] [DATA SPLIT] Train: {train_len:,} samples | Val: {val_len:,} samples")

    # 2. Model Initialization
    model = SelenPaddleOCR(d_model=256, nhead=4, vocab_size=VOCAB_SIZE, time_steps=32).to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"[{get_utc_timestamp()}] [MODEL] SelenPaddleOCR Initialized. Total Params: {params_count:,} ({params_count*4/1024/1024:.2f} MB float32)")

    # 3. Checkpoint Transfer
    base_ckpt_path = Path(resume if resume else "models/reader/paddleocr_thai/best_reader.pth").resolve()
    if not base_ckpt_path.is_file():
        base_ckpt_path = Path("models/reader/selen-lpr-0.0.1/best_reader.pth").resolve()

    if base_ckpt_path.is_file():
        print(f"[{get_utc_timestamp()}] [*] Loading checkpoint base: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Transfer compatible weights
        for k in list(state_dict.keys()):
            if "ctc_head.classifier" in k and state_dict[k].shape != model.state_dict()[k].shape:
                # Shape changed from legacy 52 to 130
                del state_dict[k]
            elif k in model.state_dict() and state_dict[k].shape != model.state_dict()[k].shape:
                del state_dict[k]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[{get_utc_timestamp()}] [+] Loaded model weights (missing={len(missing)}, unexpected={len(unexpected)}).")

        # Copy existing 52 character weights if upgrading from 52-class checkpoint
        if "ctc_head.classifier.weight" in ckpt and ckpt["ctc_head.classifier.weight"].shape[0] == 52:
            with torch.no_grad():
                model.ctc_head.classifier.weight[:52] = ckpt["ctc_head.classifier.weight"]
                model.ctc_head.classifier.bias[:52] = ckpt["ctc_head.classifier.bias"]
            print(f"[{get_utc_timestamp()}] [INIT] Successfully transferred initial 52 character weights to 130-class classifier.")

    # 4. Optimization Setup
    criterion = MultiTaskCTCLoss(blank_idx=CTC_BLANK_IDX).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
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
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(images)
                loss_dict = criterion(preds, targets)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ctc": f"{loss_dict['loss_ctc'].item():.4f}"
            })

        avg_train_loss = train_loss / max(train_batches, 1)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_batches = 0

        correct_type = 0
        correct_d3 = 0
        correct_d4 = 0
        correct_prov = 0
        correct_exact = 0
        total_eval = 0

        with torch.no_grad():
            for val_img, val_tgt in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                val_img = val_img.to(device, non_blocking=True)
                v_batch = {k: v.to(device, non_blocking=True) for k, v in val_tgt.items()}

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    v_preds = model(val_img)
                    v_loss_dict = criterion(v_preds, v_batch)

                val_loss += v_loss_dict["loss"].item()
                val_batches += 1

                # Decode predictions
                pred_type = v_preds["type_logits"].argmax(dim=-1)
                pred_tuples = decode_and_split_ctc_single_char_prov(v_preds["ctc_logits"], blank_idx=CTC_BLANK_IDX)

                for b in range(val_img.shape[0]):
                    is_t = (pred_type[b] == v_batch["type"][b]).item()

                    # Ground truth reconstruction from seq_targets
                    tgt_len = v_batch["target_lengths"][b].item()
                    gt_tokens = v_batch["seq_targets"][b][:tgt_len].cpu().tolist()
                    gt_full = "".join(CTC_IDX_TO_CHAR.get(k, "") for k in gt_tokens)

                    gt_prov = ""
                    gt_reg_chars = []
                    for ch in gt_full:
                        if 0xE000 <= ord(ch) < 0xE000 + len(PROVINCES_78):
                            gt_prov = CHAR_TO_PROV.get(ch, "")
                        else:
                            gt_reg_chars.append(ch)

                    gt_reg_str = "".join(gt_reg_chars)
                    last_thai_idx = -1
                    for i, ch in enumerate(gt_reg_str):
                        if "\u0e01" <= ch <= "\u0e2e":
                            last_thai_idx = i

                    if last_thai_idx != -1:
                        gt_d3 = gt_reg_str[:last_thai_idx + 1]
                        gt_d4 = gt_reg_str[last_thai_idx + 1:]
                    else:
                        gt_d3 = ""
                        gt_d4 = gt_reg_str

                    pred_d3, pred_d4, pred_prov = pred_tuples[b]

                    # Fallback to auxiliary prov head if CTC prov wasn't predicted
                    if not pred_prov:
                        aux_idx = v_preds["prov_logits"][b].argmax().item()
                        pred_prov = AUX_IDX_TO_PROV.get(aux_idx, "")

                    is_3 = (pred_d3 == gt_d3)
                    is_4 = (pred_d4 == gt_d4)
                    is_p = (pred_prov == gt_prov)

                    if is_t: correct_type += 1
                    if is_3: correct_d3 += 1
                    if is_4: correct_d4 += 1
                    if is_p: correct_prov += 1
                    if is_t and is_3 and is_4 and is_p:
                        correct_exact += 1
                    total_eval += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        acc_type = (correct_type / max(total_eval, 1)) * 100.0
        acc_d3 = (correct_d3 / max(total_eval, 1)) * 100.0
        acc_d4 = (correct_d4 / max(total_eval, 1)) * 100.0
        acc_prov = (correct_prov / max(total_eval, 1)) * 100.0
        acc_exact = (correct_exact / max(total_eval, 1)) * 100.0

        print(f"\n[EPOCH {epoch:02d}/{epochs:02d}] Val Loss: {avg_val_loss:.4f} | "
              f"Plate Type: {acc_type:.2f}% | d3: {acc_d3:.2f}% | d4: {acc_d4:.2f}% | Provinces: {acc_prov:.2f}% | "
              f"Exact Plate: {acc_exact:.2f}% ({correct_exact}/{total_eval})")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "acc_plate_type": round(acc_type, 2),
            "acc_d3": round(acc_d3, 2),
            "acc_d4": round(acc_d4, 2),
            "acc_provinces": round(acc_prov, 2),
            "acc_exact_plate": round(acc_exact, 2)
        }
        history.append(epoch_record)

        if not dry_run:
            torch.save(model.state_dict(), out_dir / "last_reader.pth")
            if avg_val_loss < best_val_loss or acc_exact > best_exact_acc:
                best_val_loss = min(avg_val_loss, best_val_loss)
                best_exact_acc = max(acc_exact, best_exact_acc)
                torch.save(model.state_dict(), export_path)
                print(f"[{get_utc_timestamp()}] [SAVE] New best checkpoint saved to: {export_path}")
        else:
            best_val_loss = min(avg_val_loss, best_val_loss)
            best_exact_acc = max(acc_exact, best_exact_acc)
            torch.save(model.state_dict(), out_dir / "dry_run_reader.pth")
            print(f"[{get_utc_timestamp()}] [DRY-RUN] Verified checkpoint saved to: {out_dir / 'dry_run_reader.pth'}")

        # Early stopping when target exact accuracy is reached
        if target_exact_acc > 0 and acc_exact >= target_exact_acc:
            print(f"\n[{get_utc_timestamp()}] [EARLY STOP] Target Exact Plate accuracy reached: {acc_exact:.2f}% >= {target_exact_acc:.1f}%! Halting training gracefully.")
            break

    total_time = time.time() - t_start
    best_hash = calculate_sha256(export_path) if export_path.is_file() else "N/A"

    summary = {
        "model_name": "selen-paddleocr-singlechar-0.0.1",
        "architecture": "SelenPaddleOCR (MobileNetV3 + Transformer Context + 2-Layer BiLSTM LeftToRightCTCHead)",
        "loss_function": "MultiTaskCTCLoss (nn.CTCLoss + CrossEntropy)",
        "vocab_size": VOCAB_SIZE,
        "single_char_provinces": True,
        "parameters": params_count,
        "model_size_mb": round(params_count * 4 / (1024 * 1024), 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_loss": round(best_val_loss, 4),
        "best_exact_accuracy": round(best_exact_acc, 2),
        "sha256": best_hash,
        "history": history
    }

    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[{get_utc_timestamp()}] PADDLEOCR / SVTR TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total Time      : {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Best Val Loss   : {best_val_loss:.4f}")
    print(f"Best Exact Acc  : {best_exact_acc:.2f}%")
    print(f"Best Model Hash : {best_hash}")
    print(f"Summary Saved   : {out_dir / 'training_summary.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PaddleOCR / SVTR Style Model with Single-Character Province Encoding")
    parser.add_argument("--data-dir", type=str, default="", help="Path to ready dataset directory (default: latest 202611*)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run for verification")
    parser.add_argument("--output-dir", type=str, default="models/reader/paddleocr_thai")
    parser.add_argument("--target-exact-acc", type=float, default=0.0, help="Stop training when Exact Plate accuracy exceeds this threshold (e.g., 95.0)")
    args = parser.parse_args()

    train_paddleocr_thai(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        target_exact_acc=args.target_exact_acc
    )
