#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenLPR (selen-lpr-0.0.1) Left-to-Right CTC Thai License Plate Recognition Model.

This module implements a unified multi-task Thai LPR architecture.
The model uses a Left-to-Right Connectionist Temporal Classification (CTC) sequence head
to replace autoregressive character decoders.

Standards Compliance:
- ASD-STE100 (Simplified Technical English)
- ISO 24495-1:2023 (Plain Language)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System - Clauses 7.5, 8.4, Annex A.6.7, A.9.3)
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
    """Return the current ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


# =============================================================================
# Vocabularies & Mapping Tables
# =============================================================================

# 1. Plate Types (11 classes)
PLATE_TYPES = [
    'Other', 'WhiteBlackCar', 'WhiteBlackMotor', 'WhiteBlue', 'WhiteGreen',
    'YellowBlack', 'YellowBlue', 'YellowGreen', 'GreenWhite', 'RedBlack', 'OrangeBlack'
]
TYPE_TO_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
IDX_TO_TYPE = {i: t for i, t in enumerate(PLATE_TYPES)}

# 2. Unified Left-to-Right CTC Vocabulary (52 classes)
# Blank index 0 reuses <pad> index 0.
CTC_BLANK_IDX = 0
THAI_CONSONANTS = 'กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
CTC_CHARS = ['<pad>'] + [str(d) for d in range(10)] + list(THAI_CONSONANTS)
CTC_CHAR_TO_IDX = {c: i for i, c in enumerate(CTC_CHARS)}
CTC_IDX_TO_CHAR = {i: c for i, c in enumerate(CTC_CHARS)}
VOCAB_SIZE = len(CTC_CHARS)  # 52

# Legacy mappings retained for backward compatibility
D3_CHARS = ['<pad>', '<eos>'] + [str(d) for d in range(1, 10)] + list(THAI_CONSONANTS)
D3_TO_IDX = {c: i for i, c in enumerate(D3_CHARS)}
IDX_TO_D3 = {i: c for i, c in enumerate(D3_CHARS)}

D4_CHARS = ['<pad>', '<eos>'] + [str(d) for d in range(10)]
D4_TO_IDX = {c: i for i, c in enumerate(D4_CHARS)}
IDX_TO_D4 = {i: c for i, c in enumerate(D4_CHARS)}

# 3. Provinces (79 classes: 0=Unknown, 1..78=Provinces)
PROVINCE_NAMES = ['<unknown>'] + [p[0] for p in PROVINCES_DATA]
if 'เบตง' not in PROVINCE_NAMES:
    PROVINCE_NAMES.append('เบตง')
PROV_TO_IDX = {p: i for i, p in enumerate(PROVINCE_NAMES)}
IDX_TO_PROV = {i: p for i, p in enumerate(PROVINCE_NAMES)}


# =============================================================================
# Model Architecture: Left-to-Right CTC Sequence Head with BiLSTM Neck
# =============================================================================

class LeftToRightCTCHead(nn.Module):
    """
    Unified Left-to-Right CTC Sequence Prediction Head with 2-Layer BiLSTM Neck.
    Complies with ASD-STE100 and ISO 24495-1:2023 standards.
    Integrates a 2-layer Bidirectional LSTM neck between the CNN backbone features
    and the final linear CTC classifier, mimicking EasyOCR's CRNN architecture.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int = 256,
        hidden_size: Optional[int] = None,
        vocab_size: int = VOCAB_SIZE,
        time_steps: int = 14,
        num_layers: int = 2,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        # Mimic EasyOCR / CRNN design: bidirectional hidden size chosen such that
        # 2 * hidden_size matches or projects sequence features (default d_model // 2 = 128 -> 256)
        self.hidden_size = hidden_size if hidden_size is not None else (d_model // 2)

        # Pool vertical height to 1, preserve horizontal left-to-right sequence width W=28
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, time_steps))

        # 2-layer Bidirectional LSTM neck for spatial sequence modeling
        self.neck = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )

        # Final linear CTC classifier: maps BiLSTM features (hidden_size * 2) to CTC vocabulary
        lstm_out_dim = self.hidden_size * 2
        self.classifier = nn.Linear(lstm_out_dim, vocab_size)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass converting 2D backbone features to CTC sequence predictions.

        Args:
            feat_2d: Tensor of shape [batch_size, channels, height, width] from CNN backbone.

        Returns:
            Tensor of shape [batch_size, sequence_length, vocab_size] containing CTC logits.
        """
        # Crop top 75% of feature map height to focus on top-line characters (d3 + d4)
        h = feat_2d.size(2)
        top_feat = feat_2d[:, :, :max(1, int(h * 0.75)), :]

        # 1. Spatial pooling: collapse height to 1 and adapt sequence width to time_steps
        # Output shape: [batch_size, channels, 1, sequence_length]
        pooled = self.spatial_pool(top_feat)

        # 2. Dimension permutation:
        # [batch_size, channels, 1, sequence_length] -> squeeze(2) -> [batch_size, channels, sequence_length]
        # permute(0, 2, 1) -> [batch_size, sequence_length, features] required by batch_first=True LSTM
        x = pooled.squeeze(2).permute(0, 2, 1).contiguous()

        # 3. BiLSTM sequence neck modeling
        if hasattr(self.neck, "flatten_parameters"):
            try:
                self.neck.flatten_parameters()
            except Exception:
                pass
        recurrent, _ = self.neck(x)  # [batch_size, sequence_length, hidden_size * 2]

        # 4. Final linear CTC classification: [batch_size, sequence_length, vocab_size]
        return self.classifier(recurrent)


# =============================================================================
# Greedy CTC Decoder & Deterministic Split
# =============================================================================

def decode_and_split_ctc(logits: torch.Tensor, idx2char: Dict[int, str], blank_idx: int = 0) -> List[Tuple[str, str]]:
    """
    Greedy CTC decode (B, W, V) -> list of (d3_str, d4_str) pairs.
    1. Collapses consecutive duplicate indices and removes blank index.
    2. Deterministically splits at the last Thai consonant (U+0E01 to U+0E2E).
    3. Enforces DLT max lengths: d3 ≤ 3 chars, d4 ≤ 4 digits.
    """
    MAX_D3 = 3
    MAX_D4 = 4
    preds = logits.argmax(dim=-1).cpu().tolist()
    results: List[Tuple[str, str]] = []
    for seq in preds:
        # 1. Collapse repeats and drop blank
        collapsed = [k for i, k in enumerate(seq) if k != blank_idx and (i == 0 or k != seq[i - 1])]
        full_str = "".join(idx2char.get(k, "") for k in collapsed)

        # 2. Deterministic split at the last Thai consonant (U+0E01 to U+0E2E)
        last_thai_idx = -1
        for i, ch in enumerate(full_str):
            if "\u0e01" <= ch <= "\u0e2e":
                last_thai_idx = i
        if last_thai_idx != -1:
            d3 = full_str[:last_thai_idx + 1]
            d4 = full_str[last_thai_idx + 1:]
        else:
            d3 = ""
            d4 = full_str

        # 3. DLT length enforcement: clamp to max allowed lengths
        d3 = d3[:MAX_D3]
        d4 = d4[:MAX_D4]

        results.append((d3, d4))
    return results


# =============================================================================
# Model Architecture: SelenLPR (selen-lpr-0.0.1)
# =============================================================================

class SelenLPR(nn.Module):
    """
    Unified Multi-Task Edge Thai License Plate Recognition Model.
    - Vision Backbone: Pretrained MobileNetV3-Small (ImageNet-1K, frozen during fine-tuning).
    - Sequence Context: Multi-Head Self-Attention Transformer Encoder.
    - Plate Type Head: Linear classifier for DLT vehicle categories (structurally isolated, frozen).
    - Unified CTC Head: Left-to-right 2-layer BiLSTM neck + Linear CTC classifier (active).
    - Province Head: Multi-Head Cross-Attention decoder + Linear classifier (structurally isolated, frozen).
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        vocab_size: int = VOCAB_SIZE,
        ctc_hidden_size: Optional[int] = None,
        ctc_num_layers: int = 2,
        time_steps: int = 14,
    ):
        super().__init__()
        self.d_model = d_model
        self.time_steps = time_steps

        # 1. Vision Backbone (Frozen in fine-tuning)
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = mobilenet.features  # (B, 576, H/32, W/32)
        self.proj = nn.Conv2d(576, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)

        # 2. Learnable Spatial Positional Encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)

        # 3. Transformer Encoder (Self-Attention over visual sequence)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)

        # 4. Head 1: Vehicle Plate Type Head (11 classes) - Structurally Isolated
        self.type_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, len(PLATE_TYPES))
        )

        # 5. Head 2: Upgraded Left-to-Right CTC Sequence Head with 2-Layer BiLSTM Neck (W=14, Vocab=52)
        # Max output: d3(3) + d4(4) = 7 characters, CTC needs ~2x time steps = 14
        self.ctc_head = LeftToRightCTCHead(
            in_channels=d_model,
            d_model=d_model,
            hidden_size=ctc_hidden_size,
            vocab_size=vocab_size,
            time_steps=time_steps,
            num_layers=ctc_num_layers,
        )

        # Legacy Decoders (Bypassed in forward, preserved for strict=False loading)
        self.d3_emb = nn.Embedding(len(D3_CHARS), self.d_model)
        d3_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.d3_decoder = nn.TransformerDecoder(d3_dec_layer, num_layers=1)
        self.d3_head = nn.Linear(d_model, len(D3_CHARS))

        self.d4_emb = nn.Embedding(len(D4_CHARS), self.d_model)
        d4_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.d4_decoder = nn.TransformerDecoder(d4_dec_layer, num_layers=1)
        self.d4_head = nn.Linear(d_model, len(D4_CHARS))

        # 6. Head 3: Province Classification Head (79 classes) - Structurally Isolated
        self.prov_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        prov_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, activation="relu", batch_first=True
        )
        self.prov_decoder = nn.TransformerDecoder(prov_dec_layer, num_layers=1)
        self.prov_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, len(PROVINCE_NAMES))
        )

    @property
    def ctc_neck(self) -> nn.LSTM:
        """Access the 2-layer BiLSTM neck within the sequence pipeline."""
        return self.ctc_head.neck

    @property
    def ctc_classifier(self) -> nn.Linear:
        """Access the final linear CTC classifier within the sequence pipeline."""
        return self.ctc_head.classifier

    def forward(self, x: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b_size = x.shape[0]
        # Visual feature extraction & 1x1 projection from frozen CNN backbone: [B, d_model, H', W']
        feat_2d = self.proj(self.backbone(x))
        feats = feat_2d.mean(dim=2).permute(0, 2, 1) # (B, W', d_model)
        feats = self.norm(feats)

        w_seq = feats.shape[1]
        memory = self.encoder(feats + self.pos_encoder[:, :w_seq, :])

        # Head 1: Vehicle Type (Global average pooling from memory) - Structurally Isolated
        type_emb = memory.mean(dim=1)
        type_logits = self.type_head(type_emb)

        # Head 2: Upgraded Left-to-Right CTC Sequence Head with 2-Layer BiLSTM Neck
        # Handles dimension permutation [B, C, H, W] -> [B, W, C] and outputs [B, 28, Vocab]
        ctc_logits = self.ctc_head(feat_2d)

        # Head 3: Province classification (Cross-attention from memory) - Structurally Isolated
        q_prov = self.prov_query.expand(b_size, -1, -1)
        prov_out = self.prov_decoder(tgt=q_prov, memory=memory).squeeze(1)
        prov_logits = self.prov_head(prov_out)

        return {
            "type_logits": type_logits,
            "ctc_logits": ctc_logits,
            "prov_logits": prov_logits
        }


# =============================================================================
# Loss Function: nn.CTCLoss Configuration
# =============================================================================

class SelenMultiTaskLoss(nn.Module):
    """
    Multi-Task CTC Loss for Thai LPR (ASD-STE100 / ISO 24495-1:2023 Compliant).
    Uses nn.CTCLoss(blank=0, zero_infinity=True).
    Formats logits as (T, Batch, Vocab) using log_softmax(dim=-1).
    Sets total_loss = loss_ctc during the CTC sequence head training phase.
    """
    def __init__(self, blank_idx: int = CTC_BLANK_IDX):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctc_logits = preds["ctc_logits"]
        batch_size = ctc_logits.size(0)
        time_steps = ctc_logits.size(1)

        # Format logits as (T, Batch, Vocab) with log_softmax
        log_probs = ctc_logits.log_softmax(dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=ctc_logits.device)

        loss_ctc = self.ctc_loss(
            log_probs,
            targets["seq_targets"],
            input_lengths,
            targets["target_lengths"]
        )

        total_loss = loss_ctc
        return {
            "loss": total_loss,
            "loss_ctc": loss_ctc.detach()
        }


# =============================================================================
# Multi-Task Dataset Loader
# =============================================================================

class MultiTaskPlateDataset(Dataset):
    """
    Dataset loader for multi-task plate recognition.
    Extracts image, plate_type_idx, seq_targets, target_lengths, and prov_idx.
    Strips <SOS>, <EOS>, <PAD> and creates unified target sequences.
    """
    def __init__(self, data_dirs: List[Path], img_h: int = 64, img_w: int = 384, max_seq_len: int = 7):
        self.img_h = img_h
        self.img_w = img_w
        self.max_seq_len = max_seq_len
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

                    pt, d3, d4, prov = self.parse_metadata(p)
                    if d3 or d4:
                        self.samples.append((p, pt, d3, d4, prov))

        # Deduplicate samples
        seen = set()
        deduped = []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0])
                deduped.append(s)
        self.samples = deduped
        print(f"[{get_utc_timestamp()}] [DATASET] Loaded {len(self.samples):,} unique multi-task samples.")

    # Valid DLT character set for d3: digits 0-9 and 41 Thai consonants
    _VALID_D3_CHARS = set("0123456789" + THAI_CONSONANTS)
    # Valid DLT character set for d4: digits 0-9 only
    _VALID_D4_CHARS = set("0123456789")
    # Maximum lengths per DLT specification
    _MAX_D3_LEN = 3
    _MAX_D4_LEN = 4

    @staticmethod
    def parse_metadata(p: Path) -> Tuple[str, str, str, str]:
        """
        Parse plate metadata from folder name first, then filename fallback.
        Folder naming convention: <PlateType>_<d3>_<d4>_<Province>
        Filename convention:      <prefix>_<PlateType>_<d3>_<d4>_<Province>[_<suffix>].<ext>

        DLT Enforcement:
          - d3: max 3 characters (digits + 41 Thai consonants)
          - d4: max 4 characters (digits only)
        """
        plate_type = "WhiteBlackCar"
        d3 = ""
        d4 = ""
        prov = ""
        parsed = False

        # Strategy 1: Parse from parent folder name (authoritative, no suffix noise)
        folder_name = p.parent.name
        folder_parts = folder_name.split("_")
        if len(folder_parts) >= 4:
            # Folder format: <PlateType>_<d3>_<d4>_<Province>
            # Province may contain underscores (e.g., พระนครศรีอยุธยา), so rejoin tail
            candidate_pt = folder_parts[0]
            candidate_d3 = folder_parts[1]
            candidate_d4 = folder_parts[2]
            candidate_prov = "_".join(folder_parts[3:])

            # Validate: d3 must be ≤ 3 chars of valid DLT characters, d4 must be ≤ 4 digits
            d3_valid = (
                0 < len(candidate_d3) <= MultiTaskPlateDataset._MAX_D3_LEN
                and all(c in MultiTaskPlateDataset._VALID_D3_CHARS for c in candidate_d3)
            )
            d4_valid = (
                0 < len(candidate_d4) <= MultiTaskPlateDataset._MAX_D4_LEN
                and all(c in MultiTaskPlateDataset._VALID_D4_CHARS for c in candidate_d4)
            )
            if d3_valid or d4_valid:
                plate_type = candidate_pt
                d3 = candidate_d3 if d3_valid else ""
                d4 = candidate_d4 if d4_valid else ""
                prov = candidate_prov
                parsed = True
        elif len(folder_parts) == 3:
            # Folder format: <d3>_<d4>_<Province>
            candidate_d3, candidate_d4, candidate_prov = folder_parts[0], folder_parts[1], folder_parts[2]
            d3_valid = (
                0 < len(candidate_d3) <= MultiTaskPlateDataset._MAX_D3_LEN
                and all(c in MultiTaskPlateDataset._VALID_D3_CHARS for c in candidate_d3)
            )
            d4_valid = (
                0 < len(candidate_d4) <= MultiTaskPlateDataset._MAX_D4_LEN
                and all(c in MultiTaskPlateDataset._VALID_D4_CHARS for c in candidate_d4)
            )
            if d3_valid or d4_valid:
                d3 = candidate_d3 if d3_valid else ""
                d4 = candidate_d4 if d4_valid else ""
                prov = candidate_prov
                parsed = True

        # Strategy 2: Parse from filename stem (fallback, handle _suffix)
        if not parsed:
            parts = p.stem.split("_")
            # Try 5-part tail: ..._<PlateType>_<d3>_<d4>_<Province>_<suffix>
            if len(parts) >= 5:
                # Walk backwards looking for a valid plate_type token
                for offset in range(4, min(len(parts), 8)):
                    candidate_pt = parts[-offset]
                    pt_match = match_plate_type(candidate_pt)
                    if pt_match and pt_match not in ["+", "-"]:
                        idx = len(parts) - offset
                        candidate_d3 = parts[idx + 1] if idx + 1 < len(parts) else ""
                        candidate_d4 = parts[idx + 2] if idx + 2 < len(parts) else ""
                        candidate_prov = parts[idx + 3] if idx + 3 < len(parts) else ""
                        d3_valid = (
                            0 < len(candidate_d3) <= MultiTaskPlateDataset._MAX_D3_LEN
                            and all(c in MultiTaskPlateDataset._VALID_D3_CHARS for c in candidate_d3)
                        )
                        d4_valid = (
                            0 < len(candidate_d4) <= MultiTaskPlateDataset._MAX_D4_LEN
                            and all(c in MultiTaskPlateDataset._VALID_D4_CHARS for c in candidate_d4)
                        )
                        if d3_valid or d4_valid:
                            plate_type = pt_match
                            d3 = candidate_d3 if d3_valid else ""
                            d4 = candidate_d4 if d4_valid else ""
                            prov = candidate_prov
                            parsed = True
                            break

            # Legacy fallback: last 3 parts as d3_d4_prov
            if not parsed and len(parts) >= 3:
                candidate_d3 = parts[-3]
                candidate_d4 = parts[-2]
                candidate_prov = parts[-1]
                d3_valid = (
                    0 < len(candidate_d3) <= MultiTaskPlateDataset._MAX_D3_LEN
                    and all(c in MultiTaskPlateDataset._VALID_D3_CHARS for c in candidate_d3)
                )
                d4_valid = (
                    0 < len(candidate_d4) <= MultiTaskPlateDataset._MAX_D4_LEN
                    and all(c in MultiTaskPlateDataset._VALID_D4_CHARS for c in candidate_d4)
                )
                if d3_valid or d4_valid:
                    d3 = candidate_d3 if d3_valid else ""
                    d4 = candidate_d4 if d4_valid else ""
                    prov = candidate_prov

        if d3 in ["+", "-"]: d3 = ""
        if d4 in ["+", "-"]: d4 = ""
        if prov in ["+", "-"]: prov = ""

        # Normalize plate type
        matched_pt = match_plate_type(plate_type)
        if not matched_pt or matched_pt in ["+", "-"]:
            matched_pt = "WhiteBlackCar"

        # DLT consonant remapping
        d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค")

        # Final length enforcement (hard clamp)
        d3 = d3[:MultiTaskPlateDataset._MAX_D3_LEN]
        d4 = d4[:MultiTaskPlateDataset._MAX_D4_LEN]

        prov = match_province_value(prov) if prov else ""
        return matched_pt, d3, d4, prov

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

        # 2. Unified CTC Sequence Target (Clean d3 ≤ 3 chars + clean d4 ≤ 4 digits)
        d3_clean = normalize_thai_script(d3)[:self._MAX_D3_LEN]
        d4_clean = d4[:self._MAX_D4_LEN]
        d3_indices = [CTC_CHAR_TO_IDX[c] for c in d3_clean if c in CTC_CHAR_TO_IDX and c != "<pad>"]
        d4_indices = [CTC_CHAR_TO_IDX[c] for c in d4_clean if c in CTC_CHAR_TO_IDX and c != "<pad>"]
        full_indices = d3_indices + d4_indices
        seq_len = len(full_indices)

        seq_targets = torch.zeros(self.max_seq_len, dtype=torch.long)
        if seq_len > 0:
            valid_len = min(seq_len, self.max_seq_len)
            seq_targets[:valid_len] = torch.tensor(full_indices[:valid_len], dtype=torch.long)
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
        }
        return pad, targets


# =============================================================================
# Training Pipeline
# =============================================================================

def train_selen_lpr(
    data_dirs: List[Path],
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 0.0005,
    device: str = "cuda:0",
    loss_mode: str = "ctc",
    resume: str = "",
    output_dir: str = "runs/train/selen_lpr_0.0.1",
    model_export_path: str = "models/reader/selen-lpr-0.0.1/best_reader.pth",
    device_str: Optional[str] = None,
    dry_run: bool = False
):
    if device_str is not None:
        device = device_str

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = Path(model_export_path).resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[{get_utc_timestamp()}] SELEN-LPR-0.0.1 UNIFIED LEFT-TO-RIGHT CTC TRAINING PIPELINE")
    print("=" * 80)
    device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Epochs        : {epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Learning Rate : {lr}")
    print(f"Loss Mode     : {loss_mode}")
    print(f"Dry Run Mode  : {dry_run}")
    if resume:
        print(f"Resume Ckpt   : {resume}")
    print(f"Output Dir    : {out_dir}")
    print(f"Export Checkpt: {export_path}")
    print("=" * 80)

    # 1. Dataset & DataLoader (95% Train / 5% Validation)
    full_dataset = MultiTaskPlateDataset(data_dirs=data_dirs, img_h=64, img_w=384, max_seq_len=7)
    total_len = len(full_dataset)
    val_len = int(total_len * 0.05)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    if dry_run:
        print(f"[{get_utc_timestamp()}] [DRY-RUN] Slicing minimal dataset for rapid 1-epoch execution.")
        dry_train_len = min(128, len(train_ds))
        dry_val_len = min(64, len(val_ds))
        train_ds = Subset(train_ds, range(dry_train_len))
        val_ds = Subset(val_ds, range(dry_val_len))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"[{get_utc_timestamp()}] [DRY-RUN] Active Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        print(f"[{get_utc_timestamp()}] [DATA SPLIT] Train: {train_len:,} samples | Val: {val_len:,} samples")

    # 2. Model Initialization
    model = SelenLPR(d_model=256, nhead=4, vocab_size=VOCAB_SIZE).to(device)
    params_count = sum(p.numel() for p in model.parameters())
    print(f"[{get_utc_timestamp()}] [MODEL] SelenLPR Initialized. Parameters: {params_count:,} ({params_count*4/1024/1024:.2f} MB float32)")

    # 3. Hard Freeze Converged Heads & Load Checkpoint
    base_ckpt_path = Path(resume if resume else "models/reader/selen-lpr-0.0.1/best_reader.pth").resolve()
    if base_ckpt_path.is_file():
        print(f"[{get_utc_timestamp()}] [*] Loading converged checkpoint: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Strip mismatched legacy shape keys or old Conv1d neck keys if present
        keys_to_clean = ["d4_head.weight", "d4_head.bias"]
        for k in list(state_dict.keys()):
            if k in keys_to_clean:
                del state_dict[k]
            elif "ctc_head.neck" in k and "weight_ih" not in k:
                # Discard legacy Conv1d/BatchNorm weights from previous head iteration
                del state_dict[k]
            elif k in model.state_dict() and state_dict[k].shape != model.state_dict()[k].shape:
                del state_dict[k]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[{get_utc_timestamp()}] [+] Loaded model weights with strict=False (missing={len(missing)}, unexpected={len(unexpected)}).")
    else:
        print(f"[{get_utc_timestamp()}] [!] Warning: Checkpoint not found at: {base_ckpt_path}")

    # Freeze backbone, type_head, and prov_head (param.requires_grad = False)
    # Keep dedicated 2-layer BiLSTM sequence neck and ctc_head classifier unfrozen (requires_grad = True)
    for name, param in model.named_parameters():
        if "ctc_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    locked_names = [n for n, p in model.named_parameters() if not p.requires_grad]
    train_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[{get_utc_timestamp()}] [FREEZE] Locked {len(locked_names)} parameter tensors (Type & Prov accuracies locked).")
    print(f"[{get_utc_timestamp()}] [TRAINABLE] Active {len(train_names)} parameter tensors (CTC 2-layer BiLSTM sequence head).")

    # 4. Loss & Optimizer
    criterion = SelenMultiTaskLoss(blank_idx=CTC_BLANK_IDX).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=1e-4
    )
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_exact_acc = 0.0
    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        # Keep frozen modules strictly in eval mode to preserve ImageNet/checkpoint
        # BatchNorm running statistics and disable Dropout on locked heads
        model.backbone.eval()
        model.proj.eval()
        model.encoder.eval()
        model.type_head.eval()
        model.prov_head.eval()

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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
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

                # Accuracy evaluation
                pred_type = v_preds["type_logits"].argmax(dim=-1)
                pred_prov = v_preds["prov_logits"].argmax(dim=-1)
                pred_pairs = decode_and_split_ctc(v_preds["ctc_logits"], CTC_IDX_TO_CHAR, blank_idx=CTC_BLANK_IDX)

                for b in range(val_img.shape[0]):
                    is_t = (pred_type[b] == v_batch["type"][b]).item()
                    is_p = (pred_prov[b] == v_batch["prov"][b]).item()

                    # Ground truth deterministic split
                    tgt_len = v_batch["target_lengths"][b].item()
                    gt_tokens = v_batch["seq_targets"][b][:tgt_len].cpu().tolist()
                    gt_full = "".join(CTC_IDX_TO_CHAR.get(k, "") for k in gt_tokens)
                    last_thai_idx = -1
                    for i, ch in enumerate(gt_full):
                        if "\u0e01" <= ch <= "\u0e2e":
                            last_thai_idx = i
                    if last_thai_idx != -1:
                        gt_d3 = gt_full[:last_thai_idx + 1]
                        gt_d4 = gt_full[last_thai_idx + 1:]
                    else:
                        gt_d3 = ""
                        gt_d4 = gt_full

                    pred_d3, pred_d4 = pred_pairs[b]

                    is_3 = (pred_d3 == gt_d3)
                    is_4 = (pred_d4 == gt_d4)

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
              f"Type: {acc_type:.1f}% | d3: {acc_d3:.1f}% | d4: {acc_d4:.1f}% | Prov: {acc_prov:.1f}% | "
              f"Exact Plate: {acc_exact:.2f}% ({correct_exact}/{total_eval})")

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "acc_type": round(acc_type, 2),
            "acc_d3": round(acc_d3, 2),
            "acc_d4": round(acc_d4, 2),
            "acc_prov": round(acc_prov, 2),
            "acc_exact": round(acc_exact, 2)
        }
        history.append(epoch_record)

        # Checkpoints
        if not dry_run:
            torch.save(model.state_dict(), out_dir / "last_reader.pth")
            if avg_val_loss < best_val_loss or acc_exact > best_exact_acc:
                best_val_loss = min(avg_val_loss, best_val_loss)
                best_exact_acc = max(acc_exact, best_exact_acc)
                torch.save(model.state_dict(), out_dir / "best_reader.pth")
                torch.save(model.state_dict(), export_path)
                print(f"[{get_utc_timestamp()}] [SAVE] New best checkpoint saved to: {export_path}")
        else:
            torch.save(model.state_dict(), out_dir / "dry_run_reader.pth")
            print(f"[{get_utc_timestamp()}] [DRY-RUN] Verified checkpoint saved to: {out_dir / 'dry_run_reader.pth'}")

    total_time = time.time() - t_start
    best_hash = calculate_sha256(export_path) if export_path.is_file() else "N/A"

    # Save summary metadata
    summary = {
        "model_name": "selen-lpr-0.0.1",
        "architecture": "SelenLPR (MobileNetV3 + Transformer Context + 2-Layer BiLSTM LeftToRightCTCHead)",
        "loss_function": "SelenMultiTaskLoss (nn.CTCLoss)",
        "loss_mode": loss_mode,
        "parameters": params_count,
        "trainable_parameters": sum(p.numel() for p in trainable_params),
        "model_size_mb": round(params_count * 4 / (1024 * 1024), 2),
        "epochs": epochs,
        "batch_size": batch_size,
        "total_time_sec": round(total_time, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_exact_acc": round(best_exact_acc, 2),
        "checkpoint_path": str(export_path),
        "checkpoint_sha256": best_hash,
        "history": history
    }
    with open(out_dir / "selen_lpr_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[{get_utc_timestamp()}] SELEN-LPR-0.0.1 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total Time      : {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"Best Val Loss   : {best_val_loss:.4f}")
    print(f"Best Exact Acc  : {best_exact_acc:.2f}%")
    print(f"Model Checkpoint: {export_path}")
    print(f"SHA-256 Hash    : {best_hash}")
    print("=" * 80 + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train selen-lpr-0.0.1 with Unified Left-to-Right CTC Head")
    parser.add_argument("--data-dirs", type=str, default="", help="Comma-separated dataset directories (defaults to production release)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--loss-mode", type=str, default="ctc")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="runs/train/selen_lpr_0.0.1", help="Output directory for runs and checkpoints")
    parser.add_argument("--export-path", type=str, default="models/reader/selen-lpr-0.0.1/best_reader.pth", help="Final exported checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run for verification")
    args = parser.parse_args()

    if args.data_dirs:
        data_directories = [Path(d.strip()) for d in args.data_dirs.split(",") if d.strip()]
    else:
        # Default to latest consolidated ready dataset release (202611030335)
        default_ready = ROOT_DIR / "dataset" / "ready" / "202611030335"
        if default_ready.is_dir():
            data_directories = [default_ready]
        else:
            data_directories = [
                ROOT_DIR / "dataset" / "ready" / "202610261140",
                ROOT_DIR / "dataset" / "snapshot" / "mobile",
                ROOT_DIR / "dataset" / "verified" / "dataset-staging-approved",
                ROOT_DIR / "dataset" / "dataset-cctv-synthetic" / "plates",
                ROOT_DIR / "dataset" / "dataset-mobile-synthetic" / "plates",
            ]

    train_selen_lpr(
        data_dirs=data_directories,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        loss_mode=args.loss_mode,
        resume=args.resume,
        output_dir=args.output_dir,
        model_export_path=args.export_path,
        dry_run=args.dry_run
    )
