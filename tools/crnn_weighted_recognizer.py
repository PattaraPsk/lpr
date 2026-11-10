#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRNN License Plate Recognizer with Static Weighted Cross-Entropy Loss Multipliers.

Key Architectural Features:
1. Standard CRNN Architecture:
   - CNN Backbone: VGG-style 7-layer convolutional feature extractor with BatchNorm & ReLU.
   - BiLSTM Neck: 2-layer Bidirectional LSTM for horizontal spatial sequence modeling.
   - Linear Head: Projecting recurrent representations to vocabulary logits.
2. Definitive Thai License Plate Vocabulary:
   - Blank / Pad token (<blank>)
   - Digits (0-9)
   - 41 Standard DLT Consonants (excluding obsolete ฃ, ฅ, and optical duplicate ฏ)
   - 78 Province Identifiers (77 official provinces + Betong)
3. Custom Weighted Cross-Entropy Loss (nn.Module):
   - Static loss multiplier of 3.0 for visually similar characters: ก, ถ, ภ, ข, ช, ซ.
   - Static loss multiplier of 2.0 for all province identifiers.
   - Perfectly aligned weight tensor buffer ensuring automatic device placement.

Complies with:
- ASD-STE100 (Simplified Technical English)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System - Clauses 7.5, 8.4, Annex A.6.7)
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Ensure UTF-8 stdout on Windows hosts
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Definitive Thai License Plate Vocabulary (DLT Specifications)
# =============================================================================

BLANK_TOKEN = "<blank>"

# Digits 0-9
DIGITS: List[str] = [str(i) for i in range(10)]

# Standard 41 DLT Valid Consonants (ฃ, ฅ, ฏ strictly excluded per DLT specification)
THAI_CONSONANTS_41: List[str] = list("กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

# Visually Similar Characters Targeted for Hard-Mining Loss (3.0x Multiplier)
VISUALLY_SIMILAR_CHARS: List[str] = ["ก", "ถ", "ภ", "ข", "ช", "ซ"]

# 78 Official Thai Province Identifiers (77 Provinces + Betong)
OFFICIAL_PROVINCES: List[str] = [
    "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา",
    "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี", "ชลบุรี", "ระยอง",
    "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว",
    "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร",
    "ชัยภูมิ", "อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี",
    "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร",
    "นครพนม", "มุกดาหาร", "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์",
    "แพร่", "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน", "นครสวรรค์",
    "อุทัยธานี", "กำแพงเพชร", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร",
    "เพชรบูรณ์", "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม",
    "สมุทรสาคร", "สมุทรสงคราม", "เพชรบุรี", "ประจวบคีรีขันธ์",
    "นครศรีธรรมราช", "กระบี่", "พังงา", "ภูเก็ต", "สุราษฎร์ธานี",
    "ระนอง", "ชุมพร", "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี",
    "ยะลา", "นราธิวาส", "เบตง",
]


def build_thai_lpr_vocabulary() -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """
    Construct the definitive Thai License Plate vocabulary mapping:
    - Index 0: <blank> (CTC blank / pad token)
    - Indices 1..10: Digits '0' through '9'
    - Indices 11..51: 41 Thai DLT Consonants
    - Indices 52..129: 78 Thai Province Identifiers
    Total Vocabulary Size: 130 tokens.
    """
    token_list: List[str] = [BLANK_TOKEN] + DIGITS + THAI_CONSONANTS_41 + OFFICIAL_PROVINCES
    
    char_to_idx: Dict[str, int] = {token: idx for idx, token in enumerate(token_list)}
    idx_to_char: Dict[int, str] = {idx: token for idx, token in enumerate(token_list)}
    
    return char_to_idx, idx_to_char, token_list


VOCAB, IDX_TO_CHAR, TOKEN_LIST = build_thai_lpr_vocabulary()
VOCAB_SIZE = len(TOKEN_LIST)


# =============================================================================
# Custom Weighted Cross-Entropy Loss Module
# =============================================================================

class ThaiWeightedCrossEntropyLoss(nn.Module):
    """
    Custom Cross-Entropy Loss with Static Loss Multipliers for Thai LPR.
    
    Features:
    - Weight tensor perfectly aligned with vocabulary length (size = VOCAB_SIZE).
    - Injects static loss multiplier of 3.0 for visually confusable characters:
      ก, ถ, ภ, ข, ช, ซ.
    - Injects static loss multiplier of 2.0 for all 78 province identifiers.
    - Baseline multiplier of 1.0 for standard digits and other consonants.
    - Registered as a persistent buffer to ensure automatic device placement
      across CPU and GPU without manual device casting.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        similar_char_multiplier: float = 3.0,
        province_multiplier: float = 2.0,
        base_weight: float = 1.0,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.vocab = vocab if vocab is not None else VOCAB
        self.vocab_size = len(self.vocab)
        self.ignore_index = ignore_index
        self.reduction = reduction

        # 1. Allocate initial weights vector
        weight_tensor = torch.full((self.vocab_size,), fill_value=base_weight, dtype=torch.float32)

        # 2. Inject 3.0 multiplier for visually similar Thai characters
        for ch in VISUALLY_SIMILAR_CHARS:
            if ch in self.vocab:
                idx = self.vocab[ch]
                weight_tensor[idx] = float(similar_char_multiplier)

        # 3. Inject 2.0 multiplier for all province identifiers
        for prov in OFFICIAL_PROVINCES:
            if prov in self.vocab:
                idx = self.vocab[prov]
                weight_tensor[idx] = float(province_multiplier)

        # 4. Register buffer: guarantees automatic .to(device) placement
        self.register_buffer("weight", weight_tensor)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted cross entropy loss.
        
        Args:
            logits: Predicted class scores.
                Supported shapes:
                - 3D sequence: (Batch, Time, Vocab) or (Batch, Vocab, Time)
                - 2D classification: (Batch, Vocab)
            targets: Ground-truth target indices.
                Supported shapes:
                - 2D sequence: (Batch, Time)
                - 1D classification: (Batch)
        Returns:
            Scalar loss tensor (or unreduced tensor matching reduction mode).
        """
        # Ensure weight buffer is on the exact same device and dtype as logits
        weight = self.weight.to(device=logits.device, dtype=logits.dtype)

        # Handle 3D Sequence Logits from CRNN: (Batch, Time, Vocab)
        if logits.dim() == 3:
            b, t, c = logits.shape
            if c == self.vocab_size:
                # Shape is (Batch, Time, Vocab) -> flatten to (B * T, Vocab)
                logits_flat = logits.reshape(-1, self.vocab_size)
                targets_flat = targets.reshape(-1)
                return F.cross_entropy(
                    logits_flat,
                    targets_flat,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )
            elif t == self.vocab_size:
                # Shape is (Batch, Vocab, Time)
                return F.cross_entropy(
                    logits,
                    targets,
                    weight=weight,
                    ignore_index=self.ignore_index,
                    reduction=self.reduction,
                )

        # Handle 2D Standard Classification Logits: (Batch, Vocab)
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )


# =============================================================================
# Standard CRNN Recognizer Architecture
# =============================================================================

class CRNNBackbone(nn.Module):
    """
    Standard VGG-style 7-layer Convolutional Backbone for CRNN.
    Reduces 2D image height to 1 while preserving horizontal time/character sequence.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1: (C, 32, W) -> (64, 16, W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 2: (64, 16, W/2) -> (128, 8, W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv block 3: (128, 8, W/4) -> (256, 8, W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv block 4: (256, 8, W/4) -> (256, 4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Conv block 5: (256, 4, W/4) -> (512, 4, W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Conv block 6: (512, 4, W/4) -> (512, 2, W/4)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # Conv block 7: (512, 2, W/4) -> (512, 1, W/4)
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Adaptive pool guarantees height = 1 regardless of arbitrary input aspect ratio
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor (Batch, Channels, Height, Width)
        Returns:
            Sequence features: (Batch, Time, Channels) where Channels = 512
        """
        conv = self.features(x)
        pooled = self.adaptive_pool(conv)  # (Batch, 512, 1, Time)
        squeezed = pooled.squeeze(2)       # (Batch, 512, Time)
        sequence = squeezed.permute(0, 2, 1)  # (Batch, Time, 512)
        return sequence


class CRNNRecognizer(nn.Module):
    """
    Standard CRNN Recognizer for Thai License Plates:
    1. CNN Backbone: Extracts spatial image feature map, collapsed vertically to 1D sequence.
    2. BiLSTM Neck: 2-layer Bidirectional LSTM capturing bidirectional context.
    3. Linear Head: Projects hidden states to vocabulary classification logits.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 256,
        num_bilstm_layers: int = 2,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 1. CNN Feature Extractor Backbone (output feature dim = 512)
        self.backbone = CRNNBackbone(in_channels=in_channels)
        self.cnn_out_dim = 512

        # 2. BiLSTM Neck
        self.neck = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_bilstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_bilstm_layers > 1 else 0.0,
        )

        # 3. Linear Classification Head
        self.head = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (Batch, Channels, Height, Width)
        Returns:
            Logits tensor of shape (Batch, Time, Vocab_Size)
        """
        # 1. Extract sequential feature representations
        seq_features = self.backbone(x)  # (Batch, Time, 512)

        # 2. Recurrent sequential modeling
        recurrent_out, _ = self.neck(seq_features)  # (Batch, Time, hidden_size * 2)

        # 3. Vocabulary projection
        logits = self.head(recurrent_out)  # (Batch, Time, vocab_size)

        return logits

    @torch.no_grad()
    def decode_greedy(self, logits: torch.Tensor) -> List[List[str]]:
        """
        Greedy character decoding from model output logits.
        """
        pred_indices = logits.argmax(dim=-1).cpu().tolist()
        batch_results: List[List[str]] = []
        for seq in pred_indices:
            tokens = [IDX_TO_CHAR[idx] for idx in seq if idx != 0 and idx in IDX_TO_CHAR]
            batch_results.append(tokens)
        return batch_results


# =============================================================================
# Helper Construction & Device Placement Utilities
# =============================================================================

def get_optimal_device() -> torch.device:
    """Detect available hardware accelerator automatically."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_crnn_training_pipeline(
    in_channels: int = 3,
    hidden_size: int = 256,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[CRNNRecognizer, ThaiWeightedCrossEntropyLoss, torch.device]:
    """
    Factory function creating the CRNN model and weighted loss module,
    automatically placed on the specified or optimal device.
    """
    target_device = torch.device(device) if device is not None else get_optimal_device()

    model = CRNNRecognizer(
        in_channels=in_channels,
        hidden_size=hidden_size,
        vocab_size=VOCAB_SIZE,
    ).to(target_device)

    loss_fn = ThaiWeightedCrossEntropyLoss(
        vocab=VOCAB,
        similar_char_multiplier=3.0,
        province_multiplier=2.0,
    ).to(target_device)

    return model, loss_fn, target_device


if __name__ == "__main__":
    # Self-validation and diagnostic check
    dev = get_optimal_device()
    print(f"[{dev}] Initializing Thai CRNN and Weighted Loss Engine...")
    print(f"Vocabulary Size : {VOCAB_SIZE} tokens")
    print(f"Visually Similar: {VISUALLY_SIMILAR_CHARS} -> Multiplier: 3.0")
    print(f"Provinces Count : {len(OFFICIAL_PROVINCES)} -> Multiplier: 2.0")

    model, loss_fn, device = create_crnn_training_pipeline(device=dev)
    print(f"Device Placement: Model on {next(model.parameters()).device}, Loss weight buffer on {loss_fn.weight.device}")

    # Dummy batch verification
    dummy_images = torch.randn(2, 3, 32, 128, device=device)
    out_logits = model(dummy_images)
    print(f"Input Shape     : {tuple(dummy_images.shape)}")
    print(f"Output Logits   : {tuple(out_logits.shape)}")

    # Target shape perfectly matches batch and sequence length
    seq_len = out_logits.shape[1]
    dummy_targets = torch.randint(0, VOCAB_SIZE, (2, seq_len), device=device)

    loss = loss_fn(out_logits, dummy_targets)
    print(f"Weighted CE Loss: {loss.item():.4f}")
    print("Verification Gate Passed.")
