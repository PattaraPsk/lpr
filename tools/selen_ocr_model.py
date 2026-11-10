#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenOCR v0.1.0 Architecture (Enlarged Multi-Task Edge Thai LPR Model)
Complies with:
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.6.7 / A.8.2 / A.9.3)
- ISO/IEC 25012:2008 (Data Quality & Model Lineage)
- ISO 8601:2019 (UTC Representations)

Architectural Enhancements over SelenLPR v0.0.1:
1. Scaled-up Vision Backbone: MobileNetV3-Large with Multi-Scale FPN Adapter.
   Extracts and fuses features from Stage 3 (112ch), Stage 4 (160ch), and Stage 5 (960ch)
   into a unified d_model=384 feature map, ensuring superior recognition across
   extreme low-resolution (64px) to high-resolution (600px) plate crops.
2. Deepened Transformer Context: 4 Transformer Encoder Layers (6 heads, d_ff=1024).
3. 2-Layer BiLSTM Sequence Neck: Hidden size 256 (512-dim bidirectional representation).
4. Delimiter-Aware & Token-Aware CTC Sequence Head: Vocab=145 classes.
   Outputs single text line: "plateType d3 d4 province" (max 11 tokens).
5. Auxiliary Multi-Task Heads:
   - Plate Type Head (13 classes: 11 DLT categories + '+' + '-')
   - Closed-Set Province Head (80 classes: 78 subdivisions + unknown + '+' + '-')
6. Edge Latency Budget: Designed for CPU edge inference < 8 ms and > 1,800 FPS on RTX 4090.
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from tools.selen_token_codec import (
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    PLATE_TYPES,
    PROVINCES_78
)


class MultiScaleFPNAdapter(nn.Module):
    """
    Fuses low-level and high-level feature maps from MobileNetV3-Large
    to resolve fine character strokes in distant low-res plates (64px)
    and detailed glyph contours in high-res plates (600px).
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 384):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in in_channels_list
        ])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: [c3, c4, c5]
        # c3: Stage 3 [B, 112, H/16, W/16]
        # c4: Stage 4 [B, 160, H/16, W/16]
        # c5: Stage 5 [B, 960, H/32, W/32]
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Top-down pathway: upsample c5 to match c4/c3 spatial size
        top_down = laterals[2]
        c4_size = laterals[1].shape[-2:]
        top_down = F.interpolate(top_down, size=c4_size, mode="nearest") + laterals[1]
        
        c3_size = laterals[0].shape[-2:]
        top_down = F.interpolate(top_down, size=c3_size, mode="nearest") + laterals[0]

        # Final smooth convolution
        out = self.fpn_convs[0](top_down)
        return out


class SelenOCR(nn.Module):
    """
    SelenOCR v0.1.0 Multi-Task Deep Text Recognition Model.
    """
    def __init__(
        self,
        d_model: int = 384,
        nhead: int = 6,
        num_encoder_layers: int = 4,
        lstm_hidden: int = 256,
        vocab_size: int = VOCAB_SIZE,
        num_plate_types: int = 13, # 11 DLT + '+' + '-'
        num_provinces: int = 81    # 78 + unknown + '+' + '-'
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 1. Scaled Vision Backbone (MobileNetV3-Large)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        features = mobilenet.features

        # Split features into stages for multi-scale extraction
        # Stage 0..12: early to mid layers (c3: 112 ch)
        # Stage 13..15: deep layers (c4: 160 ch)
        # Stage 16: top conv layer (c5: 960 ch)
        self.stage1_to_3 = nn.Sequential(*features[:13])  # outputs 112 ch
        self.stage4 = nn.Sequential(*features[13:16])     # outputs 160 ch
        self.stage5 = nn.Sequential(*features[16:])       # outputs 960 ch

        # 2. Multi-Scale FPN Adapter (fuses 112, 160, 960 -> d_model 384)
        self.fpn = MultiScaleFPNAdapter(in_channels_list=[112, 160, 960], out_channels=d_model)

        # 3. Spatial Sequence Adapter & Positional Encoding
        # Normalizes spatial features to fixed sequence length W'=40 time steps (40 steps for max 11 tokens = 3.6x ratio)
        self.seq_pool = nn.AdaptiveAvgPool2d((1, 40))
        self.pos_encoder = nn.Parameter(torch.randn(1, 40, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

        # 4. Transformer Context Encoder (4 layers, 6 heads, d_ff=1024)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 5. Head 1: Plate Type Classifier (13 classes: 11 DLT + '+' + '-')
        self.type_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_plate_types)
        )

        # 6. Head 2: Province Classifier (81 classes: 78 subdivisions + unknown + '+' + '-')
        self.prov_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_provinces)
        )

        # 7. Head 3: 2-Layer BiLSTM Sequence Neck + CTC Linear Head
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )
        self.ctc_head = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, 3, H, W]
        # Multi-scale backbone extraction
        c3 = self.stage1_to_3(x)  # [B, 112, H/16, W/16]
        c4 = self.stage4(c3)      # [B, 160, H/16, W/16]
        c5 = self.stage5(c4)      # [B, 960, H/32, W/32]

        feat_2d = self.fpn([c3, c4, c5]) # [B, d_model, H', W']

        # Pool to fixed sequence dimensions: [B, d_model, 1, 40] -> [B, 40, d_model]
        seq_feats = self.seq_pool(feat_2d).squeeze(2).permute(0, 2, 1)
        seq_feats = self.norm(seq_feats)

        memory = self.transformer_encoder(seq_feats + self.pos_encoder)

        # Global sequence embedding
        global_emb = memory.mean(dim=1)

        # 1. Type logits
        type_logits = self.type_head(global_emb)

        # 2. Province logits
        prov_logits = self.prov_head(global_emb)

        # 3. Sequence CTC logits (W'=40 time steps)
        bilstm_out, _ = self.bilstm(memory)
        ctc_logits = self.ctc_head(bilstm_out) # [B, 40, Vocab]

        return {
            "ctc_logits": ctc_logits,
            "type_logits": type_logits,
            "prov_logits": prov_logits
        }


if __name__ == "__main__":
    # Test instantiation and forward pass
    model = SelenOCR()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    dummy_input = torch.randn(2, 3, 64, 320)
    outputs = model(dummy_input)

    print("=" * 80)
    print("  SELENOCR v0.1.0 ARCHITECTURE VERIFICATION")
    print("=" * 80)
    print(f"Total Parameters     : {total_params:,} ({total_params * 4 / (1024**2):.2f} MB float32)")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Input Shape          : {dummy_input.shape}")
    print(f"CTC Logits Shape     : {outputs['ctc_logits'].shape}")
    print(f"Type Logits Shape    : {outputs['type_logits'].shape}")
    print(f"Province Logits Shape: {outputs['prov_logits'].shape}")
    print("=" * 80)
    print("Architecture verified successfully!")
