"""
Model Architecture for Edge Computer Vision and Thai License Plate Recognition.
Lightweight CNN-BiGRU-CTC Network.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import VOCAB_SIZE, IDX_TO_CHAR


class ConvBlock(nn.Module):
    """Convolution + BatchNorm + LeakyReLU + optional MaxPool."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool: Optional[tuple[int, int]] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(pool) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class CRNNEdgeRecognizer(nn.Module):
    """
    Lightweight CRNN (CNN + Bidirectional GRU + CTC Head) for edge license plate OCR.
    Input: [B, C, H, W]
    Output: [T, B, NumClasses] log-probabilities for CTC Loss.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = VOCAB_SIZE,
        hidden_size: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # CNN Feature Extractor
        # Height 64 -> 32 -> 16 -> 8 -> 1
        # Width 192 -> 96 -> 48 -> 48 -> 48
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 64, pool=(2, 2)),          # -> [B, 64, H/2, W/2]
            ConvBlock(64, 128, pool=(2, 2)),                  # -> [B, 128, H/4, W/4]
            ConvBlock(128, 256),                              # -> [B, 256, H/4, W/4]
            ConvBlock(256, 256, pool=(2, 1)),                 # -> [B, 256, H/8, W/4]
            ConvBlock(256, 512),                              # -> [B, 512, H/8, W/4]
            ConvBlock(512, 512, pool=(2, 1)),                 # -> [B, 512, H/16, W/4]
        )

        # Adaptive height pooling to collapse height dimension to 1
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # Sequence modeling with Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # Output projection head to vocabulary classes
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [B, C, H, W]
        Returns: [T, B, NumClasses] log probabilities
        """
        features = self.backbone(x)           # [B, 512, H', W']
        pooled = self.adaptive_pool(features) # [B, 512, 1, W']
        squeezed = pooled.squeeze(2)          # [B, 512, W']

        # Permute to [B, W', 512] for RNN sequence processing
        seq_input = squeezed.permute(0, 2, 1)

        rnn_out, _ = self.rnn(seq_input)      # [B, W', hidden_size * 2]
        logits = self.fc(rnn_out)             # [B, W', NumClasses]

        # Convert to [W', B, NumClasses] as expected by PyTorch CTCLoss
        logits_t = logits.permute(1, 0, 2)
        log_probs = F.log_softmax(logits_t, dim=-1)

        return log_probs

    @torch.no_grad()
    def decode_greedy(
        self,
        log_probs: torch.Tensor,
        idx_to_char: Optional[Dict[int, str]] = None,
        blank_idx: int = 0,
    ) -> List[str]:
        """
        Greedy CTC decoding.
        log_probs: [T, B, NumClasses]
        Returns list of decoded string predictions of length B.
        """
        if idx_to_char is None:
            idx_to_char = IDX_TO_CHAR

        # Argmax over vocabulary dimension -> [T, B]
        best_indices = log_probs.argmax(dim=-1)
        # Permute to [B, T]
        best_indices = best_indices.permute(1, 0).cpu().numpy()

        results: List[str] = []
        for batch_seq in best_indices:
            collapsed: List[int] = []
            prev = -1
            for idx in batch_seq:
                if idx != prev:
                    if idx != blank_idx:
                        collapsed.append(int(idx))
                    prev = idx

            decoded_text = "".join(idx_to_char.get(k, "") for k in collapsed)
            results.append(decoded_text)

        return results


def build_model(config: Optional[Dict] = None) -> CRNNEdgeRecognizer:
    """Factory to build model from config dictionary."""
    if config is None:
        return CRNNEdgeRecognizer()

    backbone_cfg = config.get("backbone", {})
    seq_cfg = config.get("sequence_model", {})
    head_cfg = config.get("head", {})

    in_c = backbone_cfg.get("in_channels", 3)
    hidden_size = seq_cfg.get("hidden_size", 128)
    num_layers = seq_cfg.get("num_layers", 2)
    dropout = seq_cfg.get("dropout", 0.1)
    num_classes = head_cfg.get("num_classes", VOCAB_SIZE)

    return CRNNEdgeRecognizer(
        in_channels=in_c,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_gru_layers=num_layers,
        dropout=dropout,
    )
