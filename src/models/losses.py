"""
Loss Functions for License Plate Recognition and Computer Vision Tasks.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCLossWrapper(nn.Module):
    """
    Standard CTC Loss wrapper with automatic log_softmax and zero_infinity handling.
    Expects logit tensor of shape [T, B, C] or [B, T, C].
    """

    def __init__(self, blank_index: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank_index, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates CTC Loss.
        - log_probs: [T, B, C] log probabilities
        - targets: 1D concatenated targets or [B, S]
        - input_lengths: [B]
        - target_lengths: [B]
        """
        # Ensure log_probs are proper log probabilities
        if not torch.allclose(log_probs.exp().sum(dim=-1), torch.ones_like(log_probs.sum(dim=-1)), atol=1e-2):
            log_probs = F.log_softmax(log_probs, dim=-1)

        return self.ctc(log_probs, targets, input_lengths, target_lengths)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in vision classifiers.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build loss function from configuration dictionary."""
    loss_type = config.get("type", "CTCLoss")
    if loss_type == "CTCLoss":
        return CTCLossWrapper(
            blank_index=config.get("blank_index", 0),
            zero_infinity=config.get("zero_infinity", True),
        )
    elif loss_type == "FocalLoss":
        return FocalLoss(
            alpha=config.get("alpha", 1.0),
            gamma=config.get("gamma", 2.0),
            reduction=config.get("reduction", "mean"),
        )
    elif loss_type == "CrossEntropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
