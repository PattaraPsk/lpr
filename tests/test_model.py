"""
Unit tests for model architectures, forward pass, and loss computation.
Complies with ISO/IEC/IEEE 29119 & ISO/IEC 42001:2023 Verification Standards.
"""

import unittest
import torch
import torch.nn as nn

from src.models.losses import CTCLossWrapper, FocalLoss
from src.models.model import CRNNEdgeRecognizer, build_model
from src.inference.predict import PlatePredictor


class TestModelArchitecture(unittest.TestCase):
    """Test suite for CRNN model dimensions, gradients, and loss functions."""

    def setUp(self):
        self.device = "cpu"
        self.batch_size = 2
        self.height = 64
        self.width = 192
        self.num_classes = 52
        self.model = CRNNEdgeRecognizer(in_channels=3, num_classes=self.num_classes, hidden_size=64)
        self.model.eval()

    def test_model_forward_pass_shape(self):
        """Verifies forward pass outputs [TimeSteps, BatchSize, NumClasses]."""
        x = torch.randn(self.batch_size, 3, self.height, self.width)
        with torch.no_grad():
            log_probs = self.model(x)

        # Downsampling width by 4 -> 192 / 4 = 48 time steps
        self.assertEqual(log_probs.dim(), 3)
        self.assertEqual(log_probs.size(1), self.batch_size)
        self.assertEqual(log_probs.size(2), self.num_classes)
        self.assertEqual(log_probs.size(0), self.width // 4)

    def test_ctc_loss_calculation(self):
        """Verifies CTCLossWrapper computes valid scalar loss without NaN or Inf."""
        criterion = CTCLossWrapper(blank_index=0)
        x = torch.randn(self.batch_size, 3, self.height, self.width)
        self.model.train()
        log_probs = self.model(x)

        targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
        input_lengths = torch.tensor([48, 48], dtype=torch.long)
        target_lengths = torch.tensor([4, 4], dtype=torch.long)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        self.assertFalse(torch.isinf(loss), "Loss must not be Inf")
        self.assertGreater(loss.item(), 0.0)

        # Test backward pass gradients
        loss.backward()
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())

    def test_focal_loss(self):
        """Verifies FocalLoss calculation and reduction."""
        focal = FocalLoss(gamma=2.0)
        inputs = torch.randn(4, 10)
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        loss = focal(inputs, targets)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_greedy_decoding(self):
        """Verifies greedy CTC sequence decoding."""
        log_probs = torch.zeros(5, 1, self.num_classes)
        # Sequence: [1, 1, 0 (blank), 2, 0] -> should decode to index 1 then 2
        log_probs[0, 0, 1] = 10.0
        log_probs[1, 0, 1] = 10.0
        log_probs[2, 0, 0] = 10.0
        log_probs[3, 0, 2] = 10.0
        log_probs[4, 0, 0] = 10.0

        custom_dict = {0: "<blank>", 1: "ก", 2: "ข"}
        decoded = self.model.decode_greedy(log_probs, idx_to_char=custom_dict)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0], "กข")


if __name__ == "__main__":
    unittest.main()
