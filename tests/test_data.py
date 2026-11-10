"""
Unit tests for data preparation, dataset loading, and transforms.
Complies with ISO/IEC/IEEE 29119 & ISO 25012 Data Quality.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
import torch

from src.data.dataset import (
    BLANK_TOKEN,
    CHAR_TO_IDX,
    DLT_THAI_CONSONANTS,
    IDX_TO_CHAR,
    VOCAB_SIZE,
    LicensePlateDataset,
    ctc_collate_fn,
    decode_indices,
    encode_label,
    get_dataloader,
)
from src.data.prepare import create_splits, prepare_dataset, scan_dataset_directory
from src.utils.metrics import sanitize_thai_plate_text


class TestDataPipeline(unittest.TestCase):
    """Test suite for data loading, encoding, splitting, and preprocessing."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        # Create dummy image and label
        self.img_path = self.data_dir / "sample_1กข1234.jpg"
        dummy_img = np.full((64, 192, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(self.img_path), dummy_img)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vocab_size_and_dlt_characters(self):
        """Verifies Thai DLT 41 consonants + 10 digits + blank token = 52."""
        self.assertEqual(len(DLT_THAI_CONSONANTS), 41, "Must contain exactly 41 DLT consonants")
        self.assertNotIn("ฃ", DLT_THAI_CONSONANTS, "Obsolete ฃ must be excluded")
        self.assertNotIn("ฅ", DLT_THAI_CONSONANTS, "Obsolete ฅ must be excluded")
        self.assertNotIn("ฏ", DLT_THAI_CONSONANTS, "Confusing ฏ must be excluded")
        self.assertEqual(VOCAB_SIZE, 52, "Total vocabulary size must be 52")
        self.assertEqual(IDX_TO_CHAR[0], BLANK_TOKEN)

    def test_encode_and_decode_label(self):
        """Verifies label encoding and decoding consistency."""
        raw_text = "1กข9999"
        indices = encode_label(raw_text)
        self.assertEqual(len(indices), len(raw_text))

        decoded = decode_indices(indices)
        self.assertEqual(decoded, raw_text)

    def test_obsolete_consonant_remapping(self):
        """Verifies obsolete consonants are mapped to standard equivalents."""
        sanitized = sanitize_thai_plate_text("1ฃฅ99")
        self.assertEqual(sanitized, "1ขค99")

    def test_dataset_item_shape(self):
        """Verifies dataset item retrieval and tensor dimensions."""
        dataset = LicensePlateDataset(samples=[(str(self.img_path), "1กข1234")])
        self.assertEqual(len(dataset), 1)

        img_tensor, target_tensor, target_len, label = dataset[0]
        self.assertEqual(img_tensor.shape, (3, 64, 192))
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(target_len, len("1กข1234"))
        self.assertEqual(label, "1กข1234")

    def test_ctc_collate_fn(self):
        """Verifies batch collate output shapes and length tensors."""
        dataset = LicensePlateDataset(
            samples=[
                (str(self.img_path), "1กข1234"),
                (str(self.img_path), "9ฮ99"),
            ]
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        images, targets, input_lengths, target_lengths, labels = batch

        self.assertEqual(images.shape, (2, 3, 64, 192))
        self.assertEqual(input_lengths.shape, (2,))
        self.assertEqual(target_lengths.shape, (2,))
        self.assertEqual(target_lengths.tolist(), [7, 4])
        self.assertEqual(targets.shape[0], 11)

    def test_split_generation(self):
        """Verifies deterministic dataset splitting ratios."""
        samples = [(f"img_{i}.jpg", f"plate_{i}") for i in range(100)]
        train_s, val_s, test_s = create_splits(samples, 0.7, 0.15, 0.15, seed=42)
        self.assertEqual(len(train_s), 70)
        self.assertEqual(len(val_s), 15)
        self.assertEqual(len(test_s), 15)


if __name__ == "__main__":
    unittest.main()
