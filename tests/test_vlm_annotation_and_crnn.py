#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Test Suite for:
Task 1: Offline Data Annotation (PaddleOCR-VL)
Task 2: CRNN Recognizer with Weighted Loss Multipliers

Complies with:
- ISO/IEC 42001:2023 (Verification Gate & Lifecycle Validation)
- DLT Thailand Plate Specifications
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import torch.nn as nn
from PIL import Image

from tools.offline_vlm_annotator import (
    compute_sha256,
    is_safe_subpath,
    get_target_directory_names,
    parse_vlm_thai_lpr_output,
    scan_dataset_directories,
    run_offline_annotation,
    PaddleOCRVLAnnotator,
)
from tools.crnn_weighted_recognizer import (
    VOCAB,
    IDX_TO_CHAR,
    VOCAB_SIZE,
    VISUALLY_SIMILAR_CHARS,
    OFFICIAL_PROVINCES,
    CRNNRecognizer,
    ThaiWeightedCrossEntropyLoss,
    create_crnn_training_pipeline,
    get_optimal_device,
)


class TestTask1OfflineDataAnnotation(unittest.TestCase):
    """Test suite for Task 1: Offline Data Annotation with PaddleOCR-VL."""

    def setUp(self):
        """Create a temporary dataset environment with AA..ZZ directories and dummy images."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_lpr_dataset_"))
        
        # Create sample directories AA, AB, ZZ
        for folder_name in ["AA", "AB", "ZZ"]:
            folder = self.temp_dir / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            
            # Create a 64x64 dummy plate image
            img1 = Image.new("RGB", (64, 64), color=(255, 255, 255))
            img1.save(folder / "plate_01.jpg")
            
            # Create a duplicate image
            img2 = Image.new("RGB", (64, 64), color=(255, 255, 255))
            img2.save(folder / "plate_01_dup.jpg")
            
            # Create a distinct image
            img3 = Image.new("RGB", (64, 64), color=(100, 150, 200))
            img3.save(folder / "plate_02.png")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_target_directory_generation(self):
        """Verify generation of AA..ZZ (26 * 26 = 676 combinations)."""
        dirs = get_target_directory_names()
        self.assertEqual(len(dirs), 676)
        self.assertEqual(dirs[0], "AA")
        self.assertEqual(dirs[-1], "ZZ")
        self.assertIn("AB", dirs)
        self.assertIn("TH", dirs)

    def test_path_traversal_prevention(self):
        """Verify strict path traversal blocking."""
        base_dir = self.temp_dir
        safe_path = base_dir / "AA" / "plate_01.jpg"
        unsafe_path = base_dir / ".." / "system32" / "malicious.exe"

        self.assertTrue(is_safe_subpath(base_dir, safe_path))
        self.assertFalse(is_safe_subpath(base_dir, unsafe_path))

    def test_sha256_exact_deduplication(self):
        """Verify that identical images produce identical SHA-256 hashes."""
        img1 = self.temp_dir / "AA" / "plate_01.jpg"
        img2 = self.temp_dir / "AA" / "plate_01_dup.jpg"
        img3 = self.temp_dir / "AA" / "plate_02.png"

        hash1 = compute_sha256(img1)
        hash2 = compute_sha256(img2)
        hash3 = compute_sha256(img3)

        self.assertEqual(hash1, hash2, "Identical images must yield matching SHA-256 hashes")
        self.assertNotEqual(hash1, hash3, "Distinct images must yield different SHA-256 hashes")
        self.assertEqual(len(hash1), 64, "SHA-256 hash must be 64 hexadecimal characters")

    def test_parse_vlm_thai_lpr_output(self):
        """Verify parsing of Thai characters, digits, and province with DLT corrections."""
        raw_pred = "1ฃค 8888 กรุงเทพมหานคร"
        parsed = parse_vlm_thai_lpr_output(raw_pred)

        # ฃ must be corrected to ข
        self.assertEqual(parsed["chars"], "1ขค")
        self.assertEqual(parsed["numbers"], "8888")
        self.assertEqual(parsed["province"], "กรุงเทพมหานคร")
        self.assertEqual(parsed["canonical_string"], "1ขค 8888 กรุงเทพมหานคร")

    def test_offline_annotation_manifest_export(self):
        """Verify full offline annotation run generating JSON manifest."""
        manifest_path = self.temp_dir / "manifest.json"
        result_manifest = run_offline_annotation(
            dataset_dir=self.temp_dir,
            output_manifest_path=manifest_path,
            mock_inference=True,
        )

        self.assertTrue(manifest_path.exists())
        with manifest_path.open("r", encoding="utf-8") as f:
            saved_manifest = json.load(f)

        self.assertIsInstance(saved_manifest, dict)
        # Across AA, AB, ZZ: 2 unique images total -> exactly 2 deduplicated hashes in manifest
        self.assertEqual(len(saved_manifest), 2)
        for sha256_key, pred_str in saved_manifest.items():
            self.assertEqual(len(sha256_key), 64)
            self.assertIsInstance(pred_str, str)


class TestTask2CRNNAndWeightedLoss(unittest.TestCase):
    """Test suite for Task 2: CRNN Recognizer with Weighted Loss Multipliers."""

    def test_vocabulary_completeness(self):
        """Verify vocabulary definition and exact sizes."""
        # 1 blank + 10 digits + 41 Thai consonants + 78 provinces = 130 tokens
        self.assertEqual(VOCAB_SIZE, 130)
        self.assertEqual(VOCAB["<blank>"], 0)
        self.assertEqual(VOCAB["0"], 1)
        self.assertEqual(VOCAB["9"], 10)
        self.assertEqual(VOCAB["ก"], 11)
        self.assertIn("ฮ", VOCAB)
        self.assertIn("กรุงเทพมหานคร", VOCAB)
        self.assertIn("เบตง", VOCAB)

    def test_weighted_cross_entropy_multipliers(self):
        """Verify exact loss multipliers (3.0 for similar chars, 2.0 for provinces)."""
        loss_fn = ThaiWeightedCrossEntropyLoss()
        weights = loss_fn.weight

        # 1. Aligned length
        self.assertEqual(len(weights), VOCAB_SIZE)

        # 2. Visually similar characters (ก, ถ, ภ, ข, ช, ซ) must have weight 3.0
        for ch in ["ก", "ถ", "ภ", "ข", "ช", "ซ"]:
            idx = VOCAB[ch]
            self.assertAlmostEqual(weights[idx].item(), 3.0, places=5, msg=f"Character {ch} should have weight 3.0")

        # 3. Province identifiers must have weight 2.0
        for prov in OFFICIAL_PROVINCES:
            idx = VOCAB[prov]
            self.assertAlmostEqual(weights[idx].item(), 2.0, places=5, msg=f"Province {prov} should have weight 2.0")

        # 4. Standard digits and unpenalized consonants must have base weight 1.0
        for d in "0123456789":
            self.assertAlmostEqual(weights[VOCAB[d]].item(), 1.0, places=5)
        self.assertAlmostEqual(weights[VOCAB["ค"]].item(), 1.0, places=5)
        self.assertAlmostEqual(weights[VOCAB["ง"]].item(), 1.0, places=5)

    def test_crnn_forward_pass_and_backward(self):
        """Verify CRNN forward pass, device placement, and loss backpropagation."""
        device = get_optimal_device()
        model, loss_fn, _ = create_crnn_training_pipeline(device=device)

        batch_size = 2
        in_height = 32
        in_width = 128
        dummy_images = torch.randn(batch_size, 3, in_height, in_width, device=device)

        # Forward pass
        logits = model(dummy_images)
        seq_len = logits.shape[1]

        self.assertEqual(logits.shape[0], batch_size)
        self.assertEqual(logits.shape[2], VOCAB_SIZE)

        # Targets perfectly aligned with sequence length
        dummy_targets = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)

        # Loss calculation
        loss = loss_fn(logits, dummy_targets)
        self.assertEqual(loss.ndim, 0)
        self.assertFalse(torch.isnan(loss).item())
        self.assertFalse(torch.isinf(loss).item())

        # Backpropagation test
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} missing gradient")

    def test_automatic_device_placement(self):
        """Verify automatic device migration of loss weight buffer and model."""
        loss_fn = ThaiWeightedCrossEntropyLoss()
        self.assertEqual(loss_fn.weight.device.type, "cpu")

        if torch.cuda.is_available():
            cuda_device = torch.device("cuda:0")
            loss_fn_cuda = loss_fn.to(cuda_device)
            self.assertEqual(loss_fn_cuda.weight.device.type, "cuda")
            self.assertEqual(loss_fn_cuda.weight.device.index, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
