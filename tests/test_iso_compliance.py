"""
ISO & Enterprise Global Standards Compliance Test Suite
According to ISO/IEC/IEEE 29119 Software Testing Standards.

Verifies:
1. ISO 3166-2:TH & Unicode NFC (tools/DataManipulator/app/provinces.py)
2. ISO 8601 & ISO 25012 Data Integrity (tools/detect_cam_videos.py & quality_auditor.py)
3. ISO 27001 Credential & Path Sanitization (tools/rtsp_collector/security.py)
4. ISO/IEC 5259-2:2023 AI Data Quality Auditing (tools/owod/core/quality_auditor.py)
5. ISO/IEC 42001 AI Model Governance (.gitignore enforcement)
"""

import os
import sys
import tempfile
import unittest
import unicodedata
import hashlib
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.DataManipulator.app.provinces import (
    PROVINCES_ISO_DATA,
    get_iso_code,
    get_province_by_iso,
    normalize_thai_script,
    match_province_value
)
from tools.detect_cam_videos import (
    calculate_sha256,
    format_iso_duration,
    sanitize_path
)
from tools.rtsp_collector.security import mask_rtsp_url
from tools.owod.core.quality_auditor import audit_dataset, calculate_file_sha256


class TestISO3166AndUnicode(unittest.TestCase):
    """Verifies ISO 3166-2:TH (Thai Provinces) and ISO/IEC 10646 / Unicode NFC."""

    def test_all_77_provinces_registered(self):
        """Thailand officially has 76 provinces + 1 special administrative area (Bangkok)."""
        self.assertEqual(len(PROVINCES_ISO_DATA), 77, "Must contain exactly 77 provinces")

    def test_iso_code_format(self):
        """All ISO 3166-2 codes must match the format 'TH-XX'."""
        for thai_name, en_name, code in PROVINCES_ISO_DATA:
            self.assertTrue(
                code.startswith("TH-") and len(code) == 5 and code[3:].isdigit(),
                f"Invalid ISO 3166-2 code '{code}' for province '{thai_name}' ({en_name})"
            )

    def test_key_province_lookups(self):
        """Tests lookup by Thai name, English name, and ISO code."""
        self.assertEqual(get_iso_code("กรุงเทพมหานคร"), "TH-10")
        self.assertEqual(get_iso_code("Bangkok"), "TH-10")
        self.assertEqual(get_iso_code("เชียงใหม่"), "TH-50")
        self.assertEqual(get_iso_code("Chiang Mai"), "TH-50")
        self.assertEqual(get_iso_code("ภูเก็ต"), "TH-83")
        self.assertEqual(get_iso_code("Phuket"), "TH-83")
        self.assertEqual(get_iso_code("ชลบุรี"), "TH-20")
        self.assertEqual(get_iso_code("Chonburi"), "TH-20")

    def test_get_province_by_iso(self):
        """Lookup by ISO code returns canonical tuple."""
        th, en = get_province_by_iso("TH-10")
        self.assertEqual(th, "กรุงเทพมหานคร")
        self.assertEqual(en, "Bangkok")

        th, en = get_province_by_iso("TH-50")
        self.assertEqual(th, "เชียงใหม่")
        self.assertEqual(en, "Chiang Mai")

    def test_unicode_nfc_normalization(self):
        """Verifies that decomposed Thai Unicode sequences are normalized to NFC."""
        raw_text = "เชียงใหม่"
        normalized = normalize_thai_script(raw_text)
        self.assertEqual(normalized, unicodedata.normalize("NFC", raw_text))
        self.assertEqual(unicodedata.is_normalized("NFC", normalized), True)

    def test_match_province_helper(self):
        """Verifies fuzzy and bilingual matching to canonical Thai province name."""
        self.assertEqual(match_province_value("Bangkok"), "กรุงเทพมหานคร")
        self.assertEqual(match_province_value("TH-10"), "กรุงเทพมหานคร")
        self.assertEqual(match_province_value("เชียงใหม่"), "เชียงใหม่")


class TestISO8601AndISO25012(unittest.TestCase):
    """Verifies ISO 8601 (Time & Duration) and ISO 25012 (Data Integrity / Hashing)."""

    def test_iso8601_duration_formatting(self):
        """Checks format_iso_duration generates valid ISO 8601 duration string."""
        self.assertEqual(format_iso_duration(0), "PT0S")
        self.assertEqual(format_iso_duration(45), "PT45S")
        self.assertEqual(format_iso_duration(90), "PT1M30S")
        self.assertEqual(format_iso_duration(3665), "PT1H1M5S")

    def test_iso25012_sha256_integrity(self):
        """Verifies chunked SHA-256 calculation matches standard hash values."""
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(b"ISO25012 Data Integrity Test Payload")
            tf_path = Path(tf.name)

        try:
            expected_hash = hashlib.sha256(b"ISO25012 Data Integrity Test Payload").hexdigest()
            computed_hash1 = calculate_sha256(tf_path)
            computed_hash2 = calculate_file_sha256(tf_path)
            self.assertEqual(computed_hash1, expected_hash)
            self.assertEqual(computed_hash2, expected_hash)
        finally:
            if tf_path.exists():
                tf_path.unlink()


class TestISO27001Security(unittest.TestCase):
    """Verifies ISO/IEC 27001 Credential Masking & Sensitive Path Sanitization."""

    def test_mask_rtsp_url(self):
        """RTSP URL credentials must be masked with '***'."""
        raw_url = "rtsp://admin:supersecretpassword@192.168.1.50:554/live/ch0"
        masked = mask_rtsp_url(raw_url)
        self.assertNotIn("supersecretpassword", masked)
        self.assertIn("admin:***", masked)
        self.assertIn("192.168.1.50:554/live/ch0", masked)

    def test_sanitize_path(self):
        """Internal user paths must be masked."""
        windows_path = "C:/Users/JohnDoe/AppData/Local/Temp/video.mp4"
        masked = sanitize_path(windows_path)
        self.assertNotIn("JohnDoe", masked)


class TestISO5259DataQualityAuditor(unittest.TestCase):
    """Verifies ISO/IEC 5259-2:2023 AI Data Quality Auditing."""

    def test_audit_dataset_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "labels" / "train").mkdir(parents=True)
            (root / "labels" / "val").mkdir(parents=True)
            (root / "images" / "train").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)

            # Valid YOLO annotation: class xc yc w h
            (root / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.1\n0 0.2 0.3 0.1 0.05\n", encoding="utf-8")
            (root / "images" / "train" / "img1.jpg").write_text("img_data", encoding="utf-8")

            # Out of bounds annotation
            (root / "labels" / "val" / "img2.txt").write_text("1 1.5 0.5 0.2 0.1\n", encoding="utf-8")
            (root / "images" / "val" / "img2.jpg").write_text("img_data", encoding="utf-8")

            # data.yaml
            (root / "data.yaml").write_text("names:\n  0: car_plate\n  1: motorcycle_plate\n", encoding="utf-8")

            report = audit_dataset(root, save_report=True)

            # Verify schema compliance
            self.assertIn("standard_compliance", report)
            self.assertEqual(report["standard_compliance"]["data_quality_standard"], "ISO/IEC 5259-2:2023")
            self.assertEqual(report["volume_and_split"]["total_images"], 2)
            self.assertEqual(report["volume_and_split"]["total_annotations"], 3)
            self.assertEqual(report["quality_measures"]["valid_annotations"], 2)
            self.assertEqual(report["quality_measures"]["invalid_annotations"], 1)
            self.assertEqual(report["quality_measures"]["out_of_bounds_boxes"], 1)
            self.assertTrue((root / "data_quality_report.json").exists())


class TestAIModelGovernance(unittest.TestCase):
    """Verifies ISO/IEC 42001 Model Governance & Weights Exclusion in Git."""

    def test_gitignore_contains_model_weights(self):
        gitignore_path = PROJECT_ROOT / ".gitignore"
        self.assertTrue(gitignore_path.is_file(), ".gitignore must exist")
        content = gitignore_path.read_text(encoding="utf-8")
        self.assertIn("*.pt", content, ".gitignore must exclude *.pt weights")
        self.assertIn("*.onnx", content, ".gitignore must exclude *.onnx weights")
        self.assertIn("*.engine", content, ".gitignore must exclude *.engine weights")

    def test_model_catalog_exists(self):
        catalog_path = PROJECT_ROOT / "models" / "MODEL_CATALOG.md"
        self.assertTrue(catalog_path.is_file(), "models/MODEL_CATALOG.md must exist")
        content = catalog_path.read_text(encoding="utf-8")
        self.assertIn("ISO/IEC 42001", content)
        self.assertIn("front-rear-lpr-yolov11n", content)
        self.assertIn("thai-lpr-reader-a1.3.4.1", content)


if __name__ == "__main__":
    unittest.main()
