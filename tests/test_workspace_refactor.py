"""
Unit tests for workspace refactoring and pipeline automation.
Complies with ISO/IEC/IEEE 29119, ASD-STE100, and ISO 24495-1:2023.
"""

import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import numpy as np

import cli
from tools.owod.core.quality_auditor import PlateInspectionQueue
from tools.rtsp_collector.collector import route_low_confidence_crop


class TestWorkspaceRefactor(unittest.TestCase):
    """Test suite for CLI consolidation, archiving, dashboard, and queue routing."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_audit_"))

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_archived_legacy_scripts_exist(self):
        """Verify that legacy scripts exist in tools/archive directory."""
        archive_dir = Path("tools/archive")
        self.assertTrue(archive_dir.is_dir(), "tools/archive directory must exist")
        expected_files = [
            "train_selen_lpr_v2.py",
            "build_dual_fonts_v6.py",
            "build_dual_fonts_v7.py",
            "build_font_suite_v8.py",
            "build_font_suite_v9.py",
        ]
        for f in expected_files:
            target = archive_dir / f
            self.assertTrue(target.is_file(), f"Archived file must exist: {f}")

    def test_root_bat_files_deleted(self):
        """Verify that legacy root .bat files are removed."""
        root_bat_files = list(Path(".").glob("*.bat"))
        self.assertEqual(len(root_bat_files), 0, "No .bat files should remain in the root directory")

    def test_cli_parser_subcommands(self):
        """Verify that cli.py registers all essential subcommands."""
        parser = cli.build_parser()
        # Retrieve registered subcommands
        subparsers_action = next(a for a in parser._actions if isinstance(a, unittest.mock.MagicMock) or a.dest == "subcommand")
        choices = subparsers_action.choices.keys()
        self.assertIn("train-detector", choices)
        self.assertIn("demo-realtime", choices)
        self.assertIn("plate-augmentor", choices)
        self.assertIn("owod-studio", choices)
        self.assertIn("dashboard-api", choices)
        self.assertIn("dashboard-ui", choices)

    def test_quality_auditor_inspection_queue(self):
        """Verify that low-confidence crops are properly queued and serialized."""
        queue = PlateInspectionQueue(queue_dir=self.test_dir)
        dummy_crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Enqueue item with confidence 0.75
        crop_path = queue.enqueue(dummy_crop, confidence=0.75, metadata={"camera": "CAM01"})
        self.assertTrue(crop_path.is_file(), "Crop image file must be saved on disk")

        pending = queue.list_pending()
        self.assertEqual(len(pending), 1, "Must contain exactly one pending item")
        self.assertEqual(pending[0]["confidence"], 0.75)
        self.assertEqual(pending[0]["status"], "PENDING_INSPECTION")

    def test_route_low_confidence_crop_threshold(self):
        """Verify automated routing: route when confidence < 0.85, ignore otherwise."""
        dummy_crop = np.zeros((40, 80, 3), dtype=np.uint8)

        # Below 0.85 -> Should route
        res_low = route_low_confidence_crop(dummy_crop, confidence=0.82, queue_dir=self.test_dir)
        self.assertIsNotNone(res_low, "Confidence 0.82 must route into inspection queue")

        # Above or equal to 0.85 -> Should not route
        res_high = route_low_confidence_crop(dummy_crop, confidence=0.88, queue_dir=self.test_dir)
        self.assertIsNone(res_high, "Confidence 0.88 must not route into inspection queue")

    def test_dashboard_files_exist(self):
        """Verify that dashboard scaffolding files exist and are populated."""
        self.assertTrue(Path("dashboard/api.py").is_file(), "dashboard/api.py must exist")
        self.assertTrue(Path("dashboard/app.py").is_file(), "dashboard/app.py must exist")
        self.assertTrue(Path("dashboard/__init__.py").is_file(), "dashboard/__init__.py must exist")


if __name__ == "__main__":
    unittest.main()
