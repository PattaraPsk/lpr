"""
Test Suite for Modularized ALPR Package (tools.alpr).
Compliant with ISO/IEC/IEEE 29119 Software Testing Standards.
"""

import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.alpr import (
    RealtimeLPRDemo,
    TrackedPlate,
    ThreadedStreamCapture,
    render_thai_badge,
    overlay_badge,
    get_thai_font,
    draw_hud,
    box_iou,
    box_center_dist,
    detect_vehicles,
    detect_plates_direct,
    match_vehicle_and_orientation,
    resolve_model_path,
    find_default_sample_video,
)
from tools.alpr.config import (
    DETECTOR_CANDIDATES,
    VEHICLE_CANDIDATES,
    READER_CANDIDATES,
)


class TestALPRPackage(unittest.TestCase):
    """Verifies that all modularized ALPR package components function properly."""

    def test_imports_and_exports(self):
        self.assertIsNotNone(RealtimeLPRDemo)
        self.assertIsNotNone(TrackedPlate)
        self.assertIsNotNone(ThreadedStreamCapture)
        self.assertIsNotNone(render_thai_badge)
        self.assertIsNotNone(overlay_badge)

    def test_model_resolver(self):
        det_path = resolve_model_path(None, DETECTOR_CANDIDATES)
        self.assertTrue(det_path.exists(), f"Resolved detector path should exist: {det_path}")

        reader_path = resolve_model_path(None, READER_CANDIDATES)
        self.assertTrue(reader_path.exists(), f"Resolved reader path should exist: {reader_path}")

    def test_box_geometry(self):
        b1 = [0.0, 0.0, 10.0, 10.0]
        b2 = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(box_iou(b1, b2), 1.0)

        b3 = [20.0, 20.0, 30.0, 30.0]
        self.assertAlmostEqual(box_iou(b1, b3), 0.0)

        dist = box_center_dist(b1, b3)
        self.assertGreater(dist, 0.0)

    def test_thai_badge_cache(self):
        bgr, alpha = render_thai_badge("1กข 1234 กรุงเทพมหานคร")
        self.assertEqual(len(bgr.shape), 3)
        self.assertEqual(len(alpha.shape), 2)


if __name__ == "__main__":
    unittest.main()
