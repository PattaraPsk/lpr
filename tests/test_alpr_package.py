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
    MotionDetector,
    compute_crop_sharpness,
    detect_moving_vehicles,
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

    def test_motion_detector_and_sharpness(self):
        md = MotionDetector(history=10, min_motion_area=100)
        # Static frames
        f1 = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(5):
            md.update(f1)
        # Moving object frame
        f2 = f1.copy()
        f2[100:200, 200:350] = 255
        has_motion, boxes, _ = md.update(f2)
        self.assertTrue(has_motion, "Should detect motion on changed region")
        self.assertGreaterEqual(len(boxes), 1)

        # Spatial motion overlap check
        moving_box = [200.0, 100.0, 350.0, 200.0]
        stationary_box = [10.0, 10.0, 50.0, 50.0]
        self.assertTrue(md.is_moving(moving_box, boxes))
        self.assertFalse(md.is_moving(stationary_box, boxes))

        # Sharpness score
        sharpness_zero = compute_crop_sharpness(f1)
        self.assertAlmostEqual(sharpness_zero, 0.0)
        sharp_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertGreater(compute_crop_sharpness(sharp_crop), 10.0)


if __name__ == "__main__":
    unittest.main()
