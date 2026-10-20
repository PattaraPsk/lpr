"""
tests/test_detector.py
Unit tests for bounding box calculations and boundary-clamped margin padding.
"""
import unittest
import numpy as np
from src.pipeline.types import BoundingBox
from src.detector.base_detector import crop_plate_with_padding


class TestDetectorUtils(unittest.TestCase):
    def test_bounding_box_properties(self):
        bbox = BoundingBox(xmin=100, ymin=150, xmax=300, ymax=250, confidence=0.88)
        self.assertEqual(bbox.width, 200)
        self.assertEqual(bbox.height, 100)
        self.assertEqual(bbox.area, 20000)
        self.assertEqual(bbox.aspect_ratio, 2.0)
        self.assertEqual(bbox.center, (200, 200))
        self.assertEqual(bbox.to_xyxy(), [100, 150, 300, 250])
        self.assertEqual(bbox.to_xywh(), [100, 150, 200, 100])

    def test_crop_with_padding_clamping(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Plate near corner - ensure no IndexError or negative slices
        corner_box = BoundingBox(xmin=5, ymin=5, xmax=100, ymax=60)
        crop = crop_plate_with_padding(frame, corner_box, padding_ratio=0.10)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

        # Normal plate crop
        center_box = BoundingBox(xmin=400, ymin=300, xmax=600, ymax=400)
        crop = crop_plate_with_padding(frame, center_box, padding_ratio=0.06)
        expected_w = 200 + 2 * int(round(200 * 0.06))
        expected_h = 100 + 2 * int(round(100 * 0.06))
        self.assertEqual(crop.shape[1], expected_w)
        self.assertEqual(crop.shape[0], expected_h)


if __name__ == "__main__":
    unittest.main()
