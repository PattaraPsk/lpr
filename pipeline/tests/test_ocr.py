"""
tests/test_ocr.py
Unit tests for plate deskewing and character band segmentation.
"""
import unittest
import numpy as np
from src.ocr.segmenter import PlateSegmenter


class TestPlateSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = PlateSegmenter(target_char_height=128, filter_screws=True)

    def test_empty_or_none_crop(self):
        chars, prov, angle = self.segmenter.segment(None)
        self.assertEqual(len(chars), 0)
        self.assertIsNone(prov)
        self.assertEqual(angle, 0.0)

    def test_synthetic_plate_segmentation(self):
        # Create a synthetic white plate canvas with simulated dark glyphs
        plate = np.full((150, 400, 3), 245, dtype=np.uint8)

        # Simulate top band characters
        plate[25:75, 40:80] = 20    # char 1
        plate[25:75, 100:140] = 20  # char 2
        plate[25:75, 200:240] = 20  # char 3
        plate[25:75, 260:300] = 20  # char 4

        # Simulate bottom province text
        plate[100:130, 120:280] = 20

        chars, prov, angle = self.segmenter.segment(plate)
        self.assertIsNotNone(prov)
        self.assertGreaterEqual(len(chars), 1)
        for c in chars:
            self.assertEqual(c.shape[0], 128)  # Target height 128px


if __name__ == "__main__":
    unittest.main()
