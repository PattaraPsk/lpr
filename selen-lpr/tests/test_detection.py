import unittest
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.detector import AdaptivePlateDetector
from src.schema.types import NormalizedCoordinates


class TestDetection(unittest.TestCase):

    def setUp(self):
        self.detector = AdaptivePlateDetector(use_adaptive_roi=False)

    def test_normalized_coordinates_bounds(self):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = self.detector.detect(dummy_frame)
        self.assertTrue(len(results) > 0)
        crop, s_coords, conf = results[0]
        self.assertTrue(0.0 <= s_coords.s_x <= 1.0)
        self.assertTrue(0.0 <= s_coords.s_y <= 1.0)
        self.assertTrue(0.0 <= s_coords.s_w <= 1.0)
        self.assertTrue(0.0 <= s_coords.s_h <= 1.0)


if __name__ == "__main__":
    unittest.main()
