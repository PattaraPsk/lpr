import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.adaptive_pipeline import SelenLPRPipeline


class TestPipelineIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.ocr_weights = cls.root_dir / "models" / "ocr" / "thai_plate_attn_ocr.pt"
        cls.pipeline = SelenLPRPipeline(
            ocr_weights_path=str(cls.ocr_weights) if cls.ocr_weights.exists() else None
        )
        cls.sample_path = cls.root_dir / "samples" / "01_private_car" / "sample.png"

    def test_pipeline_on_sample_car(self):
        self.assertTrue(self.sample_path.exists(), f"Sample {self.sample_path} not found")
        img = cv2.imread(str(self.sample_path))
        self.assertIsNotNone(img)

        results = self.pipeline.process_frame(img)
        self.assertTrue(len(results) > 0, "No detections returned by pipeline")

        det = results[0]
        # Verify normalized coordinates
        self.assertTrue(0.0 <= det.normalized_coords.s_x <= 1.0)
        self.assertTrue(0.0 <= det.normalized_coords.s_y <= 1.0)
        self.assertTrue(0.0 <= det.normalized_coords.s_w <= 1.0)
        self.assertTrue(0.0 <= det.normalized_coords.s_h <= 1.0)

        # Verify plate design & vehicle type fields exist
        self.assertIn(det.plate_design.background_color, ["white", "yellow", "orange", "green"])
        self.assertIn(det.plate_design.text_color, ["black", "green", "blue", "white"])
        self.assertIsNotNone(det.vehicle_type.category_id)

    def test_pipeline_on_sample_pickup(self):
        pickup_path = self.root_dir / "samples" / "02_private_pickup" / "sample.png"
        img = cv2.imread(str(pickup_path))
        results = self.pipeline.process_frame(img)
        self.assertTrue(len(results) > 0)
        det = results[0]
        self.assertEqual(det.vehicle_type.category_id, "private_pickup")
        self.assertEqual(det.plate_design.text_color, "green")


if __name__ == "__main__":
    unittest.main()
