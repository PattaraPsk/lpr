import unittest
from pathlib import Path
import cv2

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.design_classifier import PlateDesignClassifier
from src.core.vehicle_mapper import VehicleTypeMapper
from src.schema.types import PlateDesign


class TestClassification(unittest.TestCase):

    def setUp(self):
        self.classifier = PlateDesignClassifier()
        self.mapper = VehicleTypeMapper()
        self.samples_dir = Path(__file__).resolve().parent.parent / "samples"

    def test_sample_categories_exist(self):
        for idx in range(1, 9):
            folder = list(self.samples_dir.glob(f"0{idx}_*"))
            self.assertTrue(len(folder) > 0, f"Sample folder 0{idx} not found")
            sample_file = folder[0] / "sample.png"
            self.assertTrue(sample_file.exists(), f"Sample image {sample_file} not found")

    def test_vehicle_mapper_rules(self):
        # Pickup rule check
        design_pickup = PlateDesign(background_color="white", text_color="green", layout="standard_single_row")
        v_pickup = self.mapper.map_vehicle(design_pickup, d3_prefix="บป", aspect_ratio=2.2)
        self.assertEqual(v_pickup.category_id, "private_pickup")

        # Taxi rule check
        design_taxi = PlateDesign(background_color="yellow", text_color="black", layout="standard_single_row")
        v_taxi = self.mapper.map_vehicle(design_taxi, d3_prefix="ทค", aspect_ratio=2.2)
        self.assertEqual(v_taxi.category_id, "public_taxi")

        # Motorcycle aspect ratio check
        design_mc = PlateDesign(background_color="white", text_color="black", layout="motorcycle_double_row")
        v_mc = self.mapper.map_vehicle(design_mc, d3_prefix="1กข", aspect_ratio=1.1)
        self.assertEqual(v_mc.category_id, "motorcycle")


if __name__ == "__main__":
    unittest.main()
