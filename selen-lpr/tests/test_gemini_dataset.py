import unittest
import csv
from pathlib import Path
import json

DATASET_DIR = Path(r"D:\Projects-anti\lpr\dataset\labeled_gemini_flash")
LABELS_CSV = DATASET_DIR / "labels.csv"
LABELS_DETAILED_CSV = DATASET_DIR / "labels_detailed.csv"
IMAGES_DIR = DATASET_DIR / "images"
INFO_JSON = DATASET_DIR / "dataset_info.json"


class TestGeminiFlashDataset(unittest.TestCase):
    def test_dataset_files_exist(self):
        self.assertTrue(DATASET_DIR.exists(), f"Dataset directory {DATASET_DIR} missing")
        self.assertTrue(LABELS_CSV.exists(), f"Labels CSV {LABELS_CSV} missing")
        self.assertTrue(LABELS_DETAILED_CSV.exists(), f"Detailed labels CSV {LABELS_DETAILED_CSV} missing")
        self.assertTrue(IMAGES_DIR.exists(), f"Images directory {IMAGES_DIR} missing")
        self.assertTrue(INFO_JSON.exists(), f"Dataset info {INFO_JSON} missing")

    def test_all_images_present(self):
        with open(LABELS_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreaterEqual(len(rows), 40, "Dataset should have at least 40 verified samples")
        for row in rows:
            img_path = IMAGES_DIR / row["IMAGE_NAME"]
            self.assertTrue(img_path.exists(), f"Image file missing: {img_path}")
            self.assertGreater(img_path.stat().st_size, 500, f"Image file corrupted/empty: {img_path}")
            self.assertTrue(len(row["3DIGIT"]) > 0, f"Empty 3DIGIT prefix for {row['IMAGE_NAME']}")
            self.assertTrue(len(row["4DIGIT"]) > 0, f"Empty 4DIGIT digits for {row['IMAGE_NAME']}")
            self.assertTrue(len(row["PROVINCE"]) > 0, f"Empty PROVINCE for {row['IMAGE_NAME']}")

    def test_naming_convention(self):
        """Verify filenames strictly follow plate_xxx_vehicle type_d3_d4_province.jpg format."""
        with open(LABELS_CSV, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        valid_types = {"sedan", "hatchback", "pickup", "van", "motorcycle", "truck", "bus", "suv"}
        for idx, row in enumerate(rows, 1):
            fname = row["IMAGE_NAME"]
            self.assertTrue(fname.startswith("plate_"), f"Filename must start with plate_: {fname}")
            self.assertTrue(fname.endswith(".jpg"), f"Filename must end with .jpg: {fname}")
            
            # Remove prefix and extension
            body = fname[len("plate_"):-len(".jpg")]
            parts = body.split("_")
            self.assertEqual(len(parts), 5, f"Filename must contain 5 underscore-delimited parts after 'plate_': {fname}")
            
            p_idx, p_vtype, p_d3, p_d4, p_prov = parts
            self.assertEqual(p_idx, f"{idx:03d}", f"Plate index mismatch: expected {idx:03d}, got {p_idx}")
            self.assertIn(p_vtype, valid_types, f"Invalid vehicle type '{p_vtype}' in {fname}")
            self.assertEqual(p_d3, row["3DIGIT"], f"d3 mismatch in {fname}")
            self.assertEqual(p_d4, row["4DIGIT"], f"d4 mismatch in {fname}")
            self.assertEqual(p_prov, row["PROVINCE"], f"Province mismatch in {fname}")


if __name__ == "__main__":
    unittest.main()
