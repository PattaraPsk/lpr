"""
Unit test for Dataset Storage Organization (ISO 25012 / ISO 8601 Compliance).
Verifies:
1. 6-Tier Dataset Hierarchy: zip, raw, snapshot, labeled, verified, ready.
2. Version format in ready/ (YYYYMMDDhhmm).
3. Plate grouping in ready/ versions (images grouped in plate subfolders).
"""

import re
import json
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"


class TestDatasetOrganization(unittest.TestCase):

    def test_six_tier_hierarchy_exists(self):
        """Verifies that all 6 dataset tiers exist in dataset/."""
        expected_tiers = ["zip", "raw", "snapshot", "labeled", "verified", "ready"]
        for tier in expected_tiers:
            tier_path = DATASET_DIR / tier
            self.assertTrue(
                tier_path.exists() and tier_path.is_dir(),
                f"Dataset tier '{tier}' must exist in {DATASET_DIR}"
            )

    def test_ready_version_naming_convention(self):
        """Verifies that versions in dataset/ready/ adhere to YYYYMMDDhhmm."""
        ready_dir = DATASET_DIR / "ready"
        version_dirs = [d for d in ready_dir.iterdir() if d.is_dir()]
        self.assertGreater(len(version_dirs), 0, "dataset/ready/ must have at least one release version")

        timestamp_pattern = re.compile(r"^\d{12}$")  # 12 digits: YYYYMMDDhhmm
        for v in version_dirs:
            self.assertTrue(
                timestamp_pattern.match(v.name),
                f"Version folder '{v.name}' must follow 12-digit YYYYMMDDhhmm timestamp format"
            )

    def test_ready_version_plate_grouping(self):
        """Verifies that images in each ready/ version are grouped into plate subfolders."""
        ready_dir = DATASET_DIR / "ready"
        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        for v_dir in ready_dir.iterdir():
            if not v_dir.is_dir():
                continue

            # Must NOT have loose image files at the root of the version folder
            loose_images = [f for f in v_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
            self.assertEqual(
                len(loose_images), 0,
                f"Version {v_dir.name} has loose image files; all images must be in plate folders!"
            )

            # Must have plate subfolders containing images
            plate_subdirs = [d for d in v_dir.iterdir() if d.is_dir()]
            self.assertGreater(
                len(plate_subdirs), 0,
                f"Version {v_dir.name} should contain plate subfolders"
            )

            # Check that each subfolder contains images
            total_images = 0
            for p_dir in plate_subdirs:
                imgs = [f for f in p_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
                self.assertGreater(
                    len(imgs), 0,
                    f"Plate folder {p_dir.name} should not be empty"
                )
                total_images += len(imgs)

            self.assertGreater(total_images, 0, f"Version {v_dir.name} must have images")

            # Check version manifest if present
            manifest_file = v_dir / "version_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data.get("version"), v_dir.name)
                self.assertEqual(data.get("total_images"), total_images)


if __name__ == "__main__":
    unittest.main()
