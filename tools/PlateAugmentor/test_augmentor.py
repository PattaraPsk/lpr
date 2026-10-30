#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from plate_renderer import PLATE_CONFIGS, render_plate, generate_random_plate
from augmentor import AugmentationPipeline, PRESETS
from exporter import YOLOExporter

def test_pipeline():
    print("=== Testing plate_renderer ===")
    for ptype in PLATE_CONFIGS:
        img, bboxes, meta = generate_random_plate(ptype)
        print(f"  [OK] {ptype}: {meta['full_text']} | Size: {img.size} | BBoxes: {len(bboxes)}")
        assert img.width > 0 and img.height > 0
        assert len(bboxes) > 0

    print("\n=== Testing augmentations with presets ===")
    img, bboxes, meta = generate_random_plate("WhiteBlackCar2Line")
    for pname in PRESETS:
        pipeline = AugmentationPipeline()
        pipeline.set_preset(pname)
        aug_img, aug_boxes = pipeline.apply(img, bboxes)
        print(f"  [OK] Preset '{pname}': size={aug_img.size}, bboxes={len(aug_boxes)}")
        assert aug_img.size == img.size

    print("\n=== Testing YOLO exporter ===")
    test_out = ROOT_DIR / "runs" / "test_plate_augmentor"
    exporter = YOLOExporter(str(test_out), val_split=0.2)
    img_p, lbl_p = exporter.export_sample(aug_img, aug_boxes, "test_plate_01", metadata=meta, is_val=False)
    exporter.finalize()
    print(f"  [OK] Exported sample: {img_p}")
    print(f"  [OK] Exported label: {lbl_p}")
    assert img_p.exists()
    assert lbl_p.exists()
    assert (test_out / "data.yaml").exists()
    assert (test_out / "metadata.json").exists()

    # Read label content
    with open(lbl_p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  [OK] YOLO label lines count: {len(lines)}")
    print(f"  [OK] First line: {lines[0].strip()}")

    print("\n=== Testing PyQt6 Window Init ===")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"  # headless
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from main_window import PlateAugmentorWindow
    win = PlateAugmentorWindow()
    print(f"  [OK] Window successfully instantiated: '{win.windowTitle()}'")

    print("\nALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_pipeline()
