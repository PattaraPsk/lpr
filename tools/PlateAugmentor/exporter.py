#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset and YOLO Annotation Exporter
===================================
Exports synthetic/augmented license plate images alongside:
1. YOLO normalized bounding box format (.txt):
   Class indices:
     0: LicensePlate (full plate box)
     1: ThaiChar (individual Thai consonant)
     2: Digit (individual number 0-9)
2. data.yaml descriptor for YOLO training
3. metadata.json / csv for tracking ground truth text and properties
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

DIGITS = set("0123456789")


class YOLOExporter:
    """Exports augmented plate samples in YOLO format."""

    def __init__(
        self,
        output_dir: str,
        class_map: Optional[Dict[str, int]] = None,
        val_split: float = 0.15,
    ):
        self.output_dir = Path(output_dir)
        self.val_split = max(0.0, min(1.0, val_split))
        self.class_map = class_map or {
            "LicensePlate": 0,
            "ThaiChar": 1,
            "Digit": 2,
        }

        # Directories
        self.img_train = self.output_dir / "images" / "train"
        self.img_val = self.output_dir / "images" / "val"
        self.lbl_train = self.output_dir / "labels" / "train"
        self.lbl_val = self.output_dir / "labels" / "val"

        for d in [self.img_train, self.img_val, self.lbl_train, self.lbl_val]:
            d.mkdir(parents=True, exist_ok=True)

        self.metadata_records: List[Dict] = []

    def export_sample(
        self,
        image: Image.Image,
        char_bboxes: List[Dict],
        stem: str,
        metadata: Optional[Dict] = None,
        is_val: bool = False,
    ) -> Tuple[Path, Path]:
        """
        Export a single image and corresponding YOLO annotation file.
        char_bboxes format: [{"char": "ก", "bbox": (x1, y1, x2, y2)}, ...]
        """
        w, h = image.size
        img_dest_dir = self.img_val if is_val else self.img_train
        lbl_dest_dir = self.lbl_val if is_val else self.lbl_train

        img_path = img_dest_dir / f"{stem}.jpg"
        lbl_path = lbl_dest_dir / f"{stem}.txt"

        # Save image (convert to RGB if RGBA)
        if image.mode != "RGB":
            save_img = image.convert("RGB")
        else:
            save_img = image
        save_img.save(img_path, quality=95)

        lines = []
        # 1. Full plate box (0 0.5 0.5 1.0 1.0)
        plate_cls = self.class_map.get("LicensePlate", 0)
        lines.append(f"{plate_cls} 0.500000 0.500000 1.000000 1.000000")

        # 2. Individual character boxes
        for item in char_bboxes:
            ch = item["char"]
            x1, y1, x2, y2 = item["bbox"]

            # Clamp
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx = (x1 + x2) / (2.0 * w)
            cy = (y1 + y2) / (2.0 * h)

            cls_id = self.class_map.get("Digit", 2) if ch in DIGITS else self.class_map.get("ThaiChar", 1)
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Record metadata
        rec = {
            "filename": img_path.name,
            "split": "val" if is_val else "train",
            "width": w,
            "height": h,
            "num_chars": len(char_bboxes),
        }
        if metadata:
            rec.update(metadata)
        self.metadata_records.append(rec)

        return img_path, lbl_path

    def finalize(self):
        """Write data.yaml and metadata files."""
        # 1. data.yaml
        resolved_path = str(self.output_dir.resolve()).replace("\\", "/")
        yaml_lines = [
            f"path: {resolved_path}",
            "train: images/train",
            "val: images/val",
            "names:",
        ]
        # Invert class map
        inv_map = {v: k for k, v in self.class_map.items()}
        for k in sorted(inv_map.keys()):
            yaml_lines.append(f"  {k}: {inv_map[k]}")

        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yaml_lines) + "\n")

        # 2. metadata.json
        meta_json = self.output_dir / "metadata.json"
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(self.metadata_records, f, ensure_ascii=False, indent=2)

        # 3. metadata.csv
        if self.metadata_records:
            meta_csv = self.output_dir / "metadata.csv"
            keys = list(self.metadata_records[0].keys())
            with open(meta_csv, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.metadata_records)
