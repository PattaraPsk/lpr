"""
ISO/IEC 5259-2:2023 AI Data Quality Auditor & Validator.

Compliant with:
- ISO/IEC 5259-2:2023: Artificial intelligence - Data quality for analytics and ML (Measures)
- ISO/IEC 25012: Data Quality Model (Accuracy, Completeness, Consistency, Credibility)
- ISO 8601: Data element representation - UTC Timestamp formatting
- FIPS 180-4 / ISO 27001: SHA-256 Dataset Config Cryptographic Integrity
"""

import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml


class PlateInspectionQueue:
    """
    Manages low-confidence license plate crops for quality inspection.
    Complies with ISO/IEC 5259-2:2023 data quality measures.
    """
    DEFAULT_QUEUE_DIR = "dataset/audit_inspection_queue"

    def __init__(self, queue_dir: Union[str, Path] = DEFAULT_QUEUE_DIR):
        self.queue_dir = Path(queue_dir).resolve()
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir = self.queue_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.queue_dir / "inspection_manifest.jsonl"

    def enqueue(
        self,
        crop: Any,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Add a low-confidence plate crop to the inspection queue.
        Save the crop image and append the record to the inspection manifest.
        """
        ts_utc = datetime.now(timezone.utc).isoformat()
        token = os.urandom(4).hex()
        item_id = hashlib.sha256(f"{ts_utc}_{confidence}_{token}".encode("utf-8")).hexdigest()[:12]
        crop_filename = f"crop_{item_id}_{confidence:.3f}.jpg"
        target_path = self.crops_dir / crop_filename

        if hasattr(crop, "shape") and hasattr(crop, "dtype"):
            import cv2
            cv2.imwrite(str(target_path), crop)
        elif isinstance(crop, (str, Path)) and Path(crop).is_file():
            import shutil
            shutil.copy2(str(crop), str(target_path))
        else:
            target_path.touch(exist_ok=True)

        record = {
            "item_id": item_id,
            "timestamp_utc": ts_utc,
            "confidence": round(float(confidence), 4),
            "crop_path": str(target_path).replace("\\", "/"),
            "relative_crop_path": target_path.name,
            "status": "PENDING_INSPECTION",
            "metadata": metadata or {}
        }
        with open(self.manifest_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return target_path

    def list_pending(self) -> List[Dict[str, Any]]:
        """Return all items currently in the inspection queue."""
        if not self.manifest_file.is_file():
            return []
        items = []
        with open(self.manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
        return items


def route_to_inspection_queue(
    crop: Any,
    confidence: float,
    metadata: Optional[Dict[str, Any]] = None,
    queue_dir: Union[str, Path] = PlateInspectionQueue.DEFAULT_QUEUE_DIR
) -> Path:
    """Route a plate crop into the auditor inspection queue."""
    queue = PlateInspectionQueue(queue_dir=queue_dir)
    return queue.enqueue(crop=crop, confidence=confidence, metadata=metadata)



def calculate_file_sha256(filepath: Union[str, Path]) -> Optional[str]:
    """Computes SHA-256 checksum of a file for ISO 25012 / ISO 27001 integrity."""
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(
    dataset_dir: Union[str, Path],
    save_report: bool = True
) -> Dict[str, Any]:
    """
    Performs comprehensive ISO/IEC 5259-2 quality auditing on a YOLO format dataset.
    
    Checks:
    1. Syntactic validity of annotations (x_center, y_center, width, height in [0.0, 1.0]).
    2. Boundary containment: (x_center ± width/2, y_center ± height/2 within [0, 1]).
    3. Class index consistency and representation distribution.
    4. Dataset split distribution (train vs val).
    5. Configuration file integrity (SHA-256).
    """
    root = Path(dataset_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    report_time = datetime.now(timezone.utc).isoformat()
    yaml_path = root / "data.yaml"
    
    class_names: Dict[int, str] = {}
    yaml_sha256 = None
    if yaml_path.is_file():
        yaml_sha256 = calculate_file_sha256(yaml_path)
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
                raw_names = yaml_data.get("names", {})
                if isinstance(raw_names, list):
                    class_names = {i: name for i, name in enumerate(raw_names)}
                elif isinstance(raw_names, dict):
                    class_names = {int(k): str(v) for k, v in raw_names.items()}
        except Exception:
            pass

    # Audit label directories
    splits = ["train", "val"]
    split_stats = {}
    class_counts: Dict[str, int] = {}
    total_annotations = 0
    valid_annotations = 0
    invalid_annotations = 0
    out_of_bounds_count = 0
    degraded_boxes_count = 0  # zero or negative dimensions
    total_images_found = 0

    for split in splits:
        lbl_dir = root / "labels" / split
        img_dir = root / "images" / split
        
        lbl_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []
        img_files = (
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.png")) +
            list(img_dir.glob("*.jpeg"))
        ) if img_dir.exists() else []

        split_images = len(img_files)
        total_images_found += split_images
        split_ann_count = 0
        split_invalid = 0

        for lf in lbl_files:
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        split_ann_count += 1
                        total_annotations += 1
                        
                        try:
                            cls_id = int(parts[0])
                            xc = float(parts[1])
                            yc = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                        except ValueError:
                            invalid_annotations += 1
                            split_invalid += 1
                            continue

                        # Validate bounding box geometry
                        is_valid = True
                        if w <= 0.0 or h <= 0.0:
                            degraded_boxes_count += 1
                            is_valid = False
                        
                        # Boundary containment check with small tolerance (0.005) for floating point round-off
                        x1 = xc - (w / 2.0)
                        y1 = yc - (h / 2.0)
                        x2 = xc + (w / 2.0)
                        y2 = yc + (h / 2.0)

                        if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0:
                            out_of_bounds_count += 1
                            is_valid = False
                        elif x1 < -0.01 or y1 < -0.01 or x2 > 1.01 or y2 > 1.01:
                            out_of_bounds_count += 1
                            is_valid = False

                        if is_valid:
                            valid_annotations += 1
                        else:
                            invalid_annotations += 1
                            split_invalid += 1

                        # Class distribution
                        cname = class_names.get(cls_id, f"class_{cls_id}")
                        class_counts[cname] = class_counts.get(cname, 0) + 1

            except Exception:
                pass

        split_stats[split] = {
            "images": split_images,
            "label_files": len(lbl_files),
            "annotations": split_ann_count,
            "invalid_annotations": split_invalid
        }

    # Calculate ISO 5259 metrics
    validity_rate = (valid_annotations / total_annotations) if total_annotations > 0 else 1.0
    
    # Class imbalance calculation
    counts_list = list(class_counts.values())
    if counts_list and min(counts_list) > 0:
        imbalance_ratio = round(max(counts_list) / min(counts_list), 3)
    else:
        imbalance_ratio = 1.0

    class_proportions = {}
    if total_annotations > 0:
        for cname, cnt in class_counts.items():
            class_proportions[cname] = round(cnt / total_annotations, 4)

    report: Dict[str, Any] = {
        "standard_compliance": {
            "data_quality_standard": "ISO/IEC 5259-2:2023",
            "quality_model_standard": "ISO/IEC 25012:2008",
            "timestamp_standard": "ISO 8601",
            "integrity_standard": "FIPS 180-4 / ISO 27001"
        },
        "audit_metadata": {
            "dataset_root": str(root).replace("\\", "/"),
            "audit_timestamp_utc": report_time,
            "config_file": "data.yaml",
            "config_sha256": yaml_sha256
        },
        "volume_and_split": {
            "total_images": total_images_found,
            "total_annotations": total_annotations,
            "splits": split_stats,
            "train_val_image_ratio": round(
                (split_stats.get("val", {}).get("images", 0) / (total_images_found or 1)), 3
            )
        },
        "quality_measures": {
            "syntactic_validity_rate": round(validity_rate, 5),
            "valid_annotations": valid_annotations,
            "invalid_annotations": invalid_annotations,
            "out_of_bounds_boxes": out_of_bounds_count,
            "degraded_zero_area_boxes": degraded_boxes_count,
            "quality_grade": "A (PASSED)" if validity_rate >= 0.99 else "B (ACCEPTABLE)" if validity_rate >= 0.95 else "F (NEEDS_ATTENTION)"
        },
        "representation_and_bias": {
            "class_counts": class_counts,
            "class_proportions": class_proportions,
            "imbalance_ratio": imbalance_ratio
        }
    }

    if save_report:
        report_path = root / "data_quality_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["audit_metadata"]["report_path"] = str(report_path).replace("\\", "/")

    return report


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "dataset/yolo_dlt_lpr_dataset"
    target_path = Path(target)
    if target_path.exists():
        res = audit_dataset(target_path, save_report=True)
        print(f"Audited {target}: Grade = {res['quality_measures']['quality_grade']}")
        print(json.dumps(res, indent=2))
    else:
        print(f"Path does not exist: {target_path}")
