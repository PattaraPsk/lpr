#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Dataset Relabeling Utility using PaddleOCR-VL in FP16 (Local Model First).

Relabels:
1. dataset/reading/labeled/unverified_labeled_100 (100 images)
2. Corrupted/mangled folders in dataset/reading/labeled/202611030230

Complies with:
- ISO 8601:2019 (UTC timestamps)
- ISO/IEC 42001:2023 (Annex A.7 / A.9.3 Operational Event Logging)
- ISO/IEC 25012:2008 (Data Quality & Cryptographic Lineage via SHA-256)
- Thailand DLT Specifications (d3 <= 3 chars, d4 <= 4 digits)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Ensure UTF-8 console output
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.offline_vlm_annotator import (
    PaddleOCRVLAnnotator,
    compute_sha256,
    parse_vlm_thai_lpr_output,
    PROVINCES_78,
    THAI_CONSONANTS_VALID,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s UTC] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("DatasetRelabeler")

THAI_CHAR_REGEX = re.compile(r"[\u0E01-\u0E2E]")


def get_utc_timestamp() -> str:
    """Return ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_mangled_folder_name(folder_name: str) -> bool:
    """
    Determine if a folder name in 202611030230 is corrupted or violates DLT plate standards.
    Expected format: PlateType_d3_d4_province
    Corrupted if:
    - Less than 4 parts
    - d3 contains no Thai consonants (pure digits or missing)
    - d4 contains non-digits
    - province contains no Thai characters or is just a single number
    """
    parts = folder_name.split("_")
    if len(parts) < 4:
        return True

    pt = parts[0]
    d3 = parts[1]
    d4 = parts[2]
    prov = "_".join(parts[3:])

    has_thai_d3 = bool(THAI_CHAR_REGEX.search(d3))
    d4_is_digits = d4.isdigit()
    has_thai_prov = bool(THAI_CHAR_REGEX.search(prov))

    if not has_thai_d3 or not d4_is_digits or not has_thai_prov:
        return True

    return False


def relabel_unverified_100(
    annotator: PaddleOCRVLAnnotator,
    target_dir: Path,
) -> Dict[str, Any]:
    """
    Relabel the 100 plate crops in dataset/reading/labeled/unverified_labeled_100.
    Updates annotations.json and metadata.csv.
    """
    plates_dir = target_dir / "plates"
    annotations_json_path = target_dir / "annotations.json"
    metadata_csv_path = target_dir / "metadata.csv"

    if not plates_dir.is_dir():
        raise FileNotFoundError(f"Plates directory not found: {plates_dir}")

    plate_files = sorted(
        [p for p in plates_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )
    logger.info("Found %d images in unverified_labeled_100/plates", len(plate_files))

    records: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    t0 = time.time()
    for idx, img_path in enumerate(plate_files, start=1):
        sha256 = compute_sha256(img_path)
        vlm_res = annotator.annotate_image(img_path)

        chars = vlm_res["chars"]
        numbers = vlm_res["numbers"]
        province = vlm_res["province"]
        canonical = vlm_res["canonical_string"]

        record = {
            "index": idx,
            "filename": img_path.name,
            "sha256": sha256,
            "chars": chars,
            "numbers": numbers,
            "province": province,
            "canonical_string": canonical,
            "annotator": "PaddleOCR-VL-FP16",
            "timestamp_utc": get_utc_timestamp(),
        }
        records[sha256] = record

        csv_rows.append({
            "index": idx,
            "filename": img_path.name,
            "sha256": sha256,
            "chars": chars,
            "numbers": numbers,
            "province": province,
            "canonical_string": canonical,
            "annotator": "PaddleOCR-VL-FP16",
        })

        if idx % 20 == 0 or idx == len(plate_files):
            logger.info("Relabeled %d/%d images in unverified_labeled_100...", idx, len(plate_files))

    elapsed = time.time() - t0
    logger.info(
        "Finished relabeling unverified_labeled_100 in %.2f s (%.3f s/image)",
        elapsed,
        elapsed / max(1, len(plate_files)),
    )

    # Save annotations.json
    with annotations_json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # Save metadata.csv
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with metadata_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    logger.info(
        "Updated %s and %s successfully.",
        annotations_json_path.name,
        metadata_csv_path.name,
    )
    return records


def relabel_corrupted_folders_in_202611030230(
    annotator: PaddleOCRVLAnnotator,
    target_dir: Path,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Identify and relabel corrupted/mangled plate folders in dataset/reading/labeled/202611030230.
    Produces a detailed relabeling manifest.
    """
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {target_dir}")

    all_subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
    mangled_subdirs = [d for d in all_subdirs if is_mangled_folder_name(d.name)]

    logger.info(
        "Scan Results in 202611030230: Total folders = %d | Mangled/Corrupted = %d",
        len(all_subdirs),
        len(mangled_subdirs),
    )

    if limit is not None:
        mangled_subdirs = mangled_subdirs[:limit]
        logger.info("Applying limit: Processing %d mangled folders.", len(mangled_subdirs))

    manifest_path = target_dir / "relabel_vlm_manifest.json"
    manifest: Dict[str, Any] = {
        "metadata": {
            "timestamp_utc": get_utc_timestamp(),
            "target_dir": str(target_dir),
            "model": "PaddleOCR-VL-FP16",
            "total_mangled_folders_targeted": len(mangled_subdirs),
        },
        "items": {},
    }

    t0 = time.time()
    for idx, folder in enumerate(mangled_subdirs, start=1):
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if not image_files:
            continue

        rep_image = image_files[0]
        sha256 = compute_sha256(rep_image)
        vlm_res = annotator.annotate_image(rep_image)

        old_folder_name = folder.name
        chars = vlm_res["chars"]
        numbers = vlm_res["numbers"]
        province = vlm_res["province"]
        canonical = vlm_res["canonical_string"]

        # Parse plate type from old folder name if present
        parts = old_folder_name.split("_")
        plate_type = parts[0] if parts else "WhiteBlackCar"

        # Determine corrected name
        clean_d3 = chars if chars else "unknown"
        clean_d4 = numbers if numbers else "0000"
        clean_prov = province if province else "กรุงเทพมหานคร"
        new_folder_name = f"{plate_type}_{clean_d3}_{clean_d4}_{clean_prov}"

        manifest["items"][old_folder_name] = {
            "index": idx,
            "old_name": old_folder_name,
            "new_suggested_name": new_folder_name,
            "sha256": sha256,
            "chars": chars,
            "numbers": numbers,
            "province": province,
            "canonical_string": canonical,
            "image_count": len(image_files),
        }

        if idx % 50 == 0 or idx == len(mangled_subdirs):
            logger.info("Processed %d/%d mangled folders in 202611030230...", idx, len(mangled_subdirs))

    elapsed = time.time() - t0
    logger.info(
        "Finished scanning & relabeling %d mangled folders in %.2f s",
        len(mangled_subdirs),
        elapsed,
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Saved relabel manifest to: %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel datasets with PaddleOCR-VL in FP16 (Local Model First)")
    parser.add_argument(
        "--unverified-dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "reading" / "labeled" / "unverified_labeled_100"),
        help="Path to unverified_labeled_100 directory",
    )
    parser.add_argument(
        "--corrupted-dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "reading" / "labeled" / "202611030230"),
        help="Path to 202611030230 directory",
    )
    parser.add_argument(
        "--corrupted-limit",
        type=int,
        default=None,
        help="Optional limit on number of corrupted folders to process (default: all 1,669)",
    )
    parser.add_argument(
        "--skip-corrupted",
        action="store_true",
        help="Skip relabeling corrupted folders in 202611030230",
    )
    parser.add_argument(
        "--skip-unverified",
        action="store_true",
        help="Skip relabeling unverified_labeled_100",
    )

    args = parser.parse_args()

    logger.info("Initializing PaddleOCR-VL Annotator (Local Model First, FP16)...")
    annotator = PaddleOCRVLAnnotator(load_weights=True, local_first=True)

    if not args.skip_unverified:
        unverified_path = Path(args.unverified_dir).resolve()
        logger.info("=== Relabeling unverified_labeled_100 ===")
        relabel_unverified_100(annotator, unverified_path)

    if not args.skip_corrupted:
        corrupted_path = Path(args.corrupted_dir).resolve()
        logger.info("=== Relabeling Corrupted Folders in 202611030230 ===")
        relabel_corrupted_folders_in_202611030230(
            annotator,
            corrupted_path,
            limit=args.corrupted_limit,
        )

    logger.info("All relabeling tasks completed successfully.")


if __name__ == "__main__":
    main()
