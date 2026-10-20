#!/usr/bin/env python3
"""
Thai License Plate Crop Organizer & Batch Folder Packager
Author: Tailored for Local Machine Execution (WS4090 / CUDA / CPU)
Description:
    Scans cropped license plate images, groups them into batches of exactly 100 images per folder,
    names destination folders strictly from 'AA' to 'ZZ' (AA, AB, ..., AZ, BA, ..., ZZ),
    and renames each image inside as:
        {3digit}_{4digit}_{province}_{folderID}{imgID}.jpg
    where folderID is the 2-letter folder name (e.g. AA, AB) and imgID runs from 00 to 99.
    Resolves plate text from annotations/manifest metadata or filename parsing with clean fallbacks.
"""

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CropOrganizer")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def generate_folder_name(index: int, case: str = "upper") -> str:
    """
    Generates strictly two-letter folder names from AA to ZZ:
        Index 0   -> AA
        Index 1   -> AB
        ...
        Index 25  -> AZ
        Index 26  -> BA
        ...
        Index 675 -> ZZ
    Total capacity: 26 x 26 = 676 folders x 100 images = 67,600 images.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if case != "lower" else "abcdefghijklmnopqrstuvwxyz"
    max_capacity = len(alphabet) * len(alphabet)
    if index < 0 or index >= max_capacity:
        raise ValueError(f"Batch folder index {index} exceeds ZZ capacity (max {max_capacity - 1} = 676 folders).")

    first = index // len(alphabet)
    second = index % len(alphabet)
    return alphabet[first] + alphabet[second]


@dataclass
class PlateRecord:
    source_path: str
    original_filename: str
    digit_3: str
    digit_4: str
    province: str
    folder_id: str
    img_id: str
    new_filename: str
    dest_path: str


class MetadataResolver:
    """
    Resolves 3-digit registration prefix, 4-digit number, and province name
    using metadata files (JSON/CSV) and robust Thai/alphanumeric regex parsing.
    """

    def __init__(
        self,
        metadata_path: Optional[str] = None,
        search_dirs: Optional[List[str]] = None,
        default_3digit: str = "1กข",
        default_4digit: str = "0000",
        default_province: str = "กรุงเทพมหานคร",
    ):
        self.default_3digit = default_3digit
        self.default_4digit = default_4digit
        self.default_province = default_province
        self.meta_dict: Dict[str, Tuple[str, str, str]] = {}

        candidates = []
        if metadata_path and os.path.exists(metadata_path):
            candidates.append(metadata_path)

        if search_dirs:
            for sdir in search_dirs:
                if not os.path.isdir(sdir):
                    continue
                for name in ["annotations.json", "crops_metadata.json", "character_manifest.json", "annotations.csv"]:
                    p = os.path.join(sdir, name)
                    if os.path.exists(p) and p not in candidates:
                        candidates.append(p)
                parent = os.path.dirname(os.path.abspath(sdir))
                for name in ["annotations.json", "crops_metadata.json", "character_manifest.json", "annotations.csv"]:
                    p = os.path.join(parent, name)
                    if os.path.exists(p) and p not in candidates:
                        candidates.append(p)

        for meta_file in candidates:
            self._load_meta(meta_file)

    def _load_meta(self, path: str):
        try:
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        fname = item.get("filename") or item.get("image_name") or item.get("crop_filename")
                        if not fname:
                            continue
                        stem = Path(fname).stem.lower()

                        reg = item.get("registration") or item.get("full_text") or ""
                        prov = item.get("province") or ""

                        c3, d4 = self._split_reg(reg)
                        if not prov:
                            prov = self.default_province

                        self.meta_dict[stem] = (c3, d4, prov)
                        self.meta_dict[fname.lower()] = (c3, d4, prov)

            elif path.endswith(".csv"):
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fname = row.get("filename") or row.get("image_name") or row.get("crop_filename")
                        if not fname:
                            continue
                        stem = Path(fname).stem.lower()
                        reg = row.get("registration") or row.get("full_text") or ""
                        prov = row.get("province") or ""
                        c3, d4 = self._split_reg(reg)
                        if not prov:
                            prov = self.default_province
                        self.meta_dict[stem] = (c3, d4, prov)
                        self.meta_dict[fname.lower()] = (c3, d4, prov)

            logger.info(f"Loaded {len(self.meta_dict)} plate records from metadata: {path}")
        except Exception as e:
            logger.warning(f"Could not load metadata from {path}: {e}")

    def _split_reg(self, reg_text: str) -> Tuple[str, str]:
        parts = reg_text.strip().split()
        if len(parts) >= 2:
            return parts[0], parts[1].zfill(4)
        elif len(parts) == 1:
            m = re.match(r"([0-9]?[\u0e01-\u0e2eA-Za-z]{1,3})[-_ ]*([0-9]{1,4})", parts[0])
            if m:
                return m.group(1), m.group(2).zfill(4)
            return parts[0], self.default_4digit
        return self.default_3digit, self.default_4digit

    def resolve(self, image_path: Path, running_idx: int) -> Tuple[str, str, str]:
        stem = image_path.stem
        fname_lower = image_path.name.lower()
        stem_lower = stem.lower()

        # 1. Check loaded metadata
        if stem_lower in self.meta_dict:
            return self.meta_dict[stem_lower]
        if fname_lower in self.meta_dict:
            return self.meta_dict[fname_lower]

        # 2. Parse Thai Plate Regex from filename: e.g. "1กข_8920_กรุงเทพมหานคร"
        m_thai = re.search(
            r"(?P<c3>[0-9]?[\u0e01-\u0e2e]{1,2})[-_ ]*(?P<d4>[0-9]{1,4})(?:[-_ ]+(?P<prov>[\u0e00-\u0e7f]+))?",
            stem,
        )
        if m_thai:
            c3 = m_thai.group("c3")
            d4 = m_thai.group("d4").zfill(4)
            prov = m_thai.group("prov") if m_thai.group("prov") else self.default_province
            return c3, d4, prov

        # 3. Parse Alphanumeric Regex from filename: e.g. "1AB_8920_BANGKOK"
        cleaned = re.sub(r"(sample|plate|synth|crop|car|img|top|annotated)[_ -]*", "", stem, flags=re.IGNORECASE)
        m_alpha = re.search(
            r"(?:^|[_-])(?P<c3>[0-9]?[A-Za-z]{1,3})[-_ ]*(?P<d4>[0-9]{1,4})(?:[-_ ]+(?P<prov>[A-Za-z]+))?",
            cleaned,
        )
        if m_alpha:
            c3 = m_alpha.group("c3")
            d4 = m_alpha.group("d4").zfill(4)
            prov = m_alpha.group("prov") if m_alpha.group("prov") else self.default_province
            return c3, d4, prov

        # 4. Fallback using running index
        fallback_3 = self.default_3digit
        fallback_4 = f"{(running_idx % 9999) + 1:04d}"
        return fallback_3, fallback_4, self.default_province


def discover_crop_images(input_dir: Path) -> List[Path]:
    """Finds all cropped plate images in the specified directory."""
    images = []
    if input_dir.is_file():
        if input_dir.suffix.lower() in VALID_EXTENSIONS:
            return [input_dir]
        return []

    if not input_dir.exists():
        return []

    for root, _, files in os.walk(input_dir):
        if "diagnostics" in root or "annotated" in root:
            continue
        for f in sorted(files):
            p = Path(root) / f
            if p.suffix.lower() in VALID_EXTENSIONS:
                images.append(p)

    if not images:
        for prefix in ["sample_plate_", "synth_plate_", "plate_", "car_plate_", "crop_"]:
            for i in range(1, 150):
                for ext in [".jpg", ".png", ".jpeg"]:
                    cand = input_dir / f"{prefix}{i:04d}{ext}"
                    if cand.exists() and cand not in images:
                        images.append(cand)
                    cand2 = input_dir / f"{prefix}{i:02d}{ext}"
                    if cand2.exists() and cand2 not in images:
                        images.append(cand2)

    return sorted(images)


def process_image(src_path: Path, dest_path: Path, action: str = "copy"):
    """Copies or moves an image, ensuring output is high-quality JPEG."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix.lower()

    if ext in {".jpg", ".jpeg"} and action == "copy":
        shutil.copy2(src_path, dest_path)
    elif ext in {".jpg", ".jpeg"} and action == "move":
        shutil.move(src_path, dest_path)
    else:
        try:
            with Image.open(src_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(dest_path, "JPEG", quality=95)
            if action == "move":
                src_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to convert {src_path} to JPEG: {e}")
            shutil.copy2(src_path, dest_path)


def main():
    parser = argparse.ArgumentParser(
        description="Group cropped license plates into 100-image folders named AA to ZZ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="cropped",
        help="Input folder containing cropped license plates (e.g., 'cropped' or 'crop').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="organized_crops",
        help="Destination directory where grouped folders (AA, AB, ..., ZZ) are created.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of images per destination folder.",
    )
    parser.add_argument(
        "--case",
        type=str,
        choices=["upper", "lower"],
        default="upper",
        help="Folder naming letter case ('upper' = AA..ZZ; 'lower' = aa..zz).",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="File action: 'copy' leaves original files untouched; 'move' renames/moves them.",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        default=None,
        help="Path to annotations.json, crops_metadata.json, or CSV for ground-truth text mapping.",
    )
    parser.add_argument(
        "--default-province",
        type=str,
        default="กรุงเทพมหานคร",
        help="Default province name when not detectable from filename or metadata.",
    )
    parser.add_argument(
        "--default-3digit",
        type=str,
        default="1กข",
        help="Default 3-digit registration prefix when not detectable.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        if args.input == "cropped" and Path("crop").exists():
            input_path = Path("crop")
        else:
            logger.error(f"Input directory '{args.input}' not found!")
            sys.exit(1)

    images = discover_crop_images(input_path)
    if not images:
        logger.warning(f"No valid image files found in '{input_path}'.")
        sys.exit(0)

    total_images = len(images)
    total_folders = (total_images + args.batch_size - 1) // args.batch_size

    logger.info("==================================================================")
    logger.info("      Thai License Plate Crop Organizer & Batch Packager          ")
    logger.info("==================================================================")
    logger.info(f"Source Folder       : {input_path.resolve()}")
    logger.info(f"Target Output Folder: {Path(args.output).resolve()}")
    logger.info(f"Total Images Found  : {total_images}")
    logger.info(f"Batch Size          : {args.batch_size} images per folder")
    logger.info(f"Folders Required    : {total_folders} folder(s)")
    logger.info(f"Folder Sequence     : {generate_folder_name(0, args.case)} to {generate_folder_name(total_folders - 1, args.case)} (strictly AA to ZZ)")
    logger.info(f"Naming Template     : {{3digit}}_{{4digit}}_{{province}}_{{folderID}}{{imgID}}.jpg")
    logger.info(f"Action Mode         : {args.action.upper()}")
    logger.info("==================================================================")

    resolver = MetadataResolver(
        metadata_path=args.metadata,
        search_dirs=[str(input_path), "augmented_plates", "char_bank"],
        default_3digit=args.default_3digit,
        default_province=args.default_province,
    )

    records: List[PlateRecord] = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(images):
        folder_idx = idx // args.batch_size
        img_idx = idx % args.batch_size  # 00 to 99

        folder_id = generate_folder_name(folder_idx, args.case)
        img_id_str = f"{img_idx:02d}"  # 00 to 99

        c3, d4, prov = resolver.resolve(img_path, idx)

        # Sanitize parts for valid filename
        c3_clean = re.sub(r'[\/\\:*?"<>| ]+', '', c3) or args.default_3digit
        d4_clean = re.sub(r'[\/\\:*?"<>| ]+', '', d4) or "0000"
        prov_clean = re.sub(r'[\/\\:*?"<>| ]+', '', prov) or args.default_province

        # Construct new filename: 3digit_4digit_province_folderIDimgID.jpg
        # Example: 1กข_8920_กรุงเทพมหานคร_AA00.jpg
        new_filename = f"{folder_id}{img_id_str}_{c3_clean}_{d4_clean}_{prov_clean}.jpg"

        dest_folder = output_dir / folder_id
        dest_path = dest_folder / new_filename

        process_image(img_path, dest_path, action=args.action)

        record = PlateRecord(
            source_path=str(img_path.resolve()),
            original_filename=img_path.name,
            digit_3=c3_clean,
            digit_4=d4_clean,
            province=prov_clean,
            folder_id=folder_id,
            img_id=img_id_str,
            new_filename=new_filename,
            dest_path=str(dest_path.resolve()),
        )
        records.append(record)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            logger.info(f"Packaged [{idx + 1}/{total_images}] -> {folder_id}/{new_filename}")

    json_path = output_dir / "manifest.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "folder_id",
            "img_id",
            "new_filename",
            "3digit",
            "4digit",
            "province",
            "original_filename",
            "relative_path",
        ])
        for r in records:
            writer.writerow([
                r.folder_id,
                r.img_id,
                r.new_filename,
                r.digit_3,
                r.digit_4,
                r.province,
                r.original_filename,
                f"{r.folder_id}/{r.new_filename}",
            ])

    logger.info("==================================================================")
    logger.info("                ORGANIZATION PROCESS COMPLETE                     ")
    logger.info("==================================================================")
    logger.info(f"Total Images Organized : {len(records)}")
    logger.info(f"Total Folders Created  : {total_folders}")
    logger.info(f"Output Root Directory  : {output_dir.resolve()}")
    logger.info(f"Master Manifest JSON   : {json_path.resolve()}")
    logger.info(f"Master Manifest CSV    : {csv_path.resolve()}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
