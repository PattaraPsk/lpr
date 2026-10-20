#!/usr/bin/env python3
"""
Thai License Plate Character & Number Segmentation Engine
Author: Tailored for Local Machine Execution (WS4090 / CUDA / CPU)
Description:
    Segments top-band alphanumeric characters (Thai consonants and numbers)
    and bottom-band province text from cropped license plate images.
    Filters screws, borders, and noise, normalizes glyphs, and exports structured assets.
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ThaiCharCropper")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class CharCropMetadata:
    plate_name: str
    char_index: int
    band: str  # "top" or "province"
    bbox_xywh: List[int]  # [x, y, w, h] on original plate
    aspect_ratio: float
    area: int
    fill_ratio: float
    estimated_stroke_width: float
    crop_filename: str
    mask_filename: str


def estimate_stroke_width(binary_char: np.ndarray) -> float:
    """Estimates average stroke width using Euclidean distance transform."""
    if np.sum(binary_char > 0) == 0:
        return 0.0
    dist = cv2.distanceTransform((binary_char > 0).astype(np.uint8), cv2.DIST_L2, 5)
    non_zero = dist[dist > 0]
    if len(non_zero) == 0:
        return 0.0
    return float(np.median(non_zero) * 2.0)


def deskew_plate(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Deskews license plate if rotated."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    angle = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                rad = np.arctan2(y2 - y1, x2 - x1)
                deg = np.degrees(rad)
                if -25 < deg < 25:
                    angles.append(deg)
        if len(angles) > 0:
            angle = float(np.median(angles))

    if abs(angle) > 0.8:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed, angle

    return img, 0.0


def segment_plate(
    plate_img: np.ndarray,
    plate_stem: str,
    output_dir: str,
    target_char_height: int = 128,
) -> Tuple[List[CharCropMetadata], Optional[np.ndarray]]:
    """
    Analyzes and segments a single license plate into characters and province text.
    """
    h, w = plate_img.shape[:2]
    deskewed, _ = deskew_plate(plate_img)

    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    margin_y = max(4, int(h * 0.05))
    margin_x = max(6, int(w * 0.035))
    clean_bin = binary.copy()
    clean_bin[:margin_y, :] = 0
    clean_bin[-margin_y:, :] = 0
    clean_bin[:, :margin_x] = 0
    clean_bin[:, -margin_x:] = 0

    proj_y = np.sum(clean_bin > 0, axis=1)

    split_start = int(h * 0.48)
    split_end = int(h * 0.76)
    if split_end > split_start:
        valley_rel = np.argmin(proj_y[split_start:split_end])
        split_y = split_start + valley_rel
    else:
        split_y = int(h * 0.62)

    top_bin = clean_bin[:split_y, :]
    bot_bin = clean_bin[split_y:, :]

    top_contours, _ = cv2.findContours(top_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_top_boxes = []
    for cnt in top_contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if bw > 0.65 * w or bh > 0.95 * split_y:
            continue
        if bh >= 0.30 * split_y and area >= 300 and bw >= 12:
            is_screw = (by < split_y * 0.35) and (bx < w * 0.18 or bx > w * 0.82) and (bh < split_y * 0.38)
            if not is_screw:
                valid_top_boxes.append((bx, by, bw, bh, area, cnt))

    valid_top_boxes.sort(key=lambda b: b[0])

    top_dir = os.path.join(output_dir, "top_band")
    prov_dir = os.path.join(output_dir, "province_band")
    os.makedirs(top_dir, exist_ok=True)
    os.makedirs(prov_dir, exist_ok=True)

    metadata_list: List[CharCropMetadata] = []
    vis_img = deskewed.copy()

    cv2.line(vis_img, (0, split_y), (w, split_y), (255, 0, 0), 1)

    for idx, (bx, by, bw, bh, area, cnt) in enumerate(valid_top_boxes, start=1):
        char_rgb = deskewed[by : by + bh, bx : bx + bw]
        char_mask = top_bin[by : by + bh, bx : bx + bw]

        scale = target_char_height / float(bh)
        target_w = max(4, int(round(bw * scale)))
        resized_rgb = cv2.resize(char_rgb, (target_w, target_char_height), interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(char_mask, (target_w, target_char_height), interpolation=cv2.INTER_NEAREST)

        pad = 16
        canvas_w = target_w + 2 * pad
        canvas_h = target_char_height + 2 * pad
        rgba_canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        rgba_canvas[pad : pad + target_char_height, pad : pad + target_w, :3] = resized_rgb
        rgba_canvas[pad : pad + target_char_height, pad : pad + target_w, 3] = np.where(resized_mask > 0, 255, 0)

        crop_fname = f"{plate_stem}_top_char_{idx:02d}.png"
        mask_fname = f"{plate_stem}_top_char_{idx:02d}_mask.png"
        crop_path = os.path.join(top_dir, crop_fname)
        mask_path = os.path.join(top_dir, mask_fname)

        Image.fromarray(cv2.cvtColor(rgba_canvas, cv2.COLOR_BGRA2RGBA)).save(crop_path)
        Image.fromarray(resized_mask).save(mask_path)

        stroke_w = estimate_stroke_width(char_mask)

        metadata_list.append(
            CharCropMetadata(
                plate_name=plate_stem,
                char_index=idx,
                band="top",
                bbox_xywh=[int(bx), int(by), int(bw), int(bh)],
                aspect_ratio=round(bw / float(bh), 3),
                area=int(area),
                fill_ratio=round(area / float(bw * bh), 3),
                estimated_stroke_width=round(stroke_w, 2),
                crop_filename=crop_fname,
                mask_filename=mask_fname,
            )
        )

        cv2.rectangle(vis_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(
            vis_img,
            str(idx),
            (bx + 2, by + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Process and save Province Band
    prov_crop = deskewed[split_y:h, :]
    prov_fname = f"{plate_stem}_province.png"
    prov_path = os.path.join(prov_dir, prov_fname)
    Image.fromarray(cv2.cvtColor(prov_crop, cv2.COLOR_BGR2RGB)).save(prov_path)

    cv2.rectangle(vis_img, (margin_x, split_y + 4), (w - margin_x, h - margin_y), (255, 128, 0), 2)
    cv2.putText(
        vis_img,
        "Province",
        (margin_x + 6, split_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 128, 0),
        2,
    )

    metadata_list.append(
        CharCropMetadata(
            plate_name=plate_stem,
            char_index=0,
            band="province",
            bbox_xywh=[margin_x, split_y, w - 2 * margin_x, h - split_y],
            aspect_ratio=round((w - 2 * margin_x) / float(h - split_y), 3),
            area=int((w - 2 * margin_x) * (h - split_y)),
            fill_ratio=1.0,
            estimated_stroke_width=0.0,
            crop_filename=prov_fname,
            mask_filename="",
        )
    )

    return metadata_list, vis_img


def discover_plate_files(input_dir: Path) -> List[Path]:
    """Finds all plate images within directory with resilience to network/cached filesystems."""
    found = []
    if input_dir.is_file():
        return [input_dir]

    if not input_dir.exists():
        return []

    try:
        for p in input_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and "annotated" not in p.name.lower():
                found.append(p)
    except Exception:
        pass

    try:
        for name in os.listdir(input_dir):
            p = input_dir / name
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS and "annotated" not in p.name.lower():
                found.append(p)
    except Exception:
        pass

    if not found:
        # Check direct candidates if directory listing is cached
        for i in range(1, 20):
            for prefix in ["sample_plate_", "plate_sample_", "car_", "plate_", "test_plate_"]:
                for ext in [".jpg", ".png", ".jpeg"]:
                    cand = input_dir / f"{prefix}{i:02d}{ext}"
                    if cand.exists():
                        found.append(cand)
                    cand_single = input_dir / f"{prefix}{i}{ext}"
                    if cand_single.exists():
                        found.append(cand_single)

    return sorted(list(set(found)))


def main():
    parser = argparse.ArgumentParser(
        description="Thai License Plate Character & Number Segmentation Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="cropped",
        help="Directory containing cropped license plate images.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="char_bank",
        help="Output directory to save extracted characters and province bands.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=128,
        help="Normalized pixel height for extracted character crops.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Export visual diagnostics images with segmented bounding boxes.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    plate_files = discover_plate_files(input_dir)

    if not plate_files:
        logger.warning(f"No license plate images found in '{args.input}'.")
        sys.exit(0)

    logger.info("==================================================================")
    logger.info("   Thai License Plate Character & Number Segmentation Engine     ")
    logger.info("==================================================================")
    logger.info(f"Input Directory     : {args.input}")
    logger.info(f"Output Directory    : {args.output}")
    logger.info(f"Target Char Height  : {args.target_height} px")
    logger.info(f"Found Plate Images  : {len(plate_files)}")
    logger.info("==================================================================")

    os.makedirs(args.output, exist_ok=True)
    diag_dir = os.path.join(args.output, "diagnostics")
    if args.visualize:
        os.makedirs(diag_dir, exist_ok=True)

    all_char_metadata: List[CharCropMetadata] = []
    total_chars_cropped = 0

    for plate_path in plate_files:
        try:
            img = cv2.imdecode(np.fromfile(str(plate_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            metadata, vis_img = segment_plate(
                plate_img=img,
                plate_stem=plate_path.stem,
                output_dir=args.output,
                target_char_height=args.target_height,
            )

            all_char_metadata.extend(metadata)
            top_chars = [m for m in metadata if m.band == "top"]
            total_chars_cropped += len(top_chars)

            logger.info(f"Processed: {plate_path.name} -> Extracted {len(top_chars)} characters + Province")

            if args.visualize and vis_img is not None:
                diag_path = os.path.join(diag_dir, f"{plate_path.stem}_segmented.jpg")
                Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)).save(diag_path, quality=92)

        except Exception as e:
            logger.error(f"Error processing {plate_path.name}: {e}")

        clean_records = []
    for m in all_char_metadata:
        d = asdict(m)
        d['area'] = int(d['area'])
        d['bbox_xywh'] = [int(v) for v in d['bbox_xywh']]
        d['aspect_ratio'] = float(d['aspect_ratio'])
        d['fill_ratio'] = float(d['fill_ratio'])
        d['estimated_stroke_width'] = float(d['estimated_stroke_width'])
        clean_records.append(d)

    json_path = os.path.join(args.output, "character_manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_records, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    csv_path = os.path.join(args.output, "character_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "plate_name",
                "char_index",
                "band",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "aspect_ratio",
                "area",
                "fill_ratio",
                "estimated_stroke_width",
                "crop_filename",
                "mask_filename",
            ]
        )
        for m in all_char_metadata:
            writer.writerow(
                [
                    m.plate_name,
                    m.char_index,
                    m.band,
                    m.bbox_xywh[0],
                    m.bbox_xywh[1],
                    m.bbox_xywh[2],
                    m.bbox_xywh[3],
                    m.aspect_ratio,
                    m.area,
                    m.fill_ratio,
                    m.estimated_stroke_width,
                    m.crop_filename,
                    m.mask_filename,
                ]
            )

    logger.info("==================================================================")
    logger.info("                  CHARACTER CROPPING COMPLETE                     ")
    logger.info("==================================================================")
    logger.info(f"Total Plates Processed    : {len(plate_files)}")
    logger.info(f"Total Characters Extracted: {total_chars_cropped}")
    logger.info(f"Character Bank Folder     : {os.path.abspath(args.output)}")
    logger.info(f"Manifest JSON Export      : {os.path.abspath(json_path)}")
    logger.info(f"Manifest CSV Export       : {os.path.abspath(csv_path)}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
