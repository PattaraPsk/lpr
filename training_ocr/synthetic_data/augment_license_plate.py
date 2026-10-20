#!/usr/bin/env python3
"""
Thai License Plate Synthetic Generator & Advanced Augmentation Engine
Author: Tailored for Local Machine Execution (WS4090 / CUDA / CPU)
Description:
    Generates realistic Thai license plates combining analyzed font typography
    and all 77 Thai provinces. Applies photorealistic camera perspective,
    specular illumination, motion blur, and weather/dirt sensor augmentations.
    Exports images and character-level ground truth annotations (JSON & CSV).
"""

import argparse
import csv
import importlib.util
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Resilient module loader for ThaiPlateFontEngine
try:
    from font_analyzer import ThaiPlateFontEngine
except (ImportError, ModuleNotFoundError):
    mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font_analyzer.py")
    if os.path.exists(mod_path):
        spec = importlib.util.spec_from_file_location("font_analyzer", mod_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ThaiPlateFontEngine = mod.ThaiPlateFontEngine
    else:
        raise ImportError(f"Cannot find font_analyzer.py at {mod_path}")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PlateAugmentor")

THAI_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี"
]

THAI_CONSONANTS = [
    "ก", "ข", "ค", "ฆ", "ง", "จ", "ฉ", "ช", "ฌ", "ญ",
    "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท",
    "ธ", "น", "บ", "ป", "ผ", "พ", "ฟ", "ภ", "ม", "ย",
    "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
]

PLATE_CATEGORIES = {
    "private_car": {"bg": (248, 248, 248), "text": (20, 20, 20), "border": (20, 20, 20)},
    "pickup": {"bg": (248, 248, 248), "text": (10, 115, 45), "border": (10, 115, 45)},
    "van": {"bg": (248, 248, 248), "text": (15, 55, 160), "border": (15, 55, 160)},
    "taxi": {"bg": (245, 205, 30), "text": (20, 20, 20), "border": (20, 20, 20)},
}


def generate_random_registration() -> str:
    """Generates valid Thai registration code according to DLT standards."""
    mode = random.choices(["modern", "classic"], weights=[0.85, 0.15])[0]
    c1 = random.choice(THAI_CONSONANTS)
    c2 = random.choice(THAI_CONSONANTS)
    digits = random.randint(1, 9999)

    if mode == "modern":
        prefix_digit = random.randint(1, 9)
        return f"{prefix_digit}{c1}{c2} {digits}"
    else:
        return f"{c1}{c2} {digits}"


def apply_perspective_transform(img: np.ndarray, bboxes: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """Applies realistic 3D perspective homography and transforms bounding boxes."""
    h, w = img.shape[:2]
    max_dx = w * 0.12
    max_dy = h * 0.18

    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [random.uniform(0, max_dx), random.uniform(0, max_dy)],
        [w - random.uniform(0, max_dx), random.uniform(0, max_dy)],
        [w - random.uniform(0, max_dx), h - random.uniform(0, max_dy)],
        [random.uniform(0, max_dx), h - random.uniform(0, max_dy)],
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))

    transformed_bboxes = []
    for b in bboxes:
        x1, y1, x2, y2 = b["bbox_xyxy"]
        corners = np.float32([
            [[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]
        ])
        t_corners = cv2.perspectiveTransform(corners, M)
        tx = t_corners[:, 0, 0]
        ty = t_corners[:, 0, 1]
        nx1 = max(0, min(w - 1, int(round(np.min(tx)))))
        ny1 = max(0, min(h - 1, int(round(np.min(ty)))))
        nx2 = max(0, min(w, int(round(np.max(tx)))))
        ny2 = max(0, min(h, int(round(np.max(ty)))))

        if (nx2 - nx1) > 3 and (ny2 - ny1) > 3:
            transformed_bboxes.append({
                "char": b["char"],
                "bbox_xyxy": [nx1, ny1, nx2, ny2],
                "band": b["band"],
            })

    return warped, transformed_bboxes


def apply_photometric_augmentations(img: np.ndarray) -> np.ndarray:
    """Applies lighting gradients, specular reflections, and sensor noise."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    # 1. Illumination Gradient
    angle = random.uniform(0, 2 * np.pi)
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    grad = np.cos(angle) * X + np.sin(angle) * Y
    intensity = random.uniform(0.15, 0.35)
    light_mask = 1.0 + grad * intensity
    out = out * light_mask[:, :, None]

    # 2. Specular Hotspot
    if random.random() < 0.6:
        cx = random.randint(int(w * 0.2), int(w * 0.8))
        cy = random.randint(int(h * 0.2), int(h * 0.8))
        sigma_x = random.randint(int(w * 0.15), int(w * 0.35))
        sigma_y = random.randint(int(h * 0.2), int(h * 0.5))

        gx = np.exp(-((X * (w / 2) - (cx - w / 2)) ** 2) / (2 * sigma_x ** 2))
        gy = np.exp(-((Y * (h / 2) - (cy - h / 2)) ** 2) / (2 * sigma_y ** 2))
        hotspot = (gx * gy) * random.uniform(30, 80)
        out += hotspot[:, :, None]

    # 3. Motion Blur or Gaussian Defocus
    if random.random() < 0.5:
        ksize = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (ksize, ksize), random.uniform(0.5, 1.2))

    # 4. Sensor Noise (ISO grain)
    noise = np.random.normal(0, random.uniform(2, 6), out.shape)
    out = np.clip(out + noise, 0, 255).astype(np.uint8)

    # 5. Dirt / Vignette
    if random.random() < 0.4:
        r_dist = np.sqrt(X ** 2 + Y ** 2)
        vignette = np.clip(1.0 - (r_dist ** 2) * 0.25, 0.7, 1.0)
        out = (out.astype(np.float32) * vignette[:, :, None]).astype(np.uint8)

    return out


def draw_screws_and_plate_border(draw: ImageDraw.ImageDraw, w: int, h: int, border_color: Tuple[int, int, int]):
    """Renders official DLT embossed border and mounting screws."""
    margin = 12
    draw.rounded_rectangle(
        [(margin, margin), (w - margin, h - margin)],
        radius=20,
        outline=border_color,
        width=8,
    )
    draw.rounded_rectangle(
        [(margin + 6, margin + 6), (w - margin - 6, h - margin - 6)],
        radius=16,
        outline=(210, 210, 210),
        width=2,
    )

    s_rad = 12
    draw.ellipse([(65, 45), (65 + 2 * s_rad, 45 + 2 * s_rad)], fill=(120, 120, 120), outline=(50, 50, 50), width=2)
    draw.ellipse([(w - 65 - 2 * s_rad, 45), (w - 65, 45 + 2 * s_rad)], fill=(120, 120, 120), outline=(50, 50, 50), width=2)


def generate_single_plate(
    reg_code: str,
    province: str,
    category_name: str,
    font_path: str,
    plate_w: int = 680,
    plate_h: int = 300,
    augment: bool = True,
) -> Tuple[np.ndarray, List[Dict]]:
    """Generates an augmented Thai license plate with ground-truth bboxes."""
    cat = PLATE_CATEGORIES.get(category_name, PLATE_CATEGORIES["private_car"])
    bg_color = cat["bg"]
    text_color = cat["text"]
    border_color = cat["border"]

    plate = Image.new("RGB", (plate_w, plate_h), bg_color)
    draw = ImageDraw.Draw(plate)
    draw_screws_and_plate_border(draw, plate_w, plate_h, border_color)

    engine = ThaiPlateFontEngine(font_path=font_path)
    text_layer, char_bboxes = engine.render_plate_text(
        plate_w=plate_w,
        plate_h=plate_h,
        text_top=reg_code,
        province=province,
        text_color=text_color,
    )
    plate.paste(text_layer, (0, 0), text_layer)

    plate_bgr = cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)

    if augment:
        plate_bgr, char_bboxes = apply_perspective_transform(plate_bgr, char_bboxes)
        plate_bgr = apply_photometric_augmentations(plate_bgr)

    return plate_bgr, char_bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Thai License Plate Synthetic Generator & Augmentation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=30,
        help="Number of augmented synthetic license plates to generate.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="augmented_plates",
        help="Directory to save augmented plates and annotations.",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=None,
        help="Path to TrueType font. Defaults to fonts/NotoSansThai-Bold.ttf.",
    )
    parser.add_argument(
        "--provinces",
        nargs="+",
        default=None,
        help="Subset of Thai provinces to sample from. Defaults to all 77 provinces.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable 3D perspective homography and sensor augmentations.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=680,
        help="Plate canvas width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Plate canvas height in pixels.",
    )
    args = parser.parse_args()

    font_path = args.font or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansThai-Bold.ttf")
    if not os.path.exists(font_path):
        fallback = "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"
        if os.path.exists(fallback):
            font_path = fallback

    province_pool = args.provinces or THAI_PROVINCES
    os.makedirs(args.output, exist_ok=True)

    logger.info("==================================================================")
    logger.info("  Thai License Plate Synthetic Generator & Augmentation Suite    ")
    logger.info("==================================================================")
    logger.info(f"Target Count        : {args.count} plates")
    logger.info(f"Output Directory    : {args.output}")
    logger.info(f"Province Pool Size  : {len(province_pool)} provinces")
    logger.info(f"Canvas Dimensions   : {args.width}x{args.height} px")
    logger.info(f"Augmentations       : {'Enabled (3D Perspective, Noise, Glare)' if args.augment else 'Disabled'}")
    logger.info("==================================================================")

    all_records = []
    category_keys = list(PLATE_CATEGORIES.keys())

    for idx in range(1, args.count + 1):
        reg_code = generate_random_registration()
        province = random.choice(province_pool)
        category = random.choices(category_keys, weights=[0.70, 0.15, 0.10, 0.05])[0]

        img_bgr, bboxes = generate_single_plate(
            reg_code=reg_code,
            province=province,
            category_name=category,
            font_path=font_path,
            plate_w=args.width,
            plate_h=args.height,
            augment=args.augment,
        )

        plate_id = f"synth_plate_{idx:04d}"
        img_name = f"{plate_id}.jpg"
        img_path = os.path.join(args.output, img_name)

        Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(img_path, quality=93)

        all_records.append({
            "plate_id": plate_id,
            "filename": img_name,
            "registration": reg_code,
            "province": province,
            "full_text": f"{reg_code} {province}",
            "category": category,
            "width": args.width,
            "height": args.height,
            "char_bboxes": bboxes,
        })

        if idx % 10 == 0 or idx == args.count:
            logger.info(f"Generated [{idx}/{args.count}]: '{reg_code}' [{province}] ({category})")

    json_path = os.path.join(args.output, "annotations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.output, "annotations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["plate_id", "filename", "registration", "province", "full_text", "category", "num_chars"])
        for r in all_records:
            writer.writerow([
                r["plate_id"],
                r["filename"],
                r["registration"],
                r["province"],
                r["full_text"],
                r["category"],
                len(r["char_bboxes"]),
            ])

    logger.info("==================================================================")
    logger.info("                SYNTHETIC AUGMENTATION COMPLETE                   ")
    logger.info("==================================================================")
    logger.info(f"Total Plates Generated  : {len(all_records)}")
    logger.info(f"Output Directory        : {os.path.abspath(args.output)}")
    logger.info(f"Annotations JSON Export : {os.path.abspath(json_path)}")
    logger.info(f"Annotations CSV Export  : {os.path.abspath(csv_path)}")
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
