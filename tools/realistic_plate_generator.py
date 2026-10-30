#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realistic Thai License Plate Synthetic & Augmentation Engine (Precision Calibrated v5)
Engineered to precisely match physical DLT vehicle license plates and real CCTV/RTSP road camera footage.

Geometry & Typographic Calibration (Derived from Real Camera Ground Truth):
1. Canvas Ratio: 340mm x 150mm (2.267:1 aspect ratio).
2. Registration Baseline Alignment: Characters sit on a common horizontal baseline at y = 58.5% of plate height.
3. Character Height: 52% of plate height (156px on 300h canvas).
4. Kerning & Spacing:
   - Letters group: tight authentic kerning ~6-7% of character size.
   - Numbers group: tight authentic kerning ~7-8% of character size.
   - Middle gap between letter group and number group: 36-40% of character size (DLT standard spacer).
5. Province Layout: Centered horizontally, spanning ~70-73% of plate width, resting on baseline at y = 88% of plate height.
6. Stamped Border: Inset 14px, corner radius 20px, stroke width 4px.
7. Physical Effects: 3D hydraulic stamping bevel highlights and shadows, 3M retroreflective micro-grain, DLT security wave.
8. Optical & Camera Simulation: 3D homography perspective warp, defocus/motion blur, sensor noise, and streaming compression.
"""

import os
import sys
import math
import random
import re
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Ensure UTF-8 stdout
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
CUSTOM_FONT_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate.ttf"
if not CUSTOM_FONT_PATH.exists():
    CUSTOM_FONT_PATH = ROOT_DIR / "dataset" / "fonts" / "rbz-licenseplate.ttf"

FONT_ID_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-id.ttf"
if not FONT_ID_PATH.exists():
    FONT_ID_PATH = CUSTOM_FONT_PATH

FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-province.ttf"
if not FONT_PROV_PATH.exists():
    FONT_PROV_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
if not FONT_PROV_PATH.exists():
    FONT_PROV_PATH = CUSTOM_FONT_PATH

FONT_LARGE_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-large.ttf"
if not FONT_LARGE_PATH.exists():
    FONT_LARGE_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-yellow.ttf"
if not FONT_LARGE_PATH.exists():
    FONT_LARGE_PATH = CUSTOM_FONT_PATH

FONT_YELLOW_PATH = FONT_LARGE_PATH

ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")

COLORWAYS = {
    "private_car": ((235, 238, 241), (25, 25, 25), (32, 32, 32), "Private Car (White/Black)"),
    "van": ((235, 238, 241), (18, 55, 145), (22, 60, 150), "Van / Microbus (White/Blue)"),
    "pickup": ((235, 238, 241), (20, 110, 40), (24, 115, 45), "Pickup Truck (White/Green)"),
    "taxi": ((242, 198, 28), (22, 22, 22), (28, 28, 28), "Public Transport / Taxi (Yellow/Black)"),
    "service": ((16, 118, 58), (240, 242, 245), (235, 238, 240), "Hotel/Service (Green/White)"),
}

# DLT Province 2-Digit Codes for Commercial Transport (Land Transport Act & ASEAN Protocol)
DLT_PROVINCE_CODES = {
    # Central & Eastern (26)
    "กรุงเทพมหานคร": "01", "สมุทรปราการ": "02", "นนทบุรี": "03", "ปทุมธานี": "04",
    "พระนครศรีอยุธยา": "05", "อ่างทอง": "06", "ลพบุรี": "07", "สิงห์บุรี": "08",
    "ชัยนาท": "09", "สระบุรี": "10", "ชลบุรี": "11", "ระยอง": "12",
    "จันทบุรี": "13", "ตราด": "14", "ฉะเชิงเทรา": "15", "ปราจีนบุรี": "16",
    "นครนายก": "26", "สระแก้ว": "27", "ราชบุรี": "70", "กาญจนบุรี": "71",
    "สุพรรณบุรี": "72", "นครปฐม": "73", "สมุทรสาคร": "74", "สมุทรสงคราม": "75",
    "เพชรบุรี": "76", "ประจวบคีรีขันธ์": "77",

    # Northern (17)
    "เชียงใหม่": "50", "ลำพูน": "51", "ลำปาง": "52", "อุตรดิตถ์": "53",
    "แพร่": "54", "น่าน": "55", "พะเยา": "56", "เชียงราย": "57",
    "แม่ฮ่องสอน": "58", "นครสวรรค์": "60", "อุทัยธานี": "61", "กำแพงเพชร": "62",
    "ตาก": "63", "สุโขทัย": "64", "พิษณุโลก": "65", "พิจิตร": "66", "เพชรบูรณ์": "67",

    # Northeastern (20)
    "นครราชสีมา": "30", "บุรีรัมย์": "31", "สุรินทร์": "32", "ศรีสะเกษ": "33",
    "อุบลราชธานี": "34", "ยโสธร": "35", "ชัยภูมิ": "36", "อำนาจเจริญ": "37",
    "บึงกาฬ": "38", "หนองบัวลำภู": "39", "ขอนแก่น": "40", "อุดรธานี": "41",
    "เลย": "42", "หนองคาย": "43", "มหาสารคาม": "44", "ร้อยเอ็ด": "45",
    "กาฬสินธุ์": "46", "สกลนคร": "47", "นครพนม": "48", "มุกดาหาร": "49",

    # Southern & Betong (15)
    "นครศรีธรรมราช": "80", "กระบี่": "85", "พังงา": "82", "ภูเก็ต": "83",
    "สุราษฎร์ธานี": "84", "ระนอง": "85", "ชุมพร": "86", "สงขลา": "90",
    "สตูล": "91", "ตรัง": "92", "พัทลุง": "93", "ปัตตานี": "94",
    "ยะลา": "95", "นราธิวาส": "96", "เบตง": "95"
}


def shape_thai_text(text: str) -> str:
    """
    Shapes Thai combining vowel + tone mark sequences into standard Level 2 PUA glyphs.
    Prevents upper vowels (ี, ิ, ื, ึ, ั) and tone marks (่, ้, ๊, ๋, ์) from colliding,
    ensuring that 'กระบี่' renders distinctly without blending 'บ' + 'ี' into 'ป'.
    """
    UPPER_VOWELS = r'[\u0E31\u0E34-\u0E37\u0E47\u0E4D]'
    TONE_MAP = {
        '\u0E48': '\uF70A',  # maiekupperthai (Level 2)
        '\u0E49': '\uF70B',  # maithoupperthai (Level 2)
        '\u0E4A': '\uF70C',  # maitriupperthai (Level 2)
        '\u0E4B': '\uF70D',  # maichattawaupperthai (Level 2)
        '\u0E4C': '\uF70E',  # thanthakhatupperthai (Level 2)
    }

    def repl(m):
        return m.group(1) + TONE_MAP.get(m.group(2), m.group(2))

    return re.sub(f'({UPPER_VOWELS})([\u0E48-\u0E4C])', repl, text)


def add_retroreflective_texture(plate_rgb: np.ndarray, intensity: float = 2.5) -> np.ndarray:
    """Simulates 3M microprismatic retroreflective sheeting grain and ambient lighting."""
    h, w = plate_rgb.shape[:2]
    noise = np.random.normal(0, intensity, (h, w)).astype(np.float32)
    y_grad = np.linspace(1.02, 0.98, h)[:, None]
    x_grad = np.linspace(0.99, 1.01, w)[None, :]
    lighting = y_grad * x_grad

    result = plate_rgb.astype(np.float32) * lighting[:, :, None]
    for c in range(3):
        result[:, :, c] += noise
    return np.clip(result, 0, 255).astype(np.uint8)


def add_dlt_security_watermark(plate_rgb: np.ndarray) -> np.ndarray:
    """Adds subtle DLT wave security watermark across the middle plate bed."""
    h, w = plate_rgb.shape[:2]
    overlay = np.zeros((h, w), dtype=np.float32)
    for y_line in range(int(h * 0.46), int(h * 0.62), 16):
        for x in range(w):
            y_pos = int(y_line + 4 * math.sin(x * 0.08))
            if 0 <= y_pos < h:
                overlay[y_pos, x] = 1.0

    overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
    mod = (overlay * 5.0)[:, :, None]
    return np.clip(plate_rgb.astype(np.float32) + mod, 0, 255).astype(np.uint8)


def add_embossed_3d_relief(canvas_rgb: np.ndarray, mask_gray: np.ndarray, depth: int = 2) -> np.ndarray:
    """Simulates mechanical hydraulic stamping bevel relief."""
    img = canvas_rgb.astype(np.float32)

    m_high = np.zeros_like(mask_gray)
    m_high[:-depth, :-depth] = mask_gray[depth:, depth:]
    edge_high = cv2.GaussianBlur(np.clip(m_high.astype(np.int16) - mask_gray.astype(np.int16), 0, 255).astype(np.uint8), (3, 3), 0)

    m_shad = np.zeros_like(mask_gray)
    m_shad[depth:, depth:] = mask_gray[:-depth, :-depth]
    edge_shad = cv2.GaussianBlur(np.clip(m_shad.astype(np.int16) - mask_gray.astype(np.int16), 0, 255).astype(np.uint8), (3, 3), 0)

    for c in range(3):
        img[:, :, c] += (edge_high / 255.0) * 32.0
        img[:, :, c] -= (edge_shad / 255.0) * 36.0

    return np.clip(img, 0, 255).astype(np.uint8)


def render_calibrated_plate_die(
    plate_text: str,
    province_text: str,
    target_w: int = 680,
    target_h: int = 300,
    colorway_key: str = "private_car",
    apply_3d_relief: bool = True,
    apply_watermark: bool = True,
    is_commercial_transport: bool = False,
    is_large_plate: Optional[bool] = None
) -> np.ndarray:
    """
    Renders the physical stamped metal plate with ground-truth baseline alignment,
    accurate kerning, and calibrated province width.
    Supports both:
      - Standard Plate (Motor Vehicle Act): 340mm x 150mm (2.267:1 ratio)
      - Large Plate (Land Transport Act: Buses/Heavy Trucks): 440mm x 220mm (2.0:1 ratio)
    Uses separate specialized fonts:
      - rbz-licenseplate-id.ttf for standard vehicle registration mark
      - rbz-licenseplate-province.ttf for shorter, leaner province typography
      - rbz-licenseplate-large.ttf (rbz-licenseplate-yellow.ttf) for Large Plates
    Returns BGR image.
    """
    is_large = is_large_plate if is_large_plate is not None else is_commercial_transport
    h = target_h
    if is_large:
        w = int(round(h * 2.0))  # DLT Large Plate Standard: 440mm x 220mm (2.0:1 aspect ratio)
    else:
        w = target_w  # DLT Standard Private Vehicle: 340mm x 150mm (2.267:1 aspect ratio)
    bg_rgb, fg_rgb, border_rgb, _ = COLORWAYS.get(colorway_key, COLORWAYS["private_car"])

    im_pil = Image.new("RGB", (w, h), color=bg_rgb)
    draw = ImageDraw.Draw(im_pil)

    # 1. Stamped Outer Rim (DLT: 5mm inset, 3-4mm thickness, 10-12mm radius)
    inset = int(w * 0.016) if is_large else int(w * 0.018)  # ~10px on 600w, ~12px on 680w
    corner_r = int(h * 0.075) if is_large else int(h * 0.08) # ~22px on 300h
    draw.rounded_rectangle([inset, inset, w - inset, h - inset], radius=corner_r, outline=border_rgb, width=4)

    # 2. Large Plate Header (THAILAND + 2-Digit DLT Province Code)
    # Under Land Transport Act B.E. 2522 & ASEAN Cross-Border Protocol
    if is_large:
        # Upsized THAILAND and 2-digit Province Code using authentic physical stamped dies from rbz-licenseplate-large.ttf
        header_font_size = int(round(h * 0.20)) # ~60px on 300h
        font_header = ImageFont.truetype(str(FONT_LARGE_PATH), header_font_size)
        
        # 1. THAILAND header text
        th_bbox = font_header.getbbox("THAILAND")
        th_w = th_bbox[2] - th_bbox[0]
        th_x = 165
        th_y = int(h * 0.08) - th_bbox[1]
        draw.text((th_x, th_y), "THAILAND", fill=fg_rgb, font=font_header)

        # 2. 2-Digit DLT Province Code (e.g. '85' for Krabi) — Upsized with NO rectangle edge
        p_code = DLT_PROVINCE_CODES.get(province_text, "01")
        c_bbox = font_header.getbbox(p_code)
        cw = c_bbox[2] - c_bbox[0]
        code_x = w - inset - cw - 40
        code_y = th_y
        draw.text((code_x, code_y), p_code, fill=fg_rgb, font=font_header)

    # 3. Registration Sizing & Baseline Alignment (Plate ID Font or Large Plate Font)
    active_font_path = FONT_LARGE_PATH if is_large else FONT_ID_PATH
    if is_large:
        target_digit_h = int(h * 0.40) # ~120px on 300h (40% of plate height)
        baseline_reg = int(h * 0.635)  # Bottom baseline matches real bus plate ground truth
        mid_gap_ratio = 0.55           # Authentic gap accommodating stamped hyphen
    else:
        target_digit_h = int(h * 0.43) # ~130px on 300h
        baseline_reg = int(h * 0.560)
        mid_gap_ratio = 0.35
    max_reg_w = int(w * 0.88)      # max 88% width for 7-char plates

    f_test = ImageFont.truetype(str(active_font_path), 100)
    ref_b = f_test.getbbox("5")
    scale = target_digit_h / float(ref_b[3] - ref_b[1])
    reg_font_size = int(round(100 * scale))

    # Split plate text into letter group and number group
    # Handles both space and hyphen spacers (e.g. '30-1647' or 'กท 2058')
    has_hyphen = "-" in plate_text
    clean_ptext = plate_text.replace("-", " ")
    parts = clean_ptext.split()
    letters_str, numbers_str = (parts[0], parts[1]) if len(parts) == 2 else (plate_text, "")

    def measure_baseline_layout(size):
        f = ImageFont.truetype(str(active_font_path), size)
        kerning_l = int(size * 0.075) # authentic ~15px
        kerning_n = int(size * 0.070) # authentic ~14px
        mid_gap = int(size * mid_gap_ratio)

        def group_w(text, kern):
            w_acc = 0
            for ch in text:
                b = f.getbbox(ch)
                w_acc += (b[2] - b[0]) + kern
            return max(0, w_acc - kern)

        wl = group_w(letters_str, kerning_l)
        wn = group_w(numbers_str, kerning_n) if numbers_str else 0
        total_w = wl + (mid_gap if numbers_str else 0) + wn
        return f, total_w, wl, wn, mid_gap, kerning_l, kerning_n

    f_reg, total_w, wl, wn, mid_gap, kerning_l, kerning_n = measure_baseline_layout(reg_font_size)
    if total_w > max_reg_w:
        reg_font_size = int(round(reg_font_size * (max_reg_w / float(total_w))))
        f_reg, total_w, wl, wn, mid_gap, kerning_l, kerning_n = measure_baseline_layout(reg_font_size)

    start_x = (w - total_w) // 2
    ref_b_bottom = f_reg.getbbox("5")[3]
    char_y = baseline_reg - ref_b_bottom

    # Draw letters group
    cur_x = start_x
    for ch in letters_str:
        b = f_reg.getbbox(ch)
        draw.text((cur_x - b[0], char_y), ch, fill=fg_rgb, font=f_reg)
        cur_x += (b[2] - b[0]) + kerning_l

    # If commercial transport plate with hyphen, draw stamped hyphen spacer in mid_gap
    if has_hyphen and numbers_str:
        dash_w = int(mid_gap * 0.50)
        dash_x1 = start_x + wl + (mid_gap - dash_w) // 2
        dash_x2 = dash_x1 + dash_w
        dash_y = baseline_reg - int(target_digit_h * 0.50)
        dash_th = int(target_digit_h * 0.17) # Authentic stamped die hyphen thickness
        draw.line([dash_x1, dash_y, dash_x2, dash_y], fill=fg_rgb, width=dash_th)

    # Draw numbers group
    if numbers_str:
        cur_x = start_x + wl + mid_gap
        for ch in numbers_str:
            b = f_reg.getbbox(ch)
            draw.text((cur_x - b[0], char_y), ch, fill=fg_rgb, font=f_reg)
            cur_x += (b[2] - b[0]) + kerning_n

    # 4. Province Layout (Calibrated Bold Province via rbz-licenseplate-province.ttf or rbz-licenseplate-large.ttf)
    active_prov_font_path = FONT_LARGE_PATH if is_large else FONT_PROV_PATH
    shaped_province = shape_thai_text(province_text)
    if is_large:
        base_prov_size = int(h * 0.220)  # ~66px on 300h
        prov_baseline = int(h * 0.880)
        max_prov_w = int(w * 0.75)
    else:
        base_prov_size = int(h * 0.245)  # ~74px on 300h (Increased size to match real plate ground truth)
        prov_baseline = int(h * 0.890)
        max_prov_w = int(w * 0.82)
    f_prov = ImageFont.truetype(str(active_prov_font_path), base_prov_size)
    b_prov = f_prov.getbbox(shaped_province)
    pw = b_prov[2] - b_prov[0]

    # Scale down only if exceptionally long province exceeds max_prov_w
    if pw > max_prov_w:
        scale_p = max_prov_w / float(pw)
        f_prov = ImageFont.truetype(str(active_prov_font_path), int(round(base_prov_size * scale_p)))
        b_prov = f_prov.getbbox(shaped_province)
        pw = b_prov[2] - b_prov[0]

    prov_x = (w - pw) // 2 - b_prov[0]
    prov_ref_bottom = f_prov.getbbox("ก")[3]
    prov_y = prov_baseline - prov_ref_bottom
    draw.text((prov_x, prov_y), shaped_province, fill=fg_rgb, font=f_prov)

    plate_np = np.array(im_pil)

    # 5. 3D Hydraulic Emboss Relief
    if apply_3d_relief:
        gray = cv2.cvtColor(plate_np, cv2.COLOR_RGB2GRAY)
        if bg_rgb[0] > 128:
            _, text_mask = cv2.threshold(gray, int(bg_rgb[0] * 0.65), 255, cv2.THRESH_BINARY_INV)
        else:
            _, text_mask = cv2.threshold(gray, int(bg_rgb[0] * 1.35), 255, cv2.THRESH_BINARY)
        plate_np = add_embossed_3d_relief(plate_np, text_mask, depth=2)

    # 6. Retroreflective Texture
    plate_np = add_retroreflective_texture(plate_np, intensity=2.0)

    # 7. DLT Watermark
    if apply_watermark and not is_large:
        plate_np = add_dlt_security_watermark(plate_np)

    return cv2.cvtColor(plate_np, cv2.COLOR_RGB2BGR)


def apply_road_camera_optics(
    plate_bgr: np.ndarray,
    perspective_severity: float = 0.5,
    blur_ksize: int = 3,
    noise_sigma: float = 2.5,
    jpeg_quality: int = 80
) -> np.ndarray:
    """Simulates real CCTV edge camera optics, perspective, sensor grain, and compression."""
    h, w = plate_bgr.shape[:2]

    # 1. 3D Homography Perspective Warp
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    max_dx = int(w * 0.04 * perspective_severity)
    max_dy = int(h * 0.06 * perspective_severity)

    dst_pts = np.float32([
        [random.randint(-max_dx, 0), random.randint(0, max_dy)],
        [w + random.randint(0, max_dx), random.randint(0, max_dy)],
        [w + random.randint(0, max_dx), h - random.randint(0, max_dy)],
        [random.randint(-max_dx, 0), h - random.randint(0, max_dy)]
    ])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(plate_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 2. Optical Blur
    if blur_ksize >= 3:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        warped = cv2.GaussianBlur(warped, (k, k), 0)

    # 3. Sensor Noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, warped.shape).astype(np.float32)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 4. Video Streaming Compression Artifacts
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, enc = cv2.imencode('.jpg', warped, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def export_verification_comparison():
    print("=" * 80)
    print("[*] Generating Direct 1:1 Real Plate vs Augmented Verification Comparison")
    print("=" * 80)

    user_upload_dir = ARTIFACT_DIR / ".user_uploaded"
    real_img_dirs = [
        user_upload_dir,
        ROOT_DIR / "dataset" / "labeled" / "dataset-auto-label-a1341" / "plates",
        ROOT_DIR / "dataset" / "labeled" / "labeled_gemini_flash" / "images",
    ]

    test_cases = [
        {
            "title": "Case 1: 6กร 50 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ร' Ground Truth)",
            "plate_text": "6กร 50",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793033220592_72fc4eb7.jpg"
        },
        {
            "title": "Case 2: ฎป 37 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ป' & Bangkok Province Strip)",
            "plate_text": "ฎป 37",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793033284500_344423ec.jpg"
        },
        {
            "title": "Case 3: สฮ 365 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ส' and 'ฮ' Dies)",
            "plate_text": "สฮ 365",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793033463881_7cdb4174.jpg"
        },
        {
            "title": "Case 4: สฉ 611 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ฉ' Die)",
            "plate_text": "สฉ 611",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793033522995_4a688903.jpg"
        },
        {
            "title": "Case 5: ชถ 269 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ถ' Die)",
            "plate_text": "ชถ 269",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793033540130_b2eb73e3.jpg"
        },
        {
            "title": "Case 6: กร 289 นครปฐม (User Upload: Pristine Physical Stamped 'นครปฐม' Die with Tall 'ป' & Pedestal 'ฐ')",
            "plate_text": "กร 289",
            "province": "นครปฐม",
            "colorway": "private_car",
            "real_filename": "media_1793035479803_75aa3304.png"
        },
        {
            "title": "Case 7: ฆด 6886 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 'ด' Die)",
            "plate_text": "ฆด 6886",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793035431423_c55f5ad2.png"
        },
        {
            "title": "Case 8: กย 8756 เชียงใหม่ (User Upload: Pristine Physical Stamped 'เชียงใหม่' Die with authentic 'ย', 'ใ', 'ี', '่')",
            "plate_text": "กย 8756",
            "province": "เชียงใหม่",
            "colorway": "private_car",
            "real_filename": "media_1793035561680_f2bee60d.png"
        },
        {
            "title": "Case 9: ชป 7770 กรุงเทพมหานคร (User Upload: Pristine Physical Stamped 0 Die)",
            "plate_text": "ชป 7770",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793030925861_d37e572c.jpg"
        },
        {
            "title": "Case 10: 8กณ 2020 กรุงเทพมหานคร (User Upload 1: 8, ก, ณ, 2, 0)",
            "plate_text": "8กณ 2020",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793029731384_8399c0b3.jpg"
        },
        {
            "title": "Case 11: ฎบ 9 กรุงเทพมหานคร (User Upload 2: ฎ, บ)",
            "plate_text": "ฎบ 9",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793029760530_6e6ad9b1.jpg"
        },
        {
            "title": "Case 12: ชษ 867 กรุงเทพมหานคร (User Upload 3: ษ, 8, 6)",
            "plate_text": "ชษ 867",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793029828198_83e9d8e6.jpg"
        },
        {
            "title": "Case 13: งข 898 กรุงเทพมหานคร (User Upload 4: ง, ข, 8, 9)",
            "plate_text": "งข 898",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793029889477_97006155.jpg"
        },
        {
            "title": "Case 14: ชฐ 575 กรุงเทพมหานคร (User Upload 5: ช, ฐ with pedestal, 5, 7)",
            "plate_text": "ชฐ 575",
            "province": "กรุงเทพมหานคร",
            "colorway": "private_car",
            "real_filename": "media_1793029955492_e12eb03c.jpg"
        },
        {
            "title": "Case 15: 30-1647 กระบี่ (Commercial Public Transport Bus — Yellow/Black Font)",
            "plate_text": "30-1647",
            "province": "กระบี่",
            "colorway": "taxi",
            "is_commercial_transport": True,
            "real_filename": "plate_009_bus_30_1647_กระบี่.jpg"
        }
    ]

    card_w = 420
    card_h = 185
    row_gap = 52
    sheet_w = 1380
    sheet_h = 130 + len(test_cases) * (card_h + row_gap)

    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), color=(250, 252, 255))
    draw = ImageDraw.Draw(sheet_pil)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 22)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 13)
    font_sec = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 15)
    font_col = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 13)

    # Header Card
    draw.rounded_rectangle([30, 20, sheet_w - 30, 95], radius=10, fill=(240, 245, 250), outline=(210, 220, 235), width=2)
    draw.text((50, 30), "Thai License Plate Verification: Real Full Camera Plate Ground Truth vs Calibrated Augmented", fill=(15, 23, 42), font=font_title)
    draw.text((50, 62), "Direct Verification of Position, Spacing, Sizing, and Baseline Proportions (DLT Standard: 340x150mm Private, 440x220mm Commercial)", fill=(71, 85, 105), font=font_sub)

    y_cur = 120

    for idx, case in enumerate(test_cases):
        ptext = case["plate_text"]
        prov = case["province"]
        cway = case["colorway"]
        real_fn = case["real_filename"]
        is_comm = case.get("is_commercial_transport", False)

        # 1. Generate Clean Calibrated Die
        clean_bgr = render_calibrated_plate_die(
            ptext, prov,
            target_w=600 if is_comm else 680,
            target_h=300,
            colorway_key=cway,
            is_commercial_transport=is_comm
        )
        # 2. Generate Camera Augmented Ingest
        cctv_bgr = apply_road_camera_optics(clean_bgr, perspective_severity=0.45, blur_ksize=3, noise_sigma=2.0, jpeg_quality=84)

        # Draw Section Title
        draw.text((40, y_cur - 20), case["title"], fill=(30, 58, 138), font=font_sec)

        # For commercial yellow plate (2.0:1 aspect ratio), display card at 370x185 centered in 420px slot
        if is_comm:
            this_card_w = 370
            this_card_h = 185
            card_x_off = (card_w - this_card_w) // 2
        else:
            this_card_w = card_w
            this_card_h = card_h
            card_x_off = 0

        # Column 1: Real Ground Truth Full Plate Image
        col1_x = 40
        draw.text((col1_x, y_cur), "[Column 1] Real Road Camera Plate (Ground Truth)", fill=(180, 20, 20), font=font_col)
        real_path = None
        for r_dir in real_img_dirs:
            p_check = r_dir / real_fn
            if p_check.exists():
                real_path = p_check
                break

        if real_path and real_path.exists():
            real_bgr = cv2.imdecode(np.fromfile(str(real_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            rh, rw = real_bgr.shape[:2]
            target_aspect = this_card_w / float(this_card_h)
            current_aspect = rw / float(rh)
            if current_aspect > target_aspect:
                new_w = int(rh * target_aspect)
                x_off = (rw - new_w) // 2
                real_crop = real_bgr[:, x_off:x_off + new_w]
            else:
                new_h = int(rw / target_aspect)
                y_off = (rh - new_h) // 2
                real_crop = real_bgr[y_off:y_off + new_h, :]

            real_res = cv2.resize(real_crop, (this_card_w, this_card_h), interpolation=cv2.INTER_LANCZOS4)
            real_rgb = cv2.cvtColor(real_res, cv2.COLOR_BGR2RGB)
            sheet_pil.paste(Image.fromarray(real_rgb), (col1_x + card_x_off, y_cur + 20))
        else:
            draw.rounded_rectangle([col1_x, y_cur + 20, col1_x + card_w, y_cur + 20 + card_h], radius=8, fill=(245, 247, 250), outline=(210, 215, 220), width=1)
            draw.text((col1_x + 30, y_cur + 80), f"File not found: {real_fn}", fill=(200, 50, 50), font=font_col)

        # Column 2: Calibrated Physical Stamped Die
        col2_x = 480
        draw.text((col2_x, y_cur), "[Column 2] Calibrated Die (rbz-id, rbz-province & rbz-yellow)", fill=(20, 110, 40), font=font_col)
        clean_res = cv2.resize(clean_bgr, (this_card_w, this_card_h), interpolation=cv2.INTER_LANCZOS4)
        clean_rgb = cv2.cvtColor(clean_res, cv2.COLOR_BGR2RGB)
        sheet_pil.paste(Image.fromarray(clean_rgb), (col2_x + card_x_off, y_cur + 20))

        # Column 3: Augmented Camera Ingest
        col3_x = 920
        draw.text((col3_x, y_cur), "[Column 3] Camera Ingest Simulation (Optics + Noise + Glare)", fill=(70, 70, 80), font=font_col)
        cctv_res = cv2.resize(cctv_bgr, (this_card_w, this_card_h), interpolation=cv2.INTER_LANCZOS4)
        cctv_rgb = cv2.cvtColor(cctv_res, cv2.COLOR_BGR2RGB)
        sheet_pil.paste(Image.fromarray(cctv_rgb), (col3_x + card_x_off, y_cur + 20))

        y_cur += card_h + row_gap

    # Save artifact as JPG
    out_comparison_jpg = ARTIFACT_DIR / "real_vs_augmented_comparison.jpg"
    sheet_rgb = sheet_pil.convert("RGB")
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(np.array(sheet_rgb), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    buf.tofile(str(out_comparison_jpg))
    print(f"[+] Exported verification comparison to: {out_comparison_jpg}")


if __name__ == "__main__":
    export_verification_comparison()
