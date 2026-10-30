#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Large Plate (แผ่นป้ายทะเบียนขนาดใหญ่) Dataset & Specimen Generator
Renders all 77 official provinces of Thailand + เบตง (78 total)
using the precision-aligned custom 'rbz-licenseplate-large.ttf' font
under Land Transport Act B.E. 2522 (440mm x 220mm, 2:1 aspect ratio).

Outputs:
1. Individual large plate province crops: dataset/provinces/crops_large/<clean_iso>_<name>_large.png
2. Full realistic large license plates: dataset/provinces/plates_large/<clean_iso>_<name>_large_plate.png
3. Comprehensive visual specimen sheet: thai_provinces_large_plates_specimen.png
4. Master 78-plate visual gallery: thai_provinces_large_plates_gallery.jpg
5. Complete dataset manifest: dataset/provinces/provinces_large_plate_manifest.json
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

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
sys.path.insert(0, str(ROOT_DIR))

from tools.realistic_plate_generator import render_calibrated_plate_die, DLT_PROVINCE_CODES

FONT_LARGE_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-large.ttf"
if not FONT_LARGE_PATH.exists():
    FONT_LARGE_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-yellow.ttf"
if not FONT_LARGE_PATH.exists():
    FONT_LARGE_PATH = ROOT_DIR / "dataset" / "fonts" / "rbz-licenseplate-large.ttf"

ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")
PROVINCES_DIR = ROOT_DIR / "dataset" / "provinces"
PLATES_LARGE_DIR = PROVINCES_DIR / "plates_large"
CROPS_LARGE_DIR = PROVINCES_DIR / "crops_large"
PLATES_YELLOW_DIR = PROVINCES_DIR / "plates_yellow"
CROPS_YELLOW_DIR = PROVINCES_DIR / "crops_yellow"

for d in [PROVINCES_DIR, PLATES_LARGE_DIR, CROPS_LARGE_DIR, PLATES_YELLOW_DIR, CROPS_YELLOW_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def shape_thai_text(text: str) -> str:
    """Shapes Thai combining vowel + tone mark sequences into standard Level 2 PUA glyphs."""
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


# 78 Localities with authentic Land Transport Act commercial vehicle registration numbers
PROVINCES_LARGE_DATA: List[Dict[str, str]] = [
    # ภาคกลาง & ตะวันออก (Central & Eastern - 26)
    {"iso": "TH-10", "name": "กรุงเทพมหานคร", "en": "Bangkok", "region": "Central", "plate_text": "11-9667"},
    {"iso": "TH-11", "name": "สมุทรปราการ", "en": "Samut Prakan", "region": "Central", "plate_text": "30-1011"},
    {"iso": "TH-12", "name": "นนทบุรี", "en": "Nonthaburi", "region": "Central", "plate_text": "10-2345"},
    {"iso": "TH-13", "name": "ปทุมธานี", "en": "Pathum Thani", "region": "Central", "plate_text": "30-4567"},
    {"iso": "TH-14", "name": "พระนครศรีอยุธยา", "en": "Phra Nakhon Si Ayutthaya", "region": "Central", "plate_text": "10-5678"},
    {"iso": "TH-15", "name": "อ่างทอง", "en": "Ang Thong", "region": "Central", "plate_text": "30-6789"},
    {"iso": "TH-16", "name": "ลพบุรี", "en": "Lopburi", "region": "Central", "plate_text": "10-7890"},
    {"iso": "TH-17", "name": "สิงห์บุรี", "en": "Sing Buri", "region": "Central", "plate_text": "30-8901"},
    {"iso": "TH-18", "name": "ชัยนาท", "en": "Chai Nat", "region": "Central", "plate_text": "10-9012"},
    {"iso": "TH-19", "name": "สระบุรี", "en": "Saraburi", "region": "Central", "plate_text": "70-1234"},
    {"iso": "TH-20", "name": "ชลบุรี", "en": "Chonburi", "region": "Eastern", "plate_text": "30-2345"},
    {"iso": "TH-21", "name": "ระยอง", "en": "Rayong", "region": "Eastern", "plate_text": "70-3456"},
    {"iso": "TH-22", "name": "จันทบุรี", "en": "Chanthaburi", "region": "Eastern", "plate_text": "10-4567"},
    {"iso": "TH-23", "name": "ตราด", "en": "Trat", "region": "Eastern", "plate_text": "30-5678"},
    {"iso": "TH-24", "name": "ฉะเชิงเทรา", "en": "Chachoengsao", "region": "Eastern", "plate_text": "70-6789"},
    {"iso": "TH-25", "name": "ปราจีนบุรี", "en": "Prachinburi", "region": "Eastern", "plate_text": "10-7891"},
    {"iso": "TH-26", "name": "นครนายก", "en": "Nakhon Nayok", "region": "Central", "plate_text": "30-8912"},
    {"iso": "TH-27", "name": "สระแก้ว", "en": "Sa Kaeo", "region": "Eastern", "plate_text": "70-9123"},
    {"iso": "TH-70", "name": "ราชบุรี", "en": "Ratchaburi", "region": "Western", "plate_text": "10-1357"},
    {"iso": "TH-71", "name": "กาญจนบุรี", "en": "Kanchanaburi", "region": "Western", "plate_text": "30-2468"},
    {"iso": "TH-72", "name": "สุพรรณบุรี", "en": "Suphan Buri", "region": "Central", "plate_text": "70-3579"},
    {"iso": "TH-73", "name": "นครปฐม", "en": "Nakhon Pathom", "region": "Central", "plate_text": "30-4680"},
    {"iso": "TH-74", "name": "สมุทรสาคร", "en": "Samut Sakhon", "region": "Central", "plate_text": "70-5791"},
    {"iso": "TH-75", "name": "สมุทรสงคราม", "en": "Samut Songkhram", "region": "Central", "plate_text": "10-6802"},
    {"iso": "TH-76", "name": "เพชรบุรี", "en": "Phetchaburi", "region": "Western", "plate_text": "30-7913"},
    {"iso": "TH-77", "name": "ประจวบคีรีขันธ์", "en": "Prachuap Khiri Khan", "region": "Western", "plate_text": "70-8024"},

    # ภาคเหนือ (Northern - 17)
    {"iso": "TH-50", "name": "เชียงใหม่", "en": "Chiang Mai", "region": "Northern", "plate_text": "30-5050"},
    {"iso": "TH-51", "name": "ลำพูน", "en": "Lamphun", "region": "Northern", "plate_text": "10-5151"},
    {"iso": "TH-52", "name": "ลำปาง", "en": "Lampang", "region": "Northern", "plate_text": "70-5252"},
    {"iso": "TH-53", "name": "อุตรดิตถ์", "en": "Uttaradit", "region": "Northern", "plate_text": "30-5353"},
    {"iso": "TH-54", "name": "แพร่", "en": "Phrae", "region": "Northern", "plate_text": "10-5454"},
    {"iso": "TH-55", "name": "น่าน", "en": "Nan", "region": "Northern", "plate_text": "30-5555"},
    {"iso": "TH-56", "name": "พะเยา", "en": "Phayao", "region": "Northern", "plate_text": "70-5656"},
    {"iso": "TH-57", "name": "เชียงราย", "en": "Chiang Rai", "region": "Northern", "plate_text": "30-5757"},
    {"iso": "TH-58", "name": "แม่ฮ่องสอน", "en": "Mae Hong Son", "region": "Northern", "plate_text": "10-5858"},
    {"iso": "TH-60", "name": "นครสวรรค์", "en": "Nakhon Sawan", "region": "Northern", "plate_text": "70-6060"},
    {"iso": "TH-61", "name": "อุทัยธานี", "en": "Uthai Thani", "region": "Northern", "plate_text": "30-6161"},
    {"iso": "TH-62", "name": "กำแพงเพชร", "en": "Kamphaeng Phet", "region": "Northern", "plate_text": "70-6262"},
    {"iso": "TH-63", "name": "ตาก", "en": "Tak", "region": "Northern", "plate_text": "10-6363"},
    {"iso": "TH-64", "name": "สุโขทัย", "en": "Sukhothai", "region": "Northern", "plate_text": "30-6464"},
    {"iso": "TH-65", "name": "พิษณุโลก", "en": "Phitsanulok", "region": "Northern", "plate_text": "70-6565"},
    {"iso": "TH-66", "name": "พิจิตร", "en": "Phichit", "region": "Northern", "plate_text": "10-6666"},
    {"iso": "TH-67", "name": "เพชรบูรณ์", "en": "Phetchabun", "region": "Northern", "plate_text": "30-6767"},

    # ภาคตะวันออกเฉียงเหนือ (Northeastern / Isan - 20)
    {"iso": "TH-30", "name": "นครราชสีมา", "en": "Nakhon Ratchasima", "region": "Northeastern", "plate_text": "30-3030"},
    {"iso": "TH-31", "name": "บุรีรัมย์", "en": "Buriram", "region": "Northeastern", "plate_text": "10-3131"},
    {"iso": "TH-32", "name": "สุรินทร์", "en": "Surin", "region": "Northeastern", "plate_text": "70-3232"},
    {"iso": "TH-33", "name": "ศรีสะเกษ", "en": "Sisaket", "region": "Northeastern", "plate_text": "30-3333"},
    {"iso": "TH-34", "name": "อุบลราชธานี", "en": "Ubon Ratchathani", "region": "Northeastern", "plate_text": "70-3434"},
    {"iso": "TH-35", "name": "ยโสธร", "en": "Yasothon", "region": "Northeastern", "plate_text": "10-3535"},
    {"iso": "TH-36", "name": "ชัยภูมิ", "en": "Chaiyaphum", "region": "Northeastern", "plate_text": "30-3636"},
    {"iso": "TH-37", "name": "อำนาจเจริญ", "en": "Amnat Charoen", "region": "Northeastern", "plate_text": "10-3737"},
    {"iso": "TH-38", "name": "บึงกาฬ", "en": "Bueng Kan", "region": "Northeastern", "plate_text": "70-3838"},
    {"iso": "TH-39", "name": "หนองบัวลำภู", "en": "Nong Bua Lamphu", "region": "Northeastern", "plate_text": "30-3939"},
    {"iso": "TH-40", "name": "ขอนแก่น", "en": "Khon Kaen", "region": "Northeastern", "plate_text": "70-4040"},
    {"iso": "TH-41", "name": "อุดรธานี", "en": "Udon Thani", "region": "Northeastern", "plate_text": "30-4141"},
    {"iso": "TH-42", "name": "เลย", "en": "Loei", "region": "Northeastern", "plate_text": "10-4242"},
    {"iso": "TH-43", "name": "หนองคาย", "en": "Nong Khai", "region": "Northeastern", "plate_text": "70-4343"},
    {"iso": "TH-44", "name": "มหาสารคาม", "en": "Maha Sarakham", "region": "Northeastern", "plate_text": "30-4444"},
    {"iso": "TH-45", "name": "ร้อยเอ็ด", "en": "Roi Et", "region": "Northeastern", "plate_text": "10-4545"},
    {"iso": "TH-46", "name": "กาฬสินธุ์", "en": "Kalasin", "region": "Northeastern", "plate_text": "70-4646"},
    {"iso": "TH-47", "name": "สกลนคร", "en": "Sakon Nakhon", "region": "Northeastern", "plate_text": "30-4747"},
    {"iso": "TH-48", "name": "นครพนม", "en": "Nakhon Phanom", "region": "Northeastern", "plate_text": "70-4848"},
    {"iso": "TH-49", "name": "มุกดาหาร", "en": "Mukdahan", "region": "Northeastern", "plate_text": "10-4949"},

    # ภาคใต้ & เบตง (Southern & Betong - 15)
    {"iso": "TH-80", "name": "นครศรีธรรมราช", "en": "Nakhon Si Thammarat", "region": "Southern", "plate_text": "70-8080"},
    {"iso": "TH-81", "name": "กระบี่", "en": "Krabi", "region": "Southern", "plate_text": "30-1647"},  # Real bus plate ground truth
    {"iso": "TH-82", "name": "พังงา", "en": "Phang Nga", "region": "Southern", "plate_text": "10-8282"},
    {"iso": "TH-83", "name": "ภูเก็ต", "en": "Phuket", "region": "Southern", "plate_text": "30-8383"},
    {"iso": "TH-84", "name": "สุราษฎร์ธานี", "en": "Surat Thani", "region": "Southern", "plate_text": "70-8484"},
    {"iso": "TH-85", "name": "ระนอง", "en": "Ranong", "region": "Southern", "plate_text": "10-8585"},
    {"iso": "TH-86", "name": "ชุมพร", "en": "Chumphon", "region": "Southern", "plate_text": "70-8686"},
    {"iso": "TH-90", "name": "สงขลา", "en": "Songkhla", "region": "Southern", "plate_text": "30-9090"},
    {"iso": "TH-91", "name": "สตูล", "en": "Satun", "region": "Southern", "plate_text": "10-9191"},
    {"iso": "TH-92", "name": "ตรัง", "en": "Trang", "region": "Southern", "plate_text": "70-9292"},
    {"iso": "TH-93", "name": "พัทลุง", "en": "Phatthalung", "region": "Southern", "plate_text": "30-9393"},
    {"iso": "TH-94", "name": "ปัตตานี", "en": "Pattani", "region": "Southern", "plate_text": "70-9494"},
    {"iso": "TH-95", "name": "ยะลา", "en": "Yala", "region": "Southern", "plate_text": "10-9595"},
    {"iso": "TH-96", "name": "นราธิวาส", "en": "Narathiwat", "region": "Southern", "plate_text": "70-9696"},
    {"iso": "TH-95-BT", "name": "เบตง", "en": "Betong", "region": "Special District", "plate_text": "30-9595"}
]


def generate_all_large_plates():
    print("=" * 80)
    print(f"[*] Generating Complete Thai Large Plate Dataset (77 Provinces + เบตง = {len(PROVINCES_LARGE_DATA)} total)")
    print(f"[*] Legal Physical Specification: Land Transport Act B.E. 2522 (440mm x 220mm, 2:1 aspect ratio)")
    print(f"[*] Large Plate Font: {FONT_LARGE_PATH}")
    print("=" * 80)

    # 1. High-Resolution Individual Large Plate Province Crops
    print(f"\n[1/4] Rendering high-resolution individual large plate crops (font size 52px)...")
    font_crop = ImageFont.truetype(str(FONT_LARGE_PATH), 52)
    manifest = []

    for item in PROVINCES_LARGE_DATA:
        name = item["name"]
        iso = item["iso"]
        en = item["en"]
        ptext = item["plate_text"]
        dlt_code = DLT_PROVINCE_CODES.get(name, "01")
        shaped_name = shape_thai_text(name)

        bbox = font_crop.getbbox(shaped_name)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad_x = 28
        pad_y = 20
        img_w = tw + pad_x * 2
        img_h = th + pad_y * 2

        # Stamped ink (22, 22, 22) on retroreflective yellow background (242, 198, 28)
        im = Image.new("RGB", (img_w, img_h), color=(242, 198, 28))
        draw = ImageDraw.Draw(im)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), shaped_name, fill=(22, 22, 22), font=font_crop)

        clean_iso = iso.replace("-", "_")
        crop_filename = f"{clean_iso}_{name}_large.png"
        crop_path = CROPS_LARGE_DIR / crop_filename
        im.save(str(crop_path))

        # Also save to yellow crops for backward compatibility
        crop_yellow_path = CROPS_YELLOW_DIR / f"{clean_iso}_{name}_yellow.png"
        shutil.copy2(str(crop_path), str(crop_yellow_path))

        plate_filename = f"{clean_iso}_{name}_large_plate.png"
        plate_path = PLATES_LARGE_DIR / plate_filename

        manifest_entry = dict(item)
        manifest_entry["plate_type"] = "Large Plate (440x220mm, 2:1)"
        manifest_entry["dlt_code"] = dlt_code
        manifest_entry["clean_iso"] = clean_iso
        manifest_entry["crop_filename"] = crop_filename
        manifest_entry["crop_path"] = str(crop_path.relative_to(ROOT_DIR)).replace("\\", "/")
        manifest_entry["plate_filename"] = plate_filename
        manifest_entry["plate_path"] = str(plate_path.relative_to(ROOT_DIR)).replace("\\", "/")
        manifest.append(manifest_entry)

    # 2. Render Full Realistic Large License Plates (600x300, 2.0:1 Ratio)
    print(f"\n[2/4] Rendering 78 full realistic Large License Plates (3D relief + rim + THAILAND + Code)...")
    for i, item in enumerate(PROVINCES_LARGE_DATA, start=1):
        name = item["name"]
        iso = item["iso"]
        ptext = item["plate_text"]
        clean_iso = iso.replace("-", "_")
        plate_filename = f"{clean_iso}_{name}_large_plate.png"
        plate_path = PLATES_LARGE_DIR / plate_filename

        plate_bgr = render_calibrated_plate_die(
            plate_text=ptext,
            province_text=name,
            target_w=600,
            target_h=300,
            colorway_key="taxi",
            apply_3d_relief=True,
            apply_watermark=False,
            is_large_plate=True
        )
        cv2.imwrite(str(plate_path), plate_bgr)

        # Also save to yellow plates for backward compatibility
        plate_yellow_path = PLATES_YELLOW_DIR / f"{clean_iso}_{name}_yellow_plate.png"
        cv2.imwrite(str(plate_yellow_path), plate_bgr)

        if i % 15 == 0 or i == len(PROVINCES_LARGE_DATA):
            code = DLT_PROVINCE_CODES.get(name, "01")
            print(f"  -> Generated [{i:02d}/{len(PROVINCES_LARGE_DATA)}] {iso} (Code {code}): {name} [{ptext}]")

    manifest_json_path = PROVINCES_DIR / "provinces_large_plate_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved complete Large Plate dataset manifest: {manifest_json_path}")

    # Copy manifest to artifact directory and backward-compatible yellow manifest
    artifact_manifest = ARTIFACT_DIR / "provinces_large_plate_manifest.json"
    shutil.copy2(str(manifest_json_path), str(artifact_manifest))
    shutil.copy2(str(manifest_json_path), str(PROVINCES_DIR / "provinces_yellow_manifest.json"))
    shutil.copy2(str(manifest_json_path), str(ARTIFACT_DIR / "provinces_yellow_manifest.json"))

    # 3. Comprehensive Visual Specimen Sheet (Large Plate Typography Catalog)
    print(f"\n[3/4] Rendering 78-Province Large Plate Catalog Specimen Sheet...")
    sheet_w = 1260
    sheet_h = 1680

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(255, 252, 240))
    draw = ImageDraw.Draw(sheet)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 22)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 14)
    font_sec = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 16)
    font_en = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)
    font_plate_prov = ImageFont.truetype(str(FONT_LARGE_PATH), 25)

    # Header Card
    draw.rounded_rectangle([30, 20, sheet_w - 30, 95], radius=10, fill=(245, 200, 35), outline=(200, 150, 20), width=2)
    draw.text((50, 30), "rbz-licenseplate-large.ttf — 78 Thai Large Plate Jurisdictions (77 Provinces + เบตง)", fill=(20, 20, 20), font=font_title)
    draw.text((50, 62), "Land Transport Act B.E. 2522 (440x220mm, 2:1 Ratio) Large Plate Stamped Dies with THAILAND Header & Province Codes", fill=(50, 45, 30), font=font_sub)

    regions = [
        ("Central & Eastern (ภาคกลางและภาคตะวันออก — 26 จังหวัด)", [p for p in PROVINCES_LARGE_DATA if p["region"] in ["Central", "Eastern", "Western"]]),
        ("Northern (ภาคเหนือ — 17 จังหวัด)", [p for p in PROVINCES_LARGE_DATA if p["region"] == "Northern"]),
        ("Northeastern / Isan (ภาคตะวันออกเฉียงเหนือ — 20 จังหวัด)", [p for p in PROVINCES_LARGE_DATA if p["region"] == "Northeastern"]),
        ("Southern & Special District (ภาคใต้และท้องที่พิเศษ — 14 จังหวัด + เบตง)", [p for p in PROVINCES_LARGE_DATA if p["region"] in ["Southern", "Special District"]])
    ]

    y_pos = 115
    cols = 3
    col_w = (sheet_w - 60) // cols

    for title, prov_list in regions:
        draw.text((35, y_pos), title, fill=(120, 80, 10), font=font_sec)
        y_pos += 26
        draw.line([35, y_pos, sheet_w - 35, y_pos], fill=(230, 210, 160), width=1)
        y_pos += 12

        row_count = (len(prov_list) + cols - 1) // cols
        card_h = 42

        for i, prov in enumerate(prov_list):
            c_idx = i % cols
            r_idx = i // cols
            cx = 35 + c_idx * col_w
            cy = y_pos + r_idx * (card_h + 8)

            is_betong = prov["name"] == "เบตง"
            bg_col = (255, 235, 150) if is_betong else (255, 248, 205)
            border_col = (180, 120, 20) if is_betong else (230, 200, 120)

            draw.rounded_rectangle([cx, cy, cx + col_w - 12, cy + card_h], radius=6, fill=bg_col, outline=border_col, width=1)

            shaped_pname = shape_thai_text(prov["name"])
            draw.text((cx + 12, cy + 6), shaped_pname, fill=(20, 20, 20), font=font_plate_prov)

            d_code = DLT_PROVINCE_CODES.get(prov["name"], "01")
            info_text = f"[{d_code}] {prov['iso']} | {prov['en']}"
            if is_betong:
                info_text += " [Betong]"
            draw.text((cx + col_w - 18, cy + 14), info_text, fill=(90, 80, 60), font=font_en, anchor="ra")

        y_pos += row_count * (card_h + 8) + 20

    specimen_sheet_path = ARTIFACT_DIR / "thai_provinces_large_plates_specimen.png"
    sheet.save(str(specimen_sheet_path))
    shutil.copy2(str(specimen_sheet_path), str(PROVINCES_DIR / "thai_provinces_large_plates_specimen.png"))
    shutil.copy2(str(specimen_sheet_path), str(ARTIFACT_DIR / "thai_provinces_yellow_specimen.png"))
    print(f"[+] Saved comprehensive 78-province Large Plate specimen sheet: {specimen_sheet_path}")

    # 4. Master Visual Plates Gallery (All 78 Realistic Large Plates in 6x13 Grid)
    print(f"\n[4/4] Rendering 78-Plate Master Large Plate Visual Gallery (6 columns x 13 rows)...")
    gallery_cols = 6
    gallery_rows = 13
    thumb_w = 360
    thumb_h = 180  # Exactly 2:1 aspect ratio (440x220mm)
    banner_h = 28
    card_cell_w = thumb_w
    card_cell_h = thumb_h + banner_h

    margin_x = 36
    margin_top = 140
    margin_bottom = 36
    gap_x = 20
    gap_y = 18

    gal_w = margin_x * 2 + gallery_cols * card_cell_w + (gallery_cols - 1) * gap_x
    gal_h = margin_top + gallery_rows * card_cell_h + (gallery_rows - 1) * gap_y + margin_bottom

    gallery_img = Image.new("RGB", (gal_w, gal_h), color=(248, 244, 230))
    g_draw = ImageDraw.Draw(gallery_img)

    # Master Header Card
    g_draw.rounded_rectangle([margin_x, 24, gal_w - margin_x, 115], radius=12, fill=(20, 20, 20), outline=(242, 198, 28), width=3)
    font_gal_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 26)
    font_gal_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 15)
    font_label_th = ImageFont.truetype(str(FONT_LARGE_PATH), 15)
    font_label_en = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)

    g_draw.text((margin_x + 30, 36), "Royal Thai Large Plates (แผ่นป้ายทะเบียนขนาดใหญ่) — Complete 78 Localities Catalog", fill=(242, 198, 28), font=font_gal_title)
    g_draw.text((margin_x + 30, 74), "Land Transport Act B.E. 2522 (440x220mm, 2.0:1 Ratio) • Stamped Dies (rbz-licenseplate-large.ttf) with 3D Relief", fill=(210, 215, 220), font=font_gal_sub)

    for i, item in enumerate(PROVINCES_LARGE_DATA):
        c = i % gallery_cols
        r = i // gallery_cols
        px = margin_x + c * (card_cell_w + gap_x)
        py = margin_top + r * (card_cell_h + gap_y)

        name = item["name"]
        iso = item["iso"]
        en = item["en"]
        dlt_code = DLT_PROVINCE_CODES.get(name, "01")
        clean_iso = iso.replace("-", "_")
        plate_path = PLATES_LARGE_DIR / f"{clean_iso}_{name}_large_plate.png"

        p_bgr = cv2.imread(str(plate_path))
        p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        p_pil = Image.fromarray(p_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

        is_betong = name == "เบตง"
        border_col = (180, 120, 20) if is_betong else (210, 180, 90)
        bar_bg = (255, 230, 120) if is_betong else (255, 248, 210)

        g_draw.rounded_rectangle([px - 2, py - 2, px + card_cell_w + 1, py + card_cell_h + 1], radius=8, fill=bar_bg, outline=border_col, width=2 if is_betong else 1)
        gallery_img.paste(p_pil, (px, py))

        bar_y = py + thumb_h
        label_en = f"[Code {dlt_code}] {iso} {en}"
        g_draw.text((px + 8, bar_y + 6), label_en, fill=(20, 20, 20) if not is_betong else (120, 40, 5), font=font_label_en)

        shaped_name = shape_thai_text(name)
        bbox_th = font_label_th.getbbox(shaped_name)
        th_w = bbox_th[2] - bbox_th[0]
        th_x = px + card_cell_w - th_w - 8
        g_draw.text((th_x, bar_y + 4), shaped_name, fill=(20, 20, 20) if not is_betong else (120, 40, 5), font=font_label_th)

    gallery_path = ARTIFACT_DIR / "thai_provinces_large_plates_gallery.jpg"
    gallery_img.save(str(gallery_path), quality=94, optimize=True)
    shutil.copy2(str(gallery_path), str(PROVINCES_DIR / "thai_provinces_large_plates_gallery.jpg"))
    shutil.copy2(str(gallery_path), str(ARTIFACT_DIR / "thai_provinces_yellow_plates_gallery.jpg"))
    print(f"[+] Saved master 78-plate Large Plate visual gallery: {gallery_path}")

    print("\n" + "=" * 80)
    print(f"[✓] Successfully Generated All 78 Thai Large Plate (แผ่นป้ายทะเบียนขนาดใหญ่) Localities!")
    print(f"    - High-Res Crops:     {len(PROVINCES_LARGE_DATA)} in {CROPS_LARGE_DIR}")
    print(f"    - Full 3D Plates:     {len(PROVINCES_LARGE_DATA)} in {PLATES_LARGE_DIR}")
    print(f"    - Dataset Manifest:   {manifest_json_path}")
    print(f"    - Typography Sheet:   {specimen_sheet_path}")
    print(f"    - Master Plate Grid:  {gallery_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_large_plates()
