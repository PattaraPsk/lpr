#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Province Dataset & Specimen Generator
Renders all 77 official provinces of Thailand + เบตง (78 total)
using the precision-aligned custom 'rbz-licenseplate-province.ttf' and 'rbz-licenseplate-id.ttf' fonts.

Outputs:
1. Individual character/word crops: dataset/provinces/<clean_iso>_<name>.png
2. Full realistic license plates: dataset/provinces/plates/<clean_iso>_<name>_plate.png
3. Comprehensive visual specimen sheet: thai_provinces_specimen.png
4. Full realistic plates gallery: thai_provinces_plates_gallery.jpg
5. Complete dataset manifest: dataset/provinces/provinces_manifest.json
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

from tools.realistic_plate_generator import render_calibrated_plate_die

CUSTOM_FONT_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-province.ttf"
if not CUSTOM_FONT_PATH.exists():
    CUSTOM_FONT_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate-prov.ttf"
if not CUSTOM_FONT_PATH.exists():
    CUSTOM_FONT_PATH = ROOT_DIR / "models" / "fonts" / "rbz-licenseplate.ttf"
if not CUSTOM_FONT_PATH.exists():
    CUSTOM_FONT_PATH = ROOT_DIR / "dataset" / "fonts" / "rbz-licenseplate.ttf"

ARTIFACT_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")
PROVINCES_DIR = ROOT_DIR / "dataset" / "provinces"
PLATES_DIR = PROVINCES_DIR / "plates"
PROVINCES_DIR.mkdir(parents=True, exist_ok=True)
PLATES_DIR.mkdir(parents=True, exist_ok=True)


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


# 77 Provinces of Thailand + เบตง (Special DLT registered district in Yala)
# Grouped by geographic and administrative regions with authentic DLT registration marks
PROVINCES_DATA: List[Dict[str, str]] = [
    # ภาคกลาง & ตะวันออก (Central & Eastern - 26)
    {"iso": "TH-10", "name": "กรุงเทพมหานคร", "en": "Bangkok", "region": "Central", "reg_plate": "1กข 8888"},
    {"iso": "TH-11", "name": "สมุทรปราการ", "en": "Samut Prakan", "region": "Central", "reg_plate": "2ขค 1234"},
    {"iso": "TH-12", "name": "นนทบุรี", "en": "Nonthaburi", "region": "Central", "reg_plate": "3งจ 5678"},
    {"iso": "TH-13", "name": "ปทุมธานี", "en": "Pathum Thani", "region": "Central", "reg_plate": "4ฉช 9012"},
    {"iso": "TH-14", "name": "พระนครศรีอยุธยา", "en": "Phra Nakhon Si Ayutthaya", "region": "Central", "reg_plate": "5ซฌ 3456"},
    {"iso": "TH-15", "name": "อ่างทอง", "en": "Ang Thong", "region": "Central", "reg_plate": "6ญด 7890"},
    {"iso": "TH-16", "name": "ลพบุรี", "en": "Lopburi", "region": "Central", "reg_plate": "7ตถ 2345"},
    {"iso": "TH-17", "name": "สิงห์บุรี", "en": "Sing Buri", "region": "Central", "reg_plate": "8ทธ 6789"},
    {"iso": "TH-18", "name": "ชัยนาท", "en": "Chai Nat", "region": "Central", "reg_plate": "9นบ 1357"},
    {"iso": "TH-19", "name": "สระบุรี", "en": "Saraburi", "region": "Central", "reg_plate": "1ปผ 2468"},
    {"iso": "TH-20", "name": "ชลบุรี", "en": "Chonburi", "region": "Eastern", "reg_plate": "2พฟ 9876"},
    {"iso": "TH-21", "name": "ระยอง", "en": "Rayong", "region": "Eastern", "reg_plate": "3ภม 5432"},
    {"iso": "TH-22", "name": "จันทบุรี", "en": "Chanthaburi", "region": "Eastern", "reg_plate": "4ยร 1098"},
    {"iso": "TH-23", "name": "ตราด", "en": "Trat", "region": "Eastern", "reg_plate": "5ลว 6543"},
    {"iso": "TH-24", "name": "ฉะเชิงเทรา", "en": "Chachoengsao", "region": "Eastern", "reg_plate": "6ศษ 2109"},
    {"iso": "TH-25", "name": "ปราจีนบุรี", "en": "Prachinburi", "region": "Eastern", "reg_plate": "7สห 8765"},
    {"iso": "TH-26", "name": "นครนายก", "en": "Nakhon Nayok", "region": "Central", "reg_plate": "8ฬอ 4321"},
    {"iso": "TH-27", "name": "สระแก้ว", "en": "Sa Kaeo", "region": "Eastern", "reg_plate": "9ฮก 9876"},
    {"iso": "TH-70", "name": "ราชบุรี", "en": "Ratchaburi", "region": "Western", "reg_plate": "1กค 2345"},
    {"iso": "TH-71", "name": "กาญจนบุรี", "en": "Kanchanaburi", "region": "Western", "reg_plate": "2ขง 6789"},
    {"iso": "TH-72", "name": "สุพรรณบุรี", "en": "Suphan Buri", "region": "Central", "reg_plate": "3งช 1234"},
    {"iso": "TH-73", "name": "นครปฐม", "en": "Nakhon Pathom", "region": "Central", "reg_plate": "4จซ 5678"},
    {"iso": "TH-74", "name": "สมุทรสาคร", "en": "Samut Sakhon", "region": "Central", "reg_plate": "5ฉฌ 9012"},
    {"iso": "TH-75", "name": "สมุทรสงคราม", "en": "Samut Songkhram", "region": "Central", "reg_plate": "6ญด 3456"},
    {"iso": "TH-76", "name": "เพชรบุรี", "en": "Phetchaburi", "region": "Western", "reg_plate": "7ตถ 7890"},
    {"iso": "TH-77", "name": "ประจวบคีรีขันธ์", "en": "Prachuap Khiri Khan", "region": "Western", "reg_plate": "8ทธ 2345"},

    # ภาคเหนือ (Northern - 17)
    {"iso": "TH-50", "name": "เชียงใหม่", "en": "Chiang Mai", "region": "Northern", "reg_plate": "9กก 9999"},
    {"iso": "TH-51", "name": "ลำพูน", "en": "Lamphun", "region": "Northern", "reg_plate": "1ขข 1111"},
    {"iso": "TH-52", "name": "ลำปาง", "en": "Lampang", "region": "Northern", "reg_plate": "2คค 2222"},
    {"iso": "TH-53", "name": "อุตรดิตถ์", "en": "Uttaradit", "region": "Northern", "reg_plate": "3กจ 3333"},
    {"iso": "TH-54", "name": "แพร่", "en": "Phrae", "region": "Northern", "reg_plate": "4จจ 4444"},
    {"iso": "TH-55", "name": "น่าน", "en": "Nan", "region": "Northern", "reg_plate": "5ฉฉ 5555"},
    {"iso": "TH-56", "name": "พะเยา", "en": "Phayao", "region": "Northern", "reg_plate": "6ชช 6666"},
    {"iso": "TH-57", "name": "เชียงราย", "en": "Chiang Rai", "region": "Northern", "reg_plate": "7ซซ 7777"},
    {"iso": "TH-58", "name": "แม่ฮ่องสอน", "en": "Mae Hong Son", "region": "Northern", "reg_plate": "8ญญ 8888"},
    {"iso": "TH-60", "name": "นครสวรรค์", "en": "Nakhon Sawan", "region": "Northern", "reg_plate": "1ดด 1212"},
    {"iso": "TH-61", "name": "อุทัยธานี", "en": "Uthai Thani", "region": "Northern", "reg_plate": "2ตต 2323"},
    {"iso": "TH-62", "name": "กำแพงเพชร", "en": "Kamphaeng Phet", "region": "Northern", "reg_plate": "3ถถ 3434"},
    {"iso": "TH-63", "name": "ตาก", "en": "Tak", "region": "Northern", "reg_plate": "4ทท 4545"},
    {"iso": "TH-64", "name": "สุโขทัย", "en": "Sukhothai", "region": "Northern", "reg_plate": "5ธธ 5656"},
    {"iso": "TH-65", "name": "พิษณุโลก", "en": "Phitsanulok", "region": "Northern", "reg_plate": "6นน 6767"},
    {"iso": "TH-66", "name": "พิจิตร", "en": "Phichit", "region": "Northern", "reg_plate": "7บบ 7878"},
    {"iso": "TH-67", "name": "เพชรบูรณ์", "en": "Phetchabun", "region": "Northern", "reg_plate": "8ปป 8989"},

    # ภาคตะวันออกเฉียงเหนือ (Northeastern / Isan - 20)
    {"iso": "TH-30", "name": "นครราชสีมา", "en": "Nakhon Ratchasima", "region": "Northeastern", "reg_plate": "9ผผ 9090"},
    {"iso": "TH-31", "name": "บุรีรัมย์", "en": "Buriram", "region": "Northeastern", "reg_plate": "1พพ 1313"},
    {"iso": "TH-32", "name": "สุรินทร์", "en": "Surin", "region": "Northeastern", "reg_plate": "2ฟฟ 2424"},
    {"iso": "TH-33", "name": "ศรีสะเกษ", "en": "Sisaket", "region": "Northeastern", "reg_plate": "3ภภ 3535"},
    {"iso": "TH-34", "name": "อุบลราชธานี", "en": "Ubon Ratchathani", "region": "Northeastern", "reg_plate": "4มม 4646"},
    {"iso": "TH-35", "name": "ยโสธร", "en": "Yasothon", "region": "Northeastern", "reg_plate": "5ยย 5757"},
    {"iso": "TH-36", "name": "ชัยภูมิ", "en": "Chaiyaphum", "region": "Northeastern", "reg_plate": "6รร 6868"},
    {"iso": "TH-37", "name": "อำนาจเจริญ", "en": "Amnat Charoen", "region": "Northeastern", "reg_plate": "7ลล 7979"},
    {"iso": "TH-38", "name": "บึงกาฬ", "en": "Bueng Kan", "region": "Northeastern", "reg_plate": "8วว 8080"},
    {"iso": "TH-39", "name": "หนองบัวลำภู", "en": "Nong Bua Lamphu", "region": "Northeastern", "reg_plate": "9ศศ 9191"},
    {"iso": "TH-40", "name": "ขอนแก่น", "en": "Khon Kaen", "region": "Northeastern", "reg_plate": "1ษษ 1414"},
    {"iso": "TH-41", "name": "อุดรธานี", "en": "Udon Thani", "region": "Northeastern", "reg_plate": "2สส 2525"},
    {"iso": "TH-42", "name": "เลย", "en": "Loei", "region": "Northeastern", "reg_plate": "3หห 3636"},
    {"iso": "TH-43", "name": "หนองคาย", "en": "Nong Khai", "region": "Northeastern", "reg_plate": "4ฬฬ 4747"},
    {"iso": "TH-44", "name": "มหาสารคาม", "en": "Maha Sarakham", "region": "Northeastern", "reg_plate": "5ออ 5858"},
    {"iso": "TH-45", "name": "ร้อยเอ็ด", "en": "Roi Et", "region": "Northeastern", "reg_plate": "6ฮฮ 6969"},
    {"iso": "TH-46", "name": "กาฬสินธุ์", "en": "Kalasin", "region": "Northeastern", "reg_plate": "7กข 7070"},
    {"iso": "TH-47", "name": "สกลนคร", "en": "Sakon Nakhon", "region": "Northeastern", "reg_plate": "8ขค 8181"},
    {"iso": "TH-48", "name": "นครพนม", "en": "Nakhon Phanom", "region": "Northeastern", "reg_plate": "9คง 9292"},
    {"iso": "TH-49", "name": "มุกดาหาร", "en": "Mukdahan", "region": "Northeastern", "reg_plate": "1กช 1515"},

    # ภาคใต้ & เบตง (Southern & Betong - 15)
    {"iso": "TH-80", "name": "นครศรีธรรมราช", "en": "Nakhon Si Thammarat", "region": "Southern", "reg_plate": "2จฉ 2626"},
    {"iso": "TH-81", "name": "กระบี่", "en": "Krabi", "region": "Southern", "reg_plate": "3ฉช 3737"},
    {"iso": "TH-82", "name": "พังงา", "en": "Phang Nga", "region": "Southern", "reg_plate": "4ชซ 4848"},
    {"iso": "TH-83", "name": "ภูเก็ต", "en": "Phuket", "region": "Southern", "reg_plate": "5ซฌ 5959"},
    {"iso": "TH-84", "name": "สุราษฎร์ธานี", "en": "Surat Thani", "region": "Southern", "reg_plate": "6ฌญ 6060"},
    {"iso": "TH-85", "name": "ระนอง", "en": "Ranong", "region": "Southern", "reg_plate": "7ญด 7171"},
    {"iso": "TH-86", "name": "ชุมพร", "en": "Chumphon", "region": "Southern", "reg_plate": "8ดต 8282"},
    {"iso": "TH-90", "name": "สงขลา", "en": "Songkhla", "region": "Southern", "reg_plate": "9ตถ 9393"},
    {"iso": "TH-91", "name": "สตูล", "en": "Satun", "region": "Southern", "reg_plate": "1ถท 1616"},
    {"iso": "TH-92", "name": "ตรัง", "en": "Trang", "region": "Southern", "reg_plate": "2ทธ 2727"},
    {"iso": "TH-93", "name": "พัทลุง", "en": "Phatthalung", "region": "Southern", "reg_plate": "3ธน 3838"},
    {"iso": "TH-94", "name": "ปัตตานี", "en": "Pattani", "region": "Southern", "reg_plate": "4นบ 4949"},
    {"iso": "TH-95", "name": "ยะลา", "en": "Yala", "region": "Southern", "reg_plate": "5บป 5050"},
    {"iso": "TH-96", "name": "นราธิวาส", "en": "Narathiwat", "region": "Southern", "reg_plate": "6ปผ 6161"},
    {"iso": "TH-95-BT", "name": "เบตง", "en": "Betong", "region": "Special District", "reg_plate": "1กข 8888"}
]


def generate_all_provinces():
    print("=" * 80)
    print(f"[*] Generating Complete Thai Provinces Dataset (77 Provinces + เบตง = {len(PROVINCES_DATA)} total)")
    print(f"[*] Province Font: {CUSTOM_FONT_PATH}")
    print("=" * 80)

    # 1. High-Resolution Individual Province Crops
    print(f"\n[1/4] Rendering high-resolution individual crops (font size 52px)...")
    font_crop = ImageFont.truetype(str(CUSTOM_FONT_PATH), 52)
    manifest = []

    for item in PROVINCES_DATA:
        name = item["name"]
        iso = item["iso"]
        en = item["en"]
        reg_plate = item["reg_plate"]
        shaped_name = shape_thai_text(name)

        bbox = font_crop.getbbox(shaped_name)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Padding around text
        pad_x = 28
        pad_y = 20
        img_w = tw + pad_x * 2
        img_h = th + pad_y * 2

        im = Image.new("RGB", (img_w, img_h), color=(240, 242, 245))
        draw = ImageDraw.Draw(im)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), shaped_name, fill=(25, 25, 25), font=font_crop)

        clean_iso = iso.replace("-", "_")
        crop_filename = f"{clean_iso}_{name}.png"
        crop_path = PROVINCES_DIR / crop_filename
        im.save(str(crop_path))

        plate_filename = f"{clean_iso}_{name}_plate.png"
        plate_path = PLATES_DIR / plate_filename

        manifest_entry = dict(item)
        manifest_entry["clean_iso"] = clean_iso
        manifest_entry["crop_filename"] = crop_filename
        manifest_entry["crop_path"] = str(crop_path.relative_to(ROOT_DIR)).replace("\\", "/")
        manifest_entry["plate_filename"] = plate_filename
        manifest_entry["plate_path"] = str(plate_path.relative_to(ROOT_DIR)).replace("\\", "/")
        manifest.append(manifest_entry)

    # 2. Render Full Realistic License Plates for All 78 Localities
    print(f"\n[2/4] Rendering 78 full realistic license plates (3D relief + metal rim + watermark)...")
    for i, item in enumerate(PROVINCES_DATA, start=1):
        name = item["name"]
        iso = item["iso"]
        reg_plate = item["reg_plate"]
        clean_iso = iso.replace("-", "_")
        plate_filename = f"{clean_iso}_{name}_plate.png"
        plate_path = PLATES_DIR / plate_filename

        plate_bgr = render_calibrated_plate_die(
            plate_text=reg_plate,
            province_text=name,
            target_w=680,
            target_h=300,
            colorway_key="private_car",
            apply_3d_relief=True,
            apply_watermark=True,
            is_commercial_transport=False
        )
        cv2.imwrite(str(plate_path), plate_bgr)
        if i % 15 == 0 or i == len(PROVINCES_DATA):
            print(f"  -> Generated [{i:02d}/{len(PROVINCES_DATA)}] {iso}: {name} ({reg_plate})")

    manifest_json_path = PROVINCES_DIR / "provinces_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved complete dataset manifest: {manifest_json_path}")

    # Copy manifest to artifact directory as well
    artifact_manifest = ARTIFACT_DIR / "provinces_manifest.json"
    shutil.copy2(str(manifest_json_path), str(artifact_manifest))

    # 3. Comprehensive Visual Specimen Sheet (Typography Catalog)
    print(f"\n[3/4] Rendering 78-Province Catalog Specimen Sheet (Typography & Alignment)...")
    sheet_w = 1260
    sheet_h = 1680

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(250, 252, 255))
    draw = ImageDraw.Draw(sheet)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 22)
    font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 14)
    font_sec = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 16)
    font_en = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)
    font_plate_prov = ImageFont.truetype(str(CUSTOM_FONT_PATH), 25)

    # Header Card
    draw.rounded_rectangle([30, 20, sheet_w - 30, 95], radius=10, fill=(240, 245, 250), outline=(210, 220, 235), width=2)
    draw.text((50, 30), "rbz-licenseplate-province.ttf — 78 Thai Registration Localities (77 Provinces + เบตง)", fill=(15, 23, 42), font=font_title)
    draw.text((50, 62), "Precision DLT Typography with Calibrated วรรณยุกต์ (Tone Marks & Vowels) Supporting All Thai Jurisdictions", fill=(71, 85, 105), font=font_sub)

    regions = [
        ("Central & Eastern (ภาคกลางและภาคตะวันออก — 26 จังหวัด)", [p for p in PROVINCES_DATA if p["region"] in ["Central", "Eastern", "Western"]]),
        ("Northern (ภาคเหนือ — 17 จังหวัด)", [p for p in PROVINCES_DATA if p["region"] == "Northern"]),
        ("Northeastern / Isan (ภาคตะวันออกเฉียงเหนือ — 20 จังหวัด)", [p for p in PROVINCES_DATA if p["region"] == "Northeastern"]),
        ("Southern & Special District (ภาคใต้และท้องที่พิเศษ — 14 จังหวัด + เบตง)", [p for p in PROVINCES_DATA if p["region"] in ["Southern", "Special District"]])
    ]

    y_pos = 115
    cols = 3
    col_w = (sheet_w - 60) // cols

    for title, prov_list in regions:
        # Section header
        draw.text((35, y_pos), title, fill=(30, 58, 138), font=font_sec)
        y_pos += 26
        draw.line([35, y_pos, sheet_w - 35, y_pos], fill=(226, 232, 240), width=1)
        y_pos += 12

        row_count = (len(prov_list) + cols - 1) // cols
        card_h = 42

        for i, prov in enumerate(prov_list):
            c_idx = i % cols
            r_idx = i // cols
            cx = 35 + c_idx * col_w
            cy = y_pos + r_idx * (card_h + 8)

            is_betong = prov["name"] == "เบตง"
            bg_col = (255, 248, 230) if is_betong else (255, 255, 255)
            border_col = (230, 180, 50) if is_betong else (226, 232, 240)

            draw.rounded_rectangle([cx, cy, cx + col_w - 12, cy + card_h], radius=6, fill=bg_col, outline=border_col, width=1)

            # Draw Province in rbz-licenseplate-province.ttf
            shaped_pname = shape_thai_text(prov["name"])
            draw.text((cx + 12, cy + 6), shaped_pname, fill=(15, 23, 42), font=font_plate_prov)

            # Draw ISO & English romanization
            info_text = f"{prov['iso']} | {prov['en']}"
            if is_betong:
                info_text += " [DLT Special District]"
            draw.text((cx + col_w - 18, cy + 14), info_text, fill=(100, 116, 139), font=font_en, anchor="ra")

        y_pos += row_count * (card_h + 8) + 20

    specimen_sheet_path = ARTIFACT_DIR / "thai_provinces_specimen.png"
    sheet.save(str(specimen_sheet_path))
    shutil.copy2(str(specimen_sheet_path), str(PROVINCES_DIR / "thai_provinces_specimen.png"))
    print(f"[+] Saved comprehensive 78-province typography specimen sheet: {specimen_sheet_path}")

    # 4. Comprehensive Master Visual Plates Gallery (All 78 Realistic Plates in Grid)
    print(f"\n[4/4] Rendering 78-Plate Master Visual Gallery (6 columns x 13 rows)...")
    gallery_cols = 6
    gallery_rows = 13  # 6 * 13 = 78
    thumb_w = 360
    thumb_h = int(round(thumb_w * (300.0 / 680.0)))  # ~159px
    banner_h = 28
    card_cell_w = thumb_w
    card_cell_h = thumb_h + banner_h

    margin_x = 36
    margin_top = 140
    margin_bottom = 36
    gap_x = 20
    gap_y = 18

    gal_w = margin_x * 2 + gallery_cols * card_cell_w + (gallery_cols - 1) * gap_x  # 2332 px
    gal_h = margin_top + gallery_rows * card_cell_h + (gallery_rows - 1) * gap_y + margin_bottom  # 2849 px

    gallery_img = Image.new("RGB", (gal_w, gal_h), color=(245, 247, 250))
    g_draw = ImageDraw.Draw(gallery_img)

    # Master Header Card
    g_draw.rounded_rectangle([margin_x, 24, gal_w - margin_x, 115], radius=12, fill=(15, 23, 42), outline=(30, 41, 59), width=2)
    font_gal_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 26)
    font_gal_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 15)
    font_label_th = ImageFont.truetype(str(CUSTOM_FONT_PATH), 15)
    font_label_en = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)

    g_draw.text((margin_x + 30, 36), "Royal Thai License Plates — Complete 77 Provinces + เบตง Catalog (78 Localities)", fill=(248, 250, 252), font=font_gal_title)
    g_draw.text((margin_x + 30, 74), "Physical Stamped Edge Dies (rbz-licenseplate-id.ttf & rbz-licenseplate-province.ttf) with 3D Emboss & DLT Security Wave", fill=(148, 163, 184), font=font_gal_sub)

    for i, item in enumerate(PROVINCES_DATA):
        c = i % gallery_cols
        r = i // gallery_cols
        px = margin_x + c * (card_cell_w + gap_x)
        py = margin_top + r * (card_cell_h + gap_y)

        name = item["name"]
        iso = item["iso"]
        en = item["en"]
        clean_iso = iso.replace("-", "_")
        plate_path = PLATES_DIR / f"{clean_iso}_{name}_plate.png"

        # Load plate BGR -> RGB and resize to thumbnail
        p_bgr = cv2.imread(str(plate_path))
        p_rgb = cv2.cvtColor(p_bgr, cv2.COLOR_BGR2RGB)
        p_pil = Image.fromarray(p_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)

        is_betong = name == "เบตง"
        border_col = (230, 180, 50) if is_betong else (203, 213, 225)
        bar_bg = (255, 248, 230) if is_betong else (255, 255, 255)

        # Card container outline
        g_draw.rounded_rectangle([px - 2, py - 2, px + card_cell_w + 1, py + card_cell_h + 1], radius=8, fill=bar_bg, outline=border_col, width=2 if is_betong else 1)

        # Paste plate thumbnail
        gallery_img.paste(p_pil, (px, py))

        # Bottom label bar
        bar_y = py + thumb_h
        # Draw ISO and English on left
        label_en = f"{iso} {en}"
        g_draw.text((px + 8, bar_y + 6), label_en, fill=(30, 41, 59) if not is_betong else (180, 83, 9), font=font_label_en)

        # Draw Thai province name on right
        shaped_name = shape_thai_text(name)
        bbox_th = font_label_th.getbbox(shaped_name)
        th_w = bbox_th[2] - bbox_th[0]
        th_x = px + card_cell_w - th_w - 8
        g_draw.text((th_x, bar_y + 4), shaped_name, fill=(15, 23, 42) if not is_betong else (180, 83, 9), font=font_label_th)

    gallery_path = ARTIFACT_DIR / "thai_provinces_plates_gallery.jpg"
    gallery_img.save(str(gallery_path), quality=94, optimize=True)
    shutil.copy2(str(gallery_path), str(PROVINCES_DIR / "thai_provinces_plates_gallery.jpg"))
    print(f"[+] Saved master 78-plate visual gallery: {gallery_path}")

    print("\n" + "=" * 80)
    print(f"[✓] Successfully Generated All 78 Thai Registration Localities!")
    print(f"    - High-Res Crops:     {len(PROVINCES_DATA)} in {PROVINCES_DIR}")
    print(f"    - Full 3D Plates:     {len(PROVINCES_DATA)} in {PLATES_DIR}")
    print(f"    - Dataset Manifest:   {manifest_json_path}")
    print(f"    - Typography Sheet:   {specimen_sheet_path}")
    print(f"    - Master Plate Grid:  {gallery_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_all_provinces()
