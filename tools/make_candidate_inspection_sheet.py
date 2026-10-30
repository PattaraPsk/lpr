#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import cv2, numpy as np
from PIL import Image, ImageFont, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')
artifact_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b"

font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 22)
font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 13)
font_sec = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 16)
font_label = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 14)
font_body = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 12)
font_badge = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 11)

sections = [
    {
        "section_title": "1. Flat ณ (No Nen) — Ground Truth Discovery",
        "description": "Comparison of Flat Baseline plates: 8กณ 2026 (cur_BR98) & สณ 5 (cur_BS23) vs previous curved ษณ 3938",
        "items": [
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BR98_WhiteBlackCar_8กณ_2026_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "br98_char_2.jpg"),
                "plate_title": "8กณ 2026 (cur_BR98) — Ultra High-Res (1790x1074)",
                "char_title": "Character ณ (Flat Base, Aspect 0.617)",
                "metrics": "Tight Box: 221x358px | Aspect: 0.617 | Area: 50,401px",
                "verdict": "RECOMMENDED: Completely flat baseline, lean proportion matching ก",
                "badge_color": (22, 101, 52)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BS23_WhiteBlackCar_สณ_5_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "no_cand_BS23_สณ_5.png"),
                "plate_title": "สณ 5 (cur_BS23) — Daylight Camera (898x398)",
                "char_title": "Character ณ (Solid Flat Base, Aspect 0.766)",
                "metrics": "Tight Box: 111x145px | Aspect: 0.766 | Area: 10,769px",
                "verdict": "EXCELLENT: Solid flat horizontal baseline between legs",
                "badge_color": (30, 64, 175)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_024_sedan_ษณ_3938_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "inspect_crops", "crop_no_6.jpg"),
                "plate_title": "ษณ 3938 (plate_024) — Previous Reference",
                "char_title": "Character ณ (Curved Right Base, Aspect 0.907)",
                "metrics": "Tight Box: 332x366px | Aspect: 0.907",
                "verdict": "PREVIOUS: Base slightly rounded / slanted at right loop",
                "badge_color": (153, 27, 27)
            }
        ]
    },
    {
        "section_title": "2. Clean ฒ (Tho Phuthao) — True Die Width Ground Truth",
        "description": "True width calibrated to 0.624 aspect ratio (53x85px) — previous 0.884 was ~40% too wide",
        "items": [
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BR91_WhiteBlackCar_ฒข_9993_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "high_res_candidates", "cur_BR91_ฒข_9993_crop.jpg"),
                "plate_title": "ฒข 9993 (cur_BR91) — Razor Sharp (1125x409)",
                "char_title": "Character ฒ (Clean Die, Aspect 0.624)",
                "metrics": "Tight Box: 53x85px | Aspect: 0.624 | Matching ข (53x76px)",
                "verdict": "RECOMMENDED: Razor sharp unblurred contours, true lean die width",
                "badge_color": (22, 101, 52)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BR90_WhiteBlackCar_ฒฒ_999_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "high_res_candidates", "cur_BR90_ฒฒ_999_crop.jpg"),
                "plate_title": "ฒฒ 999 (cur_BR90) — Double ฒ (557x408)",
                "char_title": "Character ฒ (Dual Die Ground Truth, Aspect 0.624)",
                "metrics": "Tight Box: 53x85px | Aspect: 0.624 | Area: 2,592px",
                "verdict": "VERIFIED: Exactly confirms 0.624 aspect ratio across identical glyphs",
                "badge_color": (30, 64, 175)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_012_pickup_2ฒร_1387_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "inspect_crops", "crop_tho_6.jpg"),
                "plate_title": "2ฒร 1387 (plate_012) — Previous Reference",
                "char_title": "Character ฒ (Motion Blurred, Aspect 0.847)",
                "metrics": "Tight Box: 509x601px | Aspect: 0.847 (40% too wide)",
                "verdict": "PREVIOUS: Motion blur caused artificial dilation into bloated width",
                "badge_color": (153, 27, 27)
            }
        ]
    },
    {
        "section_title": "3. ฎ (Do Chada) — Slender Stamped Die Ground Truth",
        "description": "True ground truth aspect ratio is 0.636 (slender) — previous font was squished into 1.120 bloated block",
        "items": [
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BT08_WhiteBlackCar_ฎย_60_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "high_res_candidates", "tight", "tight_cur_BT08_ฎย_60_crop.jpg"),
                "plate_title": "ฎย 60 (cur_BT08) — High Resolution (1672x791)",
                "char_title": "Character ฎ (Slender DLT Die, Aspect 0.636)",
                "metrics": "Tight Box: 213x335px | Aspect: 0.636 | Area: 42,948px",
                "verdict": "RECOMMENDED: Authentic cap-height head, left loop, curved bottom tail",
                "badge_color": (22, 101, 52)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\dataset-auto-label-a1341\plates\cur_BS49_WhiteBlackCar_ฎม_246_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "high_res_candidates", "tight", "tight_cur_BS49_ฎม_246_crop.jpg"),
                "plate_title": "ฎม 246 (cur_BS49) — Sharp Daylight (1593x709)",
                "char_title": "Character ฎ (Compact Die, Aspect 0.647)",
                "metrics": "Tight Box: 187x289px | Aspect: 0.647 | Area: 30,897px",
                "verdict": "VERIFIED: Confirms 0.64 aspect ratio & horizontal tail loop",
                "badge_color": (30, 64, 175)
            },
            {
                "plate_path": r"d:\Projects-anti\lpr\dataset\labeled\labeled_gemini_flash\images\plate_019_sedan_ศฎ_2900_กรุงเทพมหานคร.jpg",
                "char_crop_path": os.path.join(artifact_dir, "current_font_do.png"),
                "plate_title": "ศฎ 2900 (plate_019) vs Previous Font ฎ",
                "char_title": "Previous Font ฎ (Distorted, Aspect 1.120)",
                "metrics": "Current Font: 150x168px | Aspect: 1.120 (Squished)",
                "verdict": "DEFECT: Tail compression caused glyph to balloon to 1.120 aspect",
                "badge_color": (153, 27, 27)
            }
        ]
    }
]

card_w = 410
card_h = 330
sheet_w = 1350
sheet_h = 130 + len(sections) * (card_h + 80)

sheet_pil = Image.new("RGB", (sheet_w, sheet_h), (250, 252, 255))
draw = ImageDraw.Draw(sheet_pil)

# Main Header
draw.rounded_rectangle([30, 20, sheet_w - 30, 95], radius=10, fill=(240, 245, 250), outline=(210, 220, 235), width=2)
draw.text((50, 30), "Thai License Plate Ground Truth Discovery: Flat ณ, Clean ฒ, and True ฎ", fill=(15, 23, 42), font=font_title)
draw.text((50, 62), "Direct Measurement of High-Resolution Camera Plates: Aspect Ratios, Contours, and Stamped Metal Geometry", fill=(71, 85, 105), font=font_sub)

y_sec = 120

for sec in sections:
    draw.text((40, y_sec), sec["section_title"], fill=(30, 58, 138), font=font_sec)
    draw.text((40, y_sec + 24), sec["description"], fill=(100, 116, 139), font=font_sub)
    
    y_cards = y_sec + 50
    for col_idx, it in enumerate(sec["items"]):
        cx = 40 + col_idx * (card_w + 25)
        cy = y_cards
        
        # Card outline
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=8, fill=(255, 255, 255), outline=(215, 225, 235), width=1)
        
        # Header Badge
        draw.rounded_rectangle([cx + 10, cy + 10, cx + card_w - 10, cy + 32], radius=4, fill=(245, 248, 252), outline=(220, 230, 242), width=1)
        draw.text((cx + 16, cy + 13), it["plate_title"], fill=it["badge_color"], font=font_badge)
        
        # Full Plate Image Crop
        im_plate = cv2.imdecode(np.fromfile(it["plate_path"], dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.exists(it["plate_path"]) else None
        if im_plate is not None:
            pres = cv2.resize(im_plate, (card_w - 24, 60), interpolation=cv2.INTER_AREA)
            sheet_pil.paste(Image.fromarray(cv2.cvtColor(pres, cv2.COLOR_BGR2RGB)), (cx + 12, cy + 40))
            
        # Character Crop
        draw.text((cx + 12, cy + 108), it["char_title"], fill=(30, 41, 59), font=font_label)
        draw.text((cx + 12, cy + 128), it["metrics"], fill=(20, 100, 40), font=font_body)
        
        im_char = cv2.imdecode(np.fromfile(it["char_crop_path"], dtype=np.uint8), cv2.IMREAD_COLOR) if os.path.exists(it["char_crop_path"]) else None
        if im_char is not None:
            kh, kw = im_char.shape[:2]
            sc = min(115 / float(kh), (card_w - 30) / float(kw))
            new_w = int(kw * sc)
            new_h = int(kh * sc)
            cres = cv2.resize(im_char, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            bx = cx + 12 + (card_w - 24 - new_w) // 2
            by = cy + 150 + (115 - new_h) // 2
            draw.rounded_rectangle([cx + 12, cy + 148, cx + card_w - 12, cy + 270], radius=4, fill=(248, 250, 252), outline=(225, 232, 242), width=1)
            sheet_pil.paste(Image.fromarray(cv2.cvtColor(cres, cv2.COLOR_BGR2RGB)), (bx, by))
            
        # Verdict Note
        draw.rounded_rectangle([cx + 12, cy + 278, cx + card_w - 12, cy + 318], radius=4, fill=(245, 247, 250), outline=(225, 230, 238), width=1)
        draw.text((cx + 18, cy + 284), it["verdict"][:55], fill=it["badge_color"], font=font_badge)
        if len(it["verdict"]) > 55:
            draw.text((cx + 18, cy + 300), it["verdict"][55:], fill=it["badge_color"], font=font_badge)
            
    y_sec += card_h + 80

out_final = os.path.join(artifact_dir, "candidates_real_plate_inspection.jpg")
sheet_pil.save(out_final)
print(f"[+] Successfully generated master inspection artifact: {out_final}")
