#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import cv2, numpy as np
from PIL import Image, ImageFont, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')

inspect_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\inspect_candidates"
crops_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\inspect_crops"
artifact_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b"

font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 20)
font_label = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 14)
font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 12)

def make_gallery(title, subtitle, items, out_filename):
    card_w = 280
    card_h = 340
    cols = min(4, len(items))
    rows = (len(items) + cols - 1) // cols
    
    sheet_w = 50 + cols * (card_w + 20)
    sheet_h = 110 + rows * (card_h + 20)
    
    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), (250, 252, 255))
    draw = ImageDraw.Draw(sheet_pil)
    
    # Header
    draw.rounded_rectangle([25, 20, sheet_w - 25, 85], radius=10, fill=(240, 245, 250), outline=(210, 220, 235), width=2)
    draw.text((45, 28), title, fill=(15, 23, 42), font=font_title)
    draw.text((45, 58), subtitle, fill=(71, 85, 105), font=font_sub)
    
    for i, it in enumerate(items):
        col = i % cols
        row = i // cols
        cx = 35 + col * (card_w + 20)
        cy = 100 + row * (card_h + 20)
        
        # Load plate image & crop image
        p_plate = os.path.join(inspect_dir, it['fn_plate'])
        p_crop = os.path.join(crops_dir, it['fn_crop'])
        
        im_plate = cv2.imread(p_plate)
        im_crop = cv2.imread(p_crop)
        
        # Measure character tight bounding box in crop
        gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(th > 0)
        if len(ys) > 0:
            ch_h = ys.max() - ys.min() + 1
            ch_w = xs.max() - xs.min() + 1
            aspect = ch_w / float(ch_h)
            tight_crop = im_crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
        else:
            ch_h, ch_w = im_crop.shape[:2]
            aspect = ch_w / float(ch_h)
            tight_crop = im_crop
            
        # Draw card
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=8, fill=(255, 255, 255), outline=(215, 225, 235), width=1)
        draw.text((cx + 12, cy + 12), it['label'], fill=(30, 41, 59), font=font_label)
        draw.text((cx + 12, cy + 34), f"Tight Box: {ch_w}x{ch_h}px | Aspect: {aspect:.3f}", fill=(20, 110, 40), font=font_sub)
        draw.text((cx + 12, cy + 52), it.get('note', ''), fill=(100, 116, 139), font=font_sub)
        
        # Draw Plate Image (scaled to card_w - 24, h = 65)
        pres = cv2.resize(im_plate, (card_w - 24, 65), interpolation=cv2.INTER_AREA)
        sheet_pil.paste(Image.fromarray(cv2.cvtColor(pres, cv2.COLOR_BGR2RGB)), (cx + 12, cy + 74))
        
        # Draw Character Crop (scaled preserving aspect ratio to fit 160w x 160h)
        kh, kw = tight_crop.shape[:2]
        sc = min(150 / float(kh), (card_w - 24) / float(kw))
        cw = int(kw * sc)
        ch = int(kh * sc)
        cres = cv2.resize(tight_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
        
        bx = cx + 12 + (card_w - 24 - cw) // 2
        by = cy + 155 + (160 - ch) // 2
        draw.rounded_rectangle([cx + 12, cy + 150, cx + card_w - 12, cy + 320], radius=4, fill=(248, 250, 252), outline=(225, 232, 240), width=1)
        sheet_pil.paste(Image.fromarray(cv2.cvtColor(cres, cv2.COLOR_BGR2RGB)), (bx, by))
        
    out_p = os.path.join(artifact_dir, out_filename)
    sheet_pil.save(out_p)
    print(f"[+] Successfully exported {out_filename}")

no_items = [
    {"fn_plate": "no_0.jpg", "fn_crop": "crop_no_0.jpg", "label": "1. สณ 332 (cur_AK43)", "note": "Sharp daylight; very flat baseline on ณ"},
    {"fn_plate": "no_1.jpg", "fn_crop": "crop_no_1.jpg", "label": "2. วณ 315 (cur_AK00)", "note": "High contrast; flat horizontal base"},
    {"fn_plate": "no_2.jpg", "fn_crop": "crop_no_2.jpg", "label": "3. วณ 40 (cur_AN24)", "note": "Clean stamped plate; flat baseline"},
    {"fn_plate": "no_3.jpg", "fn_crop": "crop_no_3.jpg", "label": "4. ศณ 32 (cur_AK27)", "note": "Private car series; flat bottom"},
    {"fn_plate": "no_4.jpg", "fn_crop": "crop_no_4.jpg", "label": "5. 1กณ 27 (cur_AE73)", "note": "New 3-prefix series; flat base"},
    {"fn_plate": "no_5.jpg", "fn_crop": "crop_no_5.jpg", "label": "6. ญณ 1411 (cur_AD01)", "note": "Rare double-consonant; flat base"},
    {"fn_plate": "no_6.jpg", "fn_crop": "crop_no_6.jpg", "label": "7. ษณ 3938 (plate_024)", "note": "Previous reference; curved right base"}
]

tho_items = [
    {"fn_plate": "tho_0.jpg", "fn_crop": "crop_tho_0.jpg", "label": "1. ญฒ 5225 (cur_AU76)", "note": "High resolution; clean lean ฒ"},
    {"fn_plate": "tho_1.jpg", "fn_crop": "crop_tho_1.jpg", "label": "2. ฌฒ 1168 (AB14)", "note": "Ultra clean front plate; perfect width"},
    {"fn_plate": "tho_2.jpg", "fn_crop": "crop_tho_2.jpg", "label": "3. 5ขฒ 3555 (cur_AL45)", "note": "High contrast 7-char series; lean ฒ"},
    {"fn_plate": "tho_3.jpg", "fn_crop": "crop_tho_3.jpg", "label": "4. 2ขฒ 7789 (cur_AJ11)", "note": "Standard private pickup; lean ฒ"},
    {"fn_plate": "tho_4.jpg", "fn_crop": "crop_tho_4.jpg", "label": "5. 3กฒ 1 (cur_AE75)", "note": "Auction/single digit plate; lean ฒ"},
    {"fn_plate": "tho_5.jpg", "fn_crop": "crop_tho_5.jpg", "label": "6. 3กฒ 3030 (cur_AM69)", "note": "Daylight CCTV; lean compact ฒ"},
    {"fn_plate": "tho_6.jpg", "fn_crop": "crop_tho_6.jpg", "label": "7. 2ฒร 1387 (plate_012)", "note": "Previous reference; wide & blurred"}
]

do_items = [
    {"fn_plate": "do_0.jpg", "fn_crop": "crop_do_0.jpg", "label": "1. ฎธ 118 (AB25)", "note": "Ultra clean high-res; perfect ฎ die"},
    {"fn_plate": "do_1.jpg", "fn_crop": "crop_do_1.jpg", "label": "2. ฎฉ 199 (cur_AE68)", "note": "Sharp daylight; clear loop & tail"},
    {"fn_plate": "do_2.jpg", "fn_crop": "crop_do_2.jpg", "label": "3. ฎศ 635 (cur_AY64)", "note": "High resolution camera; clean head"},
    {"fn_plate": "do_3.jpg", "fn_crop": "crop_do_3.jpg", "label": "4. ฎห 377 (cur_AM04)", "note": "Sharp front plate; balanced tail"},
    {"fn_plate": "do_4.jpg", "fn_crop": "crop_do_4.jpg", "label": "5. ฎล 250 (cur_AG99)", "note": "Standard DLT series; centered tail"},
    {"fn_plate": "do_5.jpg", "fn_crop": "crop_do_5.jpg", "label": "6. ฎข 1122 (AA80)", "note": "Clean daylight plate; bold stem"},
    {"fn_plate": "do_6.jpg", "fn_crop": "crop_do_6.jpg", "label": "7. 2กฎ 258 (cur_AJ52)", "note": "7-char series; compact ฎ"},
    {"fn_plate": "do_7.jpg", "fn_crop": "crop_do_7.jpg", "label": "8. ศฎ 2900 (plate_019)", "note": "Previous reference; daylight camera"}
]

make_gallery("Thai License Plate Candidates: Flat ณ (No Nen)", "Comparing Real Road Camera Ground Truth for Flat Baseline & Lean Proportions", no_items, "gallery_no_nen.jpg")
make_gallery("Thai License Plate Candidates: Clean ฒ (Tho Phuthao)", "Comparing Real Road Camera Ground Truth for True Die Width & Unblurred Geometry", tho_items, "gallery_tho_phuthao.jpg")
make_gallery("Thai License Plate Candidates: ฎ (Do Chada)", "Ground Truth for Head Placement, Tail Depth, Baseline Loop & Vertical Proportion", do_items, "gallery_do_chada.jpg")
