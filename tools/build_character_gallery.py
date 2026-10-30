#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, glob
import cv2, numpy as np
from PIL import Image, ImageFont, ImageDraw

sys.stdout.reconfigure(encoding='utf-8')
inspect_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\inspect_candidates"
artifact_dir = r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b"

font_title = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 16)
font_label = ImageFont.truetype(r"C:\Windows\Fonts\tahomabd.ttf", 13)
font_sub = ImageFont.truetype(r"C:\Windows\Fonts\tahoma.ttf", 11)

no_specs = [
    ("no_0.jpg", "สณ 332 (cur_AK43)", 1),
    ("no_1.jpg", "วณ 315 (cur_AK00)", 1),
    ("no_2.jpg", "วณ 40 (cur_AN24)", 1),
    ("no_3.jpg", "ศณ 32 (cur_AK27)", 1),
    ("no_4.jpg", "1กณ 27 (cur_AE73)", 2),
    ("no_5.jpg", "ญณ 1411 (cur_AD01)", 1),
    ("no_6.jpg", "ษณ 3938 (plate_024)", 1),
]

tho_specs = [
    ("tho_0.jpg", "ญฒ 5225 (cur_AU76)", 1),
    ("tho_1.jpg", "ฌฒ 1168 (AB14)", 1),
    ("tho_2.jpg", "5ขฒ 3555 (cur_AL45)", 2),
    ("tho_3.jpg", "2ขฒ 7789 (cur_AJ11)", 2),
    ("tho_4.jpg", "3กฒ 1 (cur_AE75)", 2),
    ("tho_5.jpg", "3กฒ 3030 (cur_AM69)", 2),
    ("tho_6.jpg", "2ฒร 1387 (plate_012)", 1),
]

do_specs = [
    ("do_0.jpg", "ฎธ 118 (AB25)", 0),
    ("do_1.jpg", "ฎฉ 199 (cur_AE68)", 0),
    ("do_2.jpg", "ฎศ 635 (cur_AY64)", 0),
    ("do_3.jpg", "ฎห 377 (cur_AM04)", 0),
    ("do_4.jpg", "ฎล 250 (cur_AG99)", 0),
    ("do_5.jpg", "ฎข 1122 (AA80)", 0),
    ("do_6.jpg", "2กฎ 258 (cur_AJ52)", 2),
    ("do_7.jpg", "ศฎ 2900 (plate_019)", 1),
]

def extract_char_crop(fn, title, target_idx):
    p = os.path.join(inspect_dir, fn)
    if not os.path.exists(p):
        print(f"Missing file: {fn}")
        return None
    im = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print(f"Failed to decode: {fn}")
        return None
    h, w = im.shape[:2]
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
    
    comps = []
    # Find registration characters (tall, upper half of plate)
    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        if ch > int(h * 0.35) and y < int(h * 0.55) and area > 60:
            comps.append((x, y, cw, ch, area))
            
    comps.sort(key=lambda c: c[0])
    
    if target_idx < len(comps):
        x, y, cw, ch, area = comps[target_idx]
        pad = max(2, int(ch * 0.05))
        y1 = max(0, y - pad)
        y2 = min(h, y + ch + pad)
        x1 = max(0, x - pad)
        x2 = min(w, x + cw + pad)
        crop = im[y1:y2, x1:x2]
        return {
            "title": title,
            "plate_im": im,
            "crop": crop,
            "w": cw,
            "h": ch,
            "aspect": cw / float(ch),
            "fn": fn
        }
    else:
        # Fallback if component sorting missed:
        print(f"  Warning: target_idx {target_idx} >= len(comps) {len(comps)} for {title}")
        return None

def build_gallery_for_group(grp_title, specs, out_img_name):
    extracted = []
    for fn, title, idx in specs:
        res = extract_char_crop(fn, title, idx)
        if res:
            extracted.append(res)
            print(f"[{grp_title[:2]}] {title}: bbox=({res['w']}x{res['h']}), aspect={res['aspect']:.3f}")
            
    if not extracted:
        return
        
    card_w = 260
    card_h = 320
    cols = 4
    rows = (len(extracted) + cols - 1) // cols
    
    sheet_w = 40 + cols * (card_w + 20)
    sheet_h = 100 + rows * (card_h + 20)
    
    sheet_pil = Image.new("RGB", (sheet_w, sheet_h), (250, 252, 255))
    draw = ImageDraw.Draw(sheet_pil)
    
    # Header
    draw.rounded_rectangle([20, 15, sheet_w - 20, 75], radius=8, fill=(240, 245, 250), outline=(210, 220, 235), width=2)
    draw.text((35, 25), f"Real Plate Candidates Gallery: {grp_title}", fill=(15, 23, 42), font=font_title)
    draw.text((35, 50), f"Ground truth measurements: Width, Height, Aspect Ratio & Base Flatness", fill=(71, 85, 105), font=font_sub)
    
    for i, item in enumerate(extracted):
        col = i % cols
        row = i // cols
        cx = 30 + col * (card_w + 20)
        cy = 90 + row * (card_h + 20)
        
        # Card background
        draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h], radius=8, fill=(255, 255, 255), outline=(220, 228, 238), width=1)
        draw.text((cx + 12, cy + 10), item["title"], fill=(30, 41, 59), font=font_label)
        draw.text((cx + 12, cy + 30), f"Size: {item['w']}x{item['h']}px | Aspect: {item['aspect']:.3f}", fill=(20, 100, 40) if 0.70 <= item['aspect'] <= 0.88 else (180, 50, 20), font=font_sub)
        
        # Draw full plate crop (scaled to card_w - 24, h=70)
        plate_res = cv2.resize(item["plate_im"], (card_w - 24, 70), interpolation=cv2.INTER_AREA)
        plate_rgb = cv2.cvtColor(plate_res, cv2.COLOR_BGR2RGB)
        sheet_pil.paste(Image.fromarray(plate_rgb), (cx + 12, cy + 55))
        
        # Draw isolated character crop (scaled preserving aspect ratio to fit 150x150)
        ch_crop = item["crop"]
        ch_h, ch_w = ch_crop.shape[:2]
        scale = min(150 / float(ch_h), (card_w - 24) / float(ch_w))
        new_cw = int(ch_w * scale)
        new_ch = int(ch_h * scale)
        ch_res = cv2.resize(ch_crop, (new_cw, new_ch), interpolation=cv2.INTER_LANCZOS4)
        ch_rgb = cv2.cvtColor(ch_res, cv2.COLOR_BGR2RGB)
        
        # Center in character box
        box_x = cx + 12 + (card_w - 24 - new_cw) // 2
        box_y = cy + 135 + (150 - new_ch) // 2
        sheet_pil.paste(Image.fromarray(ch_rgb), (box_x, box_y))
        draw.rounded_rectangle([cx + 12, cy + 135, cx + card_w - 12, cy + 295], radius=4, outline=(230, 235, 245), width=1)
        
    out_path = os.path.join(artifact_dir, out_img_name)
    sheet_pil.save(out_path)
    print(f"[+] Saved gallery to: {out_path}")

print("=== 1. Building ณ Gallery ===")
build_gallery_for_group("ณ (No Nen) Candidates — Seeking Flat Base & Authentic Geometry", no_specs, "gallery_no_nen.jpg")

print("=== 2. Building ฒ Gallery ===")
build_gallery_for_group("ฒ (Tho Phuthao) Candidates — Seeking Clean Contours & True Die Width", tho_specs, "gallery_tho_phuthao.jpg")

print("=== 3. Building ฎ Gallery ===")
build_gallery_for_group("ฎ (Do Chada) Candidates — Seeking Head, Tail & Baseline Ground Truth", do_specs, "gallery_do_chada.jpg")
