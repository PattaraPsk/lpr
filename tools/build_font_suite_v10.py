#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Font Suite Builder v10 — Comprehensive Physical Stamped Die Typography System
Directly integrates all characters from user-uploaded plate photographs:
1. Image 1: 8, ก, ณ, 2, 0
2. Image 2: ฎ, บ (and derived ป)
3. Image 3: ษ, 8, 6
4. Image 4: ง, ข, 8, 9
5. Image 5: ช, ฐ (with pedestal), 5, 7, 5, and Province Font (ก, ร, ุ, ง, เ, ท, พ, ม, ห, า, น, ค)
6. Image 6 (media_1793033220592_72fc4eb7.jpg): Pristine Physical Stamped 'ร' (6กร 50)
7. Calibrated Vowels: า (saraaathai), เ (saraethai), ิ (saraithai)
8. Commercial Transport Yellow Plate Font (rbz-licenseplate-yellow.ttf) with unique OpenType name table.
"""

import sys, os, copy
from pathlib import Path
import cv2, numpy as np
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"d:\Projects-anti\lpr")
MODELS_DIR = ROOT_DIR / "models" / "fonts"
DATASET_DIR = ROOT_DIR / "dataset" / "fonts"
MODEL_ALT_DIR = ROOT_DIR / "model" / "fonts"
BRAIN_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")
CROPS_DIR = BRAIN_DIR / "new_user_crops"
YELLOW_CROPS_DIR = BRAIN_DIR / "user_requested_chars" / "yellow_plate"
MEDIA_DIR = BRAIN_DIR / ".system_generated" / "logs" / "media"

# Load base master font
f_master = TTFont(str(MODELS_DIR / "rbz-licenseplate.ttf"))
glyf = f_master['glyf']
hmtx = f_master['hmtx']
cmap = f_master.getBestCmap()

CAP_HEIGHT = 1309.0

def set_font_names(font, family_name: str):
    """Ensure complete, non-conflicting OpenType name table records across Platform 1 and 3."""
    name_table = font['name']
    for record in name_table.names:
        nid = record.nameID
        enc = record.getEncoding()
        if nid == 1:
            record.string = family_name.encode(enc)
        elif nid == 2:
            record.string = "Regular".encode(enc)
        elif nid == 3:
            record.string = f"1.000;RBZ;{family_name}-Regular".encode(enc)
        elif nid == 4:
            record.string = f"{family_name} Regular".encode(enc)
        elif nid == 5:
            record.string = "Version 1.000; 2026".encode(enc)
        elif nid == 6:
            record.string = f"{family_name}-Regular".encode(enc)
        elif nid == 16:
            record.string = family_name.encode(enc)
        elif nid == 17:
            record.string = "Regular".encode(enc)

def vectorize_crop(im_path: Path, is_do: bool = False, thresh_val: int = None, tail_down: int = 178, min_area: int = 400):
    im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Could not load image: {im_path}")
    smooth = cv2.GaussianBlur(im, (5, 5), 0)
    if thresh_val is not None:
        _, th = cv2.threshold(smooth, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, th = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    major = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not major:
        raise ValueError(f"No major contours found in {im_path}")
        
    outer = max(major, key=cv2.contourArea)
    ox, oy, ow, oh = cv2.boundingRect(outer)
    
    if is_do:
        total_target_h = CAP_HEIGHT + tail_down
        scale = total_target_h / float(oh)
        y_top = CAP_HEIGHT
    else:
        scale = CAP_HEIGHT / float(oh)
        y_top = CAP_HEIGHT
        
    tpen = TTGlyphPen(glyf)
    for c in major:
        pts = c.squeeze()
        if len(pts) < 3: continue
        pts = [(((p[0] - ox) * scale + 50), (y_top - (p[1] - oy) * scale)) for p in pts]
        area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
        is_outer_cnt = (c is outer)
        # In TrueType, outer contour must be clockwise (area < 0), inner holes counter-clockwise (area > 0)
        if is_outer_cnt and area > 0:
            pts = pts[::-1]
        elif not is_outer_cnt and area < 0:
            pts = pts[::-1]
        tpen.moveTo(pts[0])
        for p in pts[1:]: tpen.lineTo(p)
        tpen.closePath()
        
    g = tpen.glyph()
    g.recalcBounds(glyf)
    adv = int(round(g.xMax + 55))
    return g, (adv, 50)

print("=" * 80)
print("[*] 1. Calibrating Master Typography from Ground Truth Plates")
print("=" * 80)

# --- Image 1: 8, ณ, 2, 0 ---
print("\n--- Image 1 (8กณ 2020): ณ, 2, 0 ---")
# Note: ก is calibrated with pristine DLT die from 6กร 50 in Image 6

g_no, h_no = vectorize_crop(CROPS_DIR / "img1_no.png")
glyf['nonenthai'] = g_no; hmtx['nonenthai'] = h_no
print(f"  [+] ณ (nonenthai): bounds=({g_no.xMin}, {g_no.yMin}, {g_no.xMax}, {g_no.yMax}), adv={h_no[0]}")

g_2, h_2 = vectorize_crop(CROPS_DIR / "img1_2.png")
glyf['two'] = g_2; hmtx['two'] = h_2
print(f"  [+] 2 (two): bounds=({g_2.xMin}, {g_2.yMin}, {g_2.xMax}, {g_2.yMax}), adv={h_2[0]}")

g_0, h_0 = vectorize_crop(CROPS_DIR / "img_zero.png")
glyf['zero'] = g_0; hmtx['zero'] = h_0
print(f"  [+] 0 (zero): bounds=({g_0.xMin}, {g_0.yMin}, {g_0.xMax}, {g_0.yMax}), adv={h_0[0]}")

# --- Image 2: ฎ, บ (and ป) ---
print("\n--- Image 2 (ฎบ): ฎ, บ, ป ---")
g_do, h_do = vectorize_crop(CROPS_DIR / "img2_do.png", is_do=True, tail_down=183)
glyf['dochadathai'] = g_do; hmtx['dochadathai'] = h_do
print(f"  [+] ฎ (dochadathai): bounds=({g_do.xMin}, {g_do.yMin}, {g_do.xMax}, {g_do.yMax}), adv={h_do[0]}")

g_bo, h_bo = vectorize_crop(CROPS_DIR / "img2_bo.png")
glyf['bobaimaithai'] = g_bo; hmtx['bobaimaithai'] = h_bo
print(f"  [+] บ (bobaimaithai): bounds=({g_bo.xMin}, {g_bo.yMin}, {g_bo.xMax}, {g_bo.yMax}), adv={h_bo[0]}")

# ป directly from media_1793033284500_344423ec.jpg (img_subaru_po_pla.png)
im_po = cv2.imread(str(CROPS_DIR / 'img_subaru_po_pla.png'), cv2.IMREAD_GRAYSCALE)
up_po = cv2.resize(im_po, (im_po.shape[1] * 8, im_po.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_po = cv2.GaussianBlur(up_po, (5, 5), 0)
_, th_po = cv2.threshold(smooth_po, 128, 255, cv2.THRESH_BINARY)
cnts_po, _ = cv2.findContours(th_po, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c0_po = max(cnts_po, key=cv2.contourArea)
bx_po, by_po, bw_po, bh_po = cv2.boundingRect(c0_po)

head_crop = th_po[:, :int(bw_po * 0.45)]
ys_head, _ = np.where(head_crop > 0)
head_top_y = ys_head.min()
base_y = ys_head.max()
head_h = base_y - head_top_y

scale_po = CAP_HEIGHT / float(head_h)
tpen_po = TTGlyphPen(glyf)
pts_po = c0_po.squeeze()
pts_tt_po = [(((p[0] - bx_po) * scale_po + 50), ((base_y - p[1]) * scale_po)) for p in pts_po]
area_po = 0.5 * sum(pts_tt_po[i][0]*pts_tt_po[(i+1)%len(pts_tt_po)][1] - pts_tt_po[(i+1)%len(pts_tt_po)][0]*pts_tt_po[i][1] for i in range(len(pts_tt_po)))
if area_po > 0: pts_tt_po = pts_tt_po[::-1]
tpen_po.moveTo(pts_tt_po[0])
for p in pts_tt_po[1:]: tpen_po.lineTo(p)
tpen_po.closePath()

g_po = tpen_po.glyph()
g_po.recalcBounds(glyf)
adv_po = int(round(g_po.xMax + 55))
glyf['poplathai'] = g_po; hmtx['poplathai'] = (adv_po, 50)
print(f"  [+] ป (poplathai): bounds=({g_po.xMin}, {g_po.yMin}, {g_po.xMax}, {g_po.yMax}), adv={adv_po}")

# --- Image 3: ษ, 6 ---
print("\n--- Image 3 (ษ 8 6): ษ, 6 ---")
g_so_ru, h_so_ru = vectorize_crop(CROPS_DIR / "img3_so_ru.png")
glyf['sorusithai'] = g_so_ru; hmtx['sorusithai'] = h_so_ru
print(f"  [+] ษ (sorusithai): bounds=({g_so_ru.xMin}, {g_so_ru.yMin}, {g_so_ru.xMax}, {g_so_ru.yMax}), adv={h_so_ru[0]}")

g_6, h_6 = vectorize_crop(CROPS_DIR / "img3_6.png")
glyf['six'] = g_6; hmtx['six'] = h_6
print(f"  [+] 6 (six): bounds=({g_6.xMin}, {g_6.yMin}, {g_6.xMax}, {g_6.yMax}), adv={h_6[0]}")

# --- Image 4: ง, ข, 8, 9 ---
print("\n--- Image 4 (งข 898): ง, ข, 8, 9 ---")
g_ngo, h_ngo = vectorize_crop(CROPS_DIR / "img4_ngo.png")
glyf['ngonguthai'] = g_ngo; hmtx['ngonguthai'] = h_ngo
print(f"  [+] ง (ngonguthai): bounds=({g_ngo.xMin}, {g_ngo.yMin}, {g_ngo.xMax}, {g_ngo.yMax}), adv={h_ngo[0]}")

g_kho, h_kho = vectorize_crop(CROPS_DIR / "img4_kho.png")
glyf['khokhaithai'] = g_kho; hmtx['khokhaithai'] = h_kho
print(f"  [+] ข (khokhaithai): bounds=({g_kho.xMin}, {g_kho.yMin}, {g_kho.xMax}, {g_kho.yMax}), adv={h_kho[0]}")

g_8, h_8 = vectorize_crop(CROPS_DIR / "img4_8.png")
glyf['eight'] = g_8; hmtx['eight'] = h_8
print(f"  [+] 8 (eight): bounds=({g_8.xMin}, {g_8.yMin}, {g_8.xMax}, {g_8.yMax}), adv={h_8[0]}")

g_9, h_9 = vectorize_crop(CROPS_DIR / "img4_9.png")
glyf['nine'] = g_9; hmtx['nine'] = h_9
print(f"  [+] 9 (nine): bounds=({g_9.xMin}, {g_9.yMin}, {g_9.xMax}, {g_9.yMax}), adv={h_9[0]}")

# --- Image 5: ช, ฐ (with pedestal), 5, 7 ---
print("\n--- Image 5 (ชฐ 575): ช, ฐ, 5, 7 ---")
g_cho, h_cho = vectorize_crop(CROPS_DIR / "img5_cho.png")
glyf['chochangthai'] = g_cho; hmtx['chochangthai'] = h_cho
print(f"  [+] ช (chochangthai): bounds=({g_cho.xMin}, {g_cho.yMin}, {g_cho.xMax}, {g_cho.yMax}), adv={h_cho[0]}")

im_tho = cv2.imread(str(CROPS_DIR / "img5_tho_than_with_pedestal.png"), cv2.IMREAD_GRAYSCALE)
_, th_tho = cv2.threshold(im_tho, 70, 255, cv2.THRESH_BINARY_INV)
cnts_tho, _ = cv2.findContours(th_tho, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_tho = [c for c in cnts_tho if cv2.contourArea(c) > 500]

scale_tho = CAP_HEIGHT / 182.0
tpen_tho = TTGlyphPen(glyf)
for c in major_tho:
    pts = c.squeeze()
    pts = [(((p[0] - 9) * scale_tho + 50), (CAP_HEIGHT - (p[1] - 6) * scale_tho)) for p in pts]
    area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
    if area > 0: pts = pts[::-1]
    tpen_tho.moveTo(pts[0])
    for p in pts[1:]: tpen_tho.lineTo(p)
    tpen_tho.closePath()

g_tho = tpen_tho.glyph()
g_tho.recalcBounds(glyf)
adv_tho = int(round(g_tho.xMax + 55))
glyf['thothanthai'] = g_tho; hmtx['thothanthai'] = (adv_tho, 50)
print(f"  [+] ฐ (thothanthai): bounds=({g_tho.xMin}, {g_tho.yMin}, {g_tho.xMax}, {g_tho.yMax}), adv={adv_tho}")

g_5, h_5 = vectorize_crop(CROPS_DIR / "img5_5.png")
glyf['five'] = g_5; hmtx['five'] = h_5
print(f"  [+] 5 (five): bounds=({g_5.xMin}, {g_5.yMin}, {g_5.xMax}, {g_5.yMax}), adv={h_5[0]}")

g_7, h_7 = vectorize_crop(CROPS_DIR / "img5_7.png")
glyf['seven'] = g_7; hmtx['seven'] = h_7
print(f"  [+] 7 (seven): bounds=({g_7.xMin}, {g_7.yMin}, {g_7.xMax}, {g_7.yMax}), adv={h_7[0]}")

# --- Image 6: Pristine Physical Stamped 'ก' and 'ร' from 6กร 50 ---
print("\n--- Image 6 (6กร 50): Pristine Physical Stamped ก (kokaithai) and ร (roruathai) ---")

# 1. ก (kokaithai) from 6กร 50
ko_candidates = [
    ROOT_DIR / "dataset" / "fonts" / "crops" / "exact_ko_pad.png",
    CROPS_DIR / "exact_ko_pad.png",
    ROOT_DIR / "dataset" / "fonts" / "crops" / "exact_ko_crop.png",
    CROPS_DIR / "exact_ko_crop.png"
]
ko_path = next((p for p in ko_candidates if p.exists()), None)
if ko_path is None:
    raise FileNotFoundError("Could not find exact_ko_pad.png or exact_ko_crop.png")

im_ko = cv2.imread(str(ko_path), cv2.IMREAD_GRAYSCALE)
up_ko = cv2.resize(im_ko, (im_ko.shape[1] * 8, im_ko.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_ko = cv2.GaussianBlur(up_ko, (5, 5), 0)
_, th_ko = cv2.threshold(smooth_ko, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_ko, _ = cv2.findContours(th_ko, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c0_ko = max(cnts_ko, key=cv2.contourArea)
bx_ko, by_ko, bw_ko, bh_ko = cv2.boundingRect(c0_ko)

scale_ko = CAP_HEIGHT / float(bh_ko)
tpen_ko = TTGlyphPen(glyf)
pts_ko = c0_ko.squeeze()
pts_tt_ko = [(((p[0] - bx_ko) * scale_ko + 50.0), (CAP_HEIGHT - (p[1] - by_ko) * scale_ko)) for p in pts_ko]
area_ko = 0.5 * sum(pts_tt_ko[i][0]*pts_tt_ko[(i+1)%len(pts_tt_ko)][1] - pts_tt_ko[(i+1)%len(pts_tt_ko)][0]*pts_tt_ko[i][1] for i in range(len(pts_tt_ko)))
if area_ko > 0:
    pts_tt_ko = pts_tt_ko[::-1]
tpen_ko.moveTo(pts_tt_ko[0])
for p in pts_tt_ko[1:]:
    tpen_ko.lineTo(p)
tpen_ko.closePath()

g_ko = tpen_ko.glyph()
g_ko.recalcBounds(glyf)
adv_ko = int(round(g_ko.xMax + 55))
glyf['kokaithai'] = g_ko
hmtx['kokaithai'] = (adv_ko, 50)
print(f"  [+] ก (kokaithai): bounds=({g_ko.xMin}, {g_ko.yMin}, {g_ko.xMax}, {g_ko.yMax}), adv={adv_ko}")

# 2. ร (roruathai) from 6กร 50
im_ro = cv2.imread(str(CROPS_DIR / "exact_ro_crop.png"), cv2.IMREAD_GRAYSCALE)
up_ro = cv2.resize(im_ro, (im_ro.shape[1] * 8, im_ro.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_ro = cv2.GaussianBlur(up_ro, (5, 5), 0)
_, th_ro = cv2.threshold(smooth_ro, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_ro, _ = cv2.findContours(th_ro, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_ro = [c for c in cnts_ro if cv2.contourArea(c) > 1000]
c0_ro = major_ro[0]
bx_ro, by_ro, bw_ro, bh_ro = cv2.boundingRect(c0_ro)

scale_ro = CAP_HEIGHT / float(bh_ro)
tpen_ro = TTGlyphPen(glyf)
pts_ro = c0_ro.squeeze()
pts_tt_ro = [(((p[0] - bx_ro) * scale_ro + 50), (CAP_HEIGHT - (p[1] - by_ro) * scale_ro)) for p in pts_ro]
area_ro = 0.5 * sum(pts_tt_ro[i][0]*pts_tt_ro[(i+1)%len(pts_tt_ro)][1] - pts_tt_ro[(i+1)%len(pts_tt_ro)][0]*pts_tt_ro[i][1] for i in range(len(pts_tt_ro)))
if area_ro > 0: pts_tt_ro = pts_tt_ro[::-1]
tpen_ro.moveTo(pts_tt_ro[0])
for p in pts_tt_ro[1:]: tpen_ro.lineTo(p)
tpen_ro.closePath()

g_ro = tpen_ro.glyph()
g_ro.recalcBounds(glyf)
adv_ro = int(round(g_ro.xMax + 55))
glyf['roruathai'] = g_ro; hmtx['roruathai'] = (adv_ro, 50)
print(f"  [+] ร (roruathai): bounds=({g_ro.xMin}, {g_ro.yMin}, {g_ro.xMax}, {g_ro.yMax}), adv={adv_ro}")

# --- Image 7: Pristine Physical Stamped 'ส' and 'ฮ' from media_1793033463881_7cdb4174.jpg (สฮ 365) ---
print("\n--- Image 7 (สฮ 365): Pristine Physical Stamped ส (sosueathai) and ฮ (honokhukthai) ---")
# 1. ส (sosueathai)
im_so = cv2.imread(str(CROPS_DIR / "img_so_suea.png"), cv2.IMREAD_GRAYSCALE)
up_so = cv2.resize(im_so, (im_so.shape[1] * 8, im_so.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_so = cv2.GaussianBlur(up_so, (5, 5), 0)
_, th_so = cv2.threshold(smooth_so, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_so, _ = cv2.findContours(th_so, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_so = [c for c in cnts_so if cv2.contourArea(c) > 500]
outer_so = max(major_so, key=cv2.contourArea)
bx_s, by_s, bw_s, bh_s = cv2.boundingRect(outer_so)

scale_so = CAP_HEIGHT / float(bh_s - 130)
tpen_so = TTGlyphPen(glyf)
c_list_so = [outer_so] + [c for c in major_so if c is not outer_so]
for idx, c in enumerate(c_list_so):
    pts = c.squeeze()
    pts_tt = [(((p[0] - bx_s) * scale_so + 50), ((by_s + bh_s - p[1]) * scale_so)) for p in pts]
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if idx == 0 and area > 0:
        pts_tt = pts_tt[::-1]
    elif idx > 0 and area < 0:
        pts_tt = pts_tt[::-1]
    tpen_so.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen_so.lineTo(p)
    tpen_so.closePath()

g_so = tpen_so.glyph()
g_so.recalcBounds(glyf)
adv_so = int(round(g_so.xMax + 55))
glyf['sosueathai'] = g_so; hmtx['sosueathai'] = (adv_so, 50)
print(f"  [+] ส (sosueathai): bounds=({g_so.xMin}, {g_so.yMin}, {g_so.xMax}, {g_so.yMax}), adv={adv_so}")

# 2. ฮ (honokhukthai)
im_ho = cv2.imread(str(CROPS_DIR / "img_ho_nokhuk.png"), cv2.IMREAD_GRAYSCALE)
up_ho = cv2.resize(im_ho, (im_ho.shape[1] * 8, im_ho.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_ho = cv2.GaussianBlur(up_ho, (5, 5), 0)
_, th_ho = cv2.threshold(smooth_ho, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_ho, _ = cv2.findContours(th_ho, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_ho = [c for c in cnts_ho if cv2.contourArea(c) > 500]
outer_ho = max(major_ho, key=cv2.contourArea)
bx_h, by_h, bw_h, bh_h = cv2.boundingRect(outer_ho)

scale_ho = CAP_HEIGHT / float(bh_h)
tpen_ho = TTGlyphPen(glyf)
c_list_ho = [outer_ho] + [c for c in major_ho if c is not outer_ho]
for idx, c in enumerate(c_list_ho):
    pts = c.squeeze()
    pts_tt = [(((p[0] - bx_h) * scale_ho + 50), ((by_h + bh_h - p[1]) * scale_ho)) for p in pts]
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if idx == 0 and area > 0:
        pts_tt = pts_tt[::-1]
    elif idx > 0 and area < 0:
        pts_tt = pts_tt[::-1]
    tpen_ho.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen_ho.lineTo(p)
    tpen_ho.closePath()

g_ho = tpen_ho.glyph()
g_ho.recalcBounds(glyf)
adv_ho = int(round(g_ho.xMax + 55))
glyf['honokhukthai'] = g_ho; hmtx['honokhukthai'] = (adv_ho, 50)
print(f"  [+] ฮ (honokhukthai): bounds=({g_ho.xMin}, {g_ho.yMin}, {g_ho.xMax}, {g_ho.yMax}), adv={adv_ho}")

# --- Image 8: Pristine Physical Stamped 'ฉ' from media_1793033522995_4a688903.jpg (สฉ 611) ---
print("\n--- Image 8 (สฉ 611): Pristine Physical Stamped ฉ (chochingthai) ---")
im_ch = cv2.imread(str(CROPS_DIR / "img_cho_ching.png"), cv2.IMREAD_GRAYSCALE)
up_ch = cv2.resize(im_ch, (im_ch.shape[1] * 4, im_ch.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
smooth_ch = cv2.GaussianBlur(up_ch, (5, 5), 0)
_, th_ch = cv2.threshold(smooth_ch, 128, 255, cv2.THRESH_BINARY)
cnts_ch, hier_ch = cv2.findContours(th_ch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_ch = [c for c in cnts_ch if cv2.contourArea(c) > 500]
outer_ch = max(major_ch, key=cv2.contourArea)
bx_c, by_c, bw_c, bh_c = cv2.boundingRect(outer_ch)

scale_ch = CAP_HEIGHT / float(bh_c)
tpen_ch = TTGlyphPen(glyf)
c_list_ch = [outer_ch] + [c for c in major_ch if c is not outer_ch]
for idx, c in enumerate(c_list_ch):
    pts = c.squeeze()
    pts_tt = [(((p[0] - bx_c) * scale_ch + 50), ((by_c + bh_c - p[1]) * scale_ch)) for p in pts]
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if idx == 0 and area > 0:
        pts_tt = pts_tt[::-1]
    elif idx > 0 and area < 0:
        pts_tt = pts_tt[::-1]
    tpen_ch.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen_ch.lineTo(p)
    tpen_ch.closePath()

g_ch = tpen_ch.glyph()
g_ch.recalcBounds(glyf)
adv_ch = int(round(g_ch.xMax + 55))
glyf['chochingthai'] = g_ch; hmtx['chochingthai'] = (adv_ch, 50)
print(f"  [+] ฉ (chochingthai): bounds=({g_ch.xMin}, {g_ch.yMin}, {g_ch.xMax}, {g_ch.yMax}), adv={adv_ch}")

# --- Image 9: Pristine Physical Stamped 'ถ' from media_1793033540130_b2eb73e3.jpg (ชถ 269) ---
print("\n--- Image 9 (ชถ 269): Pristine Physical Stamped ถ (thothungthai) ---")
im_th = cv2.imread(str(CROPS_DIR / "img_tho_thung.png"), cv2.IMREAD_GRAYSCALE)
up_th = cv2.resize(im_th, (im_th.shape[1] * 4, im_th.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
smooth_th = cv2.GaussianBlur(up_th, (5, 5), 0)
_, th_th = cv2.threshold(smooth_th, 100, 255, cv2.THRESH_BINARY_INV)
cnts_th, hier_th = cv2.findContours(th_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_th = [c for c in cnts_th if cv2.contourArea(c) > 500]
outer_th = max(major_th, key=cv2.contourArea)
bx_th, by_th, bw_th, bh_th = cv2.boundingRect(outer_th)

scale_th = CAP_HEIGHT / float(bh_th)
tpen_th = TTGlyphPen(glyf)
c_list_th = [outer_th] + [c for c in major_th if c is not outer_th]
for idx, c in enumerate(c_list_th):
    pts = c.squeeze()
    pts_tt = [(((p[0] - bx_th) * scale_th + 50), ((by_th + bh_th - p[1]) * scale_th)) for p in pts]
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if idx == 0 and area > 0:
        pts_tt = pts_tt[::-1]
    elif idx > 0 and area < 0:
        pts_tt = pts_tt[::-1]
    tpen_th.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen_th.lineTo(p)
    tpen_th.closePath()

g_th = tpen_th.glyph()
g_th.recalcBounds(glyf)
adv_th = int(round(g_th.xMax + 55))
glyf['thothungthai'] = g_th; hmtx['thothungthai'] = (adv_th, 50)
print(f"  [+] ถ (thothungthai): bounds=({g_th.xMin}, {g_th.yMin}, {g_th.xMax}, {g_th.yMax}), adv={adv_th}")

# --- Image 10 (ฆด 6886): Pristine Physical Stamped ด (dodekthai) ---
print("\n--- Image 10 (ฆด 6886): Pristine Physical Stamped ด (dodekthai) ---")
im_do_p = CROPS_DIR / "img_do_dek.png"
if not im_do_p.exists():
    p_do_user = BRAIN_DIR / ".user_uploaded" / "media_1793035431423_c55f5ad2.png"
    if p_do_user.exists():
        im_u_do = cv2.imread(str(p_do_user))
        cv2.imwrite(str(im_do_p), im_u_do[65:165, 115:185])

im_do_raw = cv2.imread(str(im_do_p), cv2.IMREAD_GRAYSCALE)
smooth_do = cv2.GaussianBlur(im_do_raw, (5, 5), 0)
_, th_do = cv2.threshold(smooth_do, 80, 255, cv2.THRESH_BINARY_INV)

cnts_do, hier_do = cv2.findContours(th_do, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_do = [c for c in cnts_do if cv2.contourArea(c) > 400]
outer_do = max(major_do, key=cv2.contourArea)
ox_do, oy_do, ow_do, oh_do = cv2.boundingRect(outer_do)

scale_do_y = CAP_HEIGHT / float(oh_do)
scale_do_x = scale_do_y

tpen_do = TTGlyphPen(glyf)
for c in major_do:
    pts = c.squeeze()
    if len(pts) < 3: continue
    pts = [(((p[0] - ox_do) * scale_do_x + 50), ((CAP_HEIGHT - (p[1] - oy_do) * scale_do_y))) for p in pts]
    area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
    is_outer_cnt = (c is outer_do)
    if is_outer_cnt and area > 0:
        pts = pts[::-1]
    elif not is_outer_cnt and area < 0:
        pts = pts[::-1]
    tpen_do.moveTo(pts[0])
    for p in pts[1:]: tpen_do.lineTo(p)
    tpen_do.closePath()

g_do = tpen_do.glyph()
g_do.recalcBounds(glyf)
y_shift_do = -g_do.yMin
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates
g_do.coordinates = GlyphCoordinates([(x, y + y_shift_do) for x, y in g_do.coordinates])
g_do.recalcBounds(glyf)
adv_do = int(round(g_do.xMax + 55))
glyf['dodekthai'] = g_do
hmtx['dodekthai'] = (adv_do, 50)
print(f"  [+] ด (dodekthai): bounds=({g_do.xMin}, {g_do.yMin}, {g_do.xMax}, {g_do.yMax}), adv={adv_do}")

# --- Calibrated Vowels: า, เ, ิ ---
print("\n--- Calibrated Vowels: า, เ, ิ ---")
# า (saraaathai, U+0E32)
im_aa = cv2.imread(str(MEDIA_DIR / "step_2403_media_0.png"), cv2.IMREAD_UNCHANGED)
gray_aa = cv2.cvtColor(im_aa[:, :, :3], cv2.COLOR_BGR2GRAY) if im_aa.shape[2] == 4 else cv2.cvtColor(im_aa, cv2.COLOR_BGR2GRAY)
up_aa = cv2.resize(gray_aa, (gray_aa.shape[1] * 8, gray_aa.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_aa = cv2.GaussianBlur(up_aa, (5, 5), 0)
_, th_aa = cv2.threshold(smooth_aa, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_aa, _ = cv2.findContours(th_aa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c_aa = max(cnts_aa, key=cv2.contourArea)
bx_aa, by_aa, bw_aa, bh_aa = cv2.boundingRect(c_aa)
scale_aa = 1120.0 / float(bh_aa)
tpen_aa = TTGlyphPen(glyf)
pts_aa = c_aa.squeeze()
pts_tt_aa = [(((p[0] - bx_aa) * scale_aa + 40), (1120.0 - (p[1] - by_aa) * scale_aa)) for p in pts_aa]
area_aa = 0.5 * sum(pts_tt_aa[i][0]*pts_tt_aa[(i+1)%len(pts_tt_aa)][1] - pts_tt_aa[(i+1)%len(pts_tt_aa)][0]*pts_tt_aa[i][1] for i in range(len(pts_tt_aa)))
if area_aa > 0: pts_tt_aa = pts_tt_aa[::-1]
tpen_aa.moveTo(pts_tt_aa[0])
for p in pts_tt_aa[1:]: tpen_aa.lineTo(p)
tpen_aa.closePath()
g_aa = tpen_aa.glyph()
g_aa.recalcBounds(glyf)
adv_aa = int(round(g_aa.xMax + 45))
glyf['saraaathai'] = g_aa; hmtx['saraaathai'] = (adv_aa, 40)
print(f"  [+] า (saraaathai): bounds=({g_aa.xMin}, {g_aa.yMin}, {g_aa.xMax}, {g_aa.yMax}), adv={adv_aa}")

# เ (saraethai, U+0E40)
im_e = cv2.imread(str(MEDIA_DIR / "step_2403_media_1.png"), cv2.IMREAD_UNCHANGED)
gray_e = cv2.cvtColor(im_e[:, :, :3], cv2.COLOR_BGR2GRAY) if im_e.shape[2] == 4 else cv2.cvtColor(im_e, cv2.COLOR_BGR2GRAY)
up_e = cv2.resize(gray_e, (gray_e.shape[1] * 8, gray_e.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_e = cv2.GaussianBlur(up_e, (5, 5), 0)
_, th_e = cv2.threshold(smooth_e, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_e, _ = cv2.findContours(th_e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c_e = max(cnts_e, key=cv2.contourArea)
bx_e, by_e, bw_e, bh_e = cv2.boundingRect(c_e)
scale_e = CAP_HEIGHT / float(bh_e)
tpen_e = TTGlyphPen(glyf)
pts_e = c_e.squeeze()
pts_tt_e = [(((p[0] - bx_e) * scale_e + 40), (CAP_HEIGHT - (p[1] - by_e) * scale_e)) for p in pts_e]
area_e = 0.5 * sum(pts_tt_e[i][0]*pts_tt_e[(i+1)%len(pts_tt_e)][1] - pts_tt_e[(i+1)%len(pts_tt_e)][0]*pts_tt_e[i][1] for i in range(len(pts_tt_e)))
if area_e > 0: pts_tt_e = pts_tt_e[::-1]
tpen_e.moveTo(pts_tt_e[0])
for p in pts_tt_e[1:]: tpen_e.lineTo(p)
tpen_e.closePath()
g_e = tpen_e.glyph()
g_e.recalcBounds(glyf)
adv_e = int(round(g_e.xMax + 45))
glyf['saraethai'] = g_e; hmtx['saraethai'] = (adv_e, 40)
print(f"  [+] เ (saraethai): bounds=({g_e.xMin}, {g_e.yMin}, {g_e.xMax}, {g_e.yMax}), adv={adv_e}")

# ิ (saraithai, U+0E34)
im_i = cv2.imread(str(MEDIA_DIR / "step_2403_media_2.png"), cv2.IMREAD_UNCHANGED)
gray_i = cv2.cvtColor(im_i[:, :, :3], cv2.COLOR_BGR2GRAY) if im_i.shape[2] == 4 else cv2.cvtColor(im_i, cv2.COLOR_BGR2GRAY)
up_i = cv2.resize(gray_i, (gray_i.shape[1] * 8, gray_i.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
smooth_i = cv2.GaussianBlur(up_i, (5, 5), 0)
_, th_i = cv2.threshold(smooth_i, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_i, _ = cv2.findContours(th_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c_i = max(cnts_i, key=cv2.contourArea)
bx_i, by_i, bw_i, bh_i = cv2.boundingRect(c_i)

# saraithai is an upper combining vowel typed after base consonant with adv=0.
# In master font, base consonants have adv ~ 900-1050 and right stem at adv - 45.
# Calibrated metrics to sit centered/right-aligned over preceding base consonant:
# width = 720.0, xMax = -80.0 => xMin = -800.0, height = 340.0, yMin = 1350.0.
scale_w_i = 720.0 / float(bw_i)
scale_h_i = 340.0 / float(bh_i)
tpen_i = TTGlyphPen(glyf)
pts_i = c_i.squeeze()
pts_tt_i = [(((p[0] - bx_i) * scale_w_i - 800.0), (1350.0 + 340.0 - (p[1] - by_i) * scale_h_i)) for p in pts_i]
area_i = 0.5 * sum(pts_tt_i[i][0]*pts_tt_i[(i+1)%len(pts_tt_i)][1] - pts_tt_i[(i+1)%len(pts_tt_i)][0]*pts_tt_i[i][1] for i in range(len(pts_tt_i)))
if area_i > 0: pts_tt_i = pts_tt_i[::-1]
tpen_i.moveTo(pts_tt_i[0])
for p in pts_tt_i[1:]: tpen_i.lineTo(p)
tpen_i.closePath()
g_i = tpen_i.glyph()
g_i.recalcBounds(glyf)
glyf['saraithai'] = g_i; hmtx['saraithai'] = (0, int(g_i.xMin))
print(f"  [+] ิ (saraithai): bounds=({g_i.xMin}, {g_i.yMin}, {g_i.xMax}, {g_i.yMax}), adv=0")

# Update master font name table
set_font_names(f_master, "rbz-licenseplate")

# -----------------------------------------------------------------------------
# 2. Build rbz-licenseplate-id.ttf
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[*] 2. Building rbz-licenseplate-id.ttf")
print("=" * 80)
font_id = copy.deepcopy(f_master)
set_font_names(font_id, "rbz-licenseplate-id")
print("  [+] rbz-licenseplate-id name table configured.")

# -----------------------------------------------------------------------------
# 3. Build rbz-licenseplate-prov.ttf (Directly calibrated from Image 5 Province Strip + New Ground Truth ro)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[*] 3. Calibrating rbz-licenseplate-prov.ttf")
print("=" * 80)

font_prov = copy.deepcopy(f_master)
glyf_prov = font_prov['glyf']
hmtx_prov = font_prov['hmtx']

# Base consonant scaling for letters not in Bangkok strip
prov_sx = 0.88
prov_sy = 0.80
consonants_list = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
for ch in consonants_list:
    code = ord(ch)
    if code in font_prov.getBestCmap():
        gname = font_prov.getBestCmap()[code]
        g = glyf_prov[gname]
        pen = RecordingPen()
        g.draw(pen, glyf_prov)
        tpen = TTGlyphPen(glyf_prov)
        for cmd, args in pen.value:
            if cmd == 'moveTo':
                tpen.moveTo((args[0][0] * prov_sx, args[0][1] * prov_sy))
            elif cmd == 'lineTo':
                tpen.lineTo((args[0][0] * prov_sx, args[0][1] * prov_sy))
            elif cmd == 'qCurveTo':
                new_pts = [(p[0] * prov_sx, p[1] * prov_sy) for p in args]
                tpen.qCurveTo(*new_pts)
            elif cmd == 'closePath':
                tpen.closePath()
        new_g = tpen.glyph()
        new_g.recalcBounds(glyf_prov)
        glyf_prov[gname] = new_g
        old_adv, old_lsb = hmtx_prov[gname]
        hmtx_prov[gname] = (int(round(old_adv * prov_sx)), int(round(old_lsb * prov_sx)))

# Vectorize exact Bangkok province glyphs from media_1793033284500_344423ec.jpg (img_subaru_bangkok_strip.png)
p_prov = CROPS_DIR / "img_subaru_bangkok_strip.png"
im_prov = cv2.imread(str(p_prov))
gray_prov = cv2.cvtColor(im_prov, cv2.COLOR_BGR2GRAY)
_, th_p = cv2.threshold(gray_prov, 140, 255, cv2.THRESH_BINARY_INV)

cnts_p, _ = cv2.findContours(th_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
major_p = [c for c in cnts_p if cv2.contourArea(c) > 100]
major_p.sort(key=lambda c: cv2.boundingRect(c)[0])

prov_names = [
    'kokaithai', 'roruathai', 'sarauthai', 'ngonguthai', 'saraethai',
    'thothahanthai', 'phophanthai', 'momathai', 'hohipthai', 'saraaathai',
    'nonuthai', 'khokhwaithai', 'roruathai'
]

PROV_CAP = 1047.0 # 1309 * 0.80
scale_prov_h = PROV_CAP / 43.0
base_y_prov = 52.0

# 1. Vectorize roruathai first (c1) to determine ro_center and adv_ro
c_ro_p = major_p[1]
rx_p, ry_p, rw_p, rh_p = cv2.boundingRect(c_ro_p)
tpen_ro_p = TTGlyphPen(glyf_prov)
pts_ro_p = c_ro_p.squeeze()
pts_tt_ro_p = [(((p[0] - rx_p) * scale_prov_h + 40), ((base_y_prov - p[1]) * scale_prov_h)) for p in pts_ro_p]
area_ro_p = 0.5 * sum(pts_tt_ro_p[i][0]*pts_tt_ro_p[(i+1)%len(pts_tt_ro_p)][1] - pts_tt_ro_p[(i+1)%len(pts_tt_ro_p)][0]*pts_tt_ro_p[i][1] for i in range(len(pts_tt_ro_p)))
if area_ro_p > 0: pts_tt_ro_p = pts_tt_ro_p[::-1]
tpen_ro_p.moveTo(pts_tt_ro_p[0])
for p in pts_tt_ro_p[1:]: tpen_ro_p.lineTo(p)
tpen_ro_p.closePath()
g_ro_p = tpen_ro_p.glyph()
g_ro_p.recalcBounds(glyf_prov)
adv_ro_p = int(round(g_ro_p.xMax + 35))
glyf_prov['roruathai'] = g_ro_p
hmtx_prov['roruathai'] = (adv_ro_p, 40)
print(f"  [+] Prov glyph 'roruathai': bounds=({g_ro_p.xMin}, {g_ro_p.yMin}, {g_ro_p.xMax}, {g_ro_p.yMax}), adv={adv_ro_p}")

# 2. Vectorize remaining Bangkok strip glyphs (handling centered sarauthai under roruathai)
for idx in range(min(12, len(major_p))):
    if idx == 1: continue # already done roruathai
    c = major_p[idx]
    bx, by, bw, bh = cv2.boundingRect(c)
    gname = prov_names[idx]
    
    tpen = TTGlyphPen(glyf_prov)
    pts = c.squeeze()
    if idx == 2:  # ุ (sarauthai, authentic die from user reference step_3368_media_2.png)
        p_u = CROPS_DIR / "user_crop_sarauthai.png"
        if not p_u.exists():
            p_u = MEDIA_DIR / "step_3368_media_2.png"
        im_u = cv2.imread(str(p_u), cv2.IMREAD_GRAYSCALE)
        up_u = cv2.resize(im_u, (im_u.shape[1] * 8, im_u.shape[0] * 8), interpolation=cv2.INTER_CUBIC)
        smooth_u = cv2.GaussianBlur(up_u, (5, 5), 0)
        _, th_u = cv2.threshold(smooth_u, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cnts_u, _ = cv2.findContours(th_u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        c_u = max(cnts_u, key=cv2.contourArea)
        bx_u, by_u, bw_u, bh_u = cv2.boundingRect(c_u)
        
        target_u_h = 320.0
        scale_u = target_u_h / float(bh_u)
        # Bottom hook center of roruathai is at X=600.0, adv_ro_p=903
        # Centering directly under ร's bottom hook: 600.0 - 903 = -303.0
        target_center_x = -303.0
        target_top_y = -70.0
        
        pts_u = c_u.squeeze()
        pts_tt = [(((p[0] - bx_u - bw_u/2.0) * scale_u + target_center_x), (target_top_y - (p[1] - by_u) * scale_u)) for p in pts_u]
    else:
        pts_tt = [(((p[0] - bx) * scale_prov_h + 40), ((base_y_prov - p[1]) * scale_prov_h)) for p in pts]
        
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if area > 0: pts_tt = pts_tt[::-1]
    tpen.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen.lineTo(p)
    tpen.closePath()
    
    new_gp = tpen.glyph()
    new_gp.recalcBounds(glyf_prov)
    if idx == 2:
        hmtx_prov[gname] = (0, int(new_gp.xMin))
        adv_p = 0
    else:
        adv_p = int(round(new_gp.xMax + 35))
        hmtx_prov[gname] = (adv_p, 40)
    glyf_prov[gname] = new_gp
    print(f"  [+] Prov glyph '{gname}': bounds=({new_gp.xMin}, {new_gp.yMin}, {new_gp.xMax}, {new_gp.yMax}), adv={adv_p}")

# 3. Vectorize authentic Nakhon Pathom (นครปฐม) province dies from media_1793035479803_75aa3304.png
# Providing physical stamped dies for:
# - poplathai (ป): authentic elevated tail die
# - thothanthai (ฐ): authentic die body + pedestal resting cleanly below baseline
p_np = CROPS_DIR / "img_nakhon_pathom_strip.png"
if not p_np.exists():
    p_np_user = BRAIN_DIR / ".user_uploaded" / "media_1793035479803_75aa3304.png"
    if p_np_user.exists():
        im_u = cv2.imread(str(p_np_user))
        hu, wu = im_u.shape[:2]
        strip_u = im_u[int(hu*0.58):int(hu*0.93), int(wu*0.35):int(wu*0.72)]
        cv2.imwrite(str(p_np), strip_u)

if p_np.exists():
    im_np = cv2.imread(str(p_np))
    up_np = cv2.resize(im_np, (im_np.shape[1] * 4, im_np.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    gray_np = cv2.cvtColor(up_np, cv2.COLOR_BGR2GRAY)
    smooth_np = cv2.GaussianBlur(gray_np, (5, 5), 0)
    _, th_np = cv2.threshold(smooth_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts_np, hier_np = cv2.findContours(th_np, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    valid_np = []
    for i, c in enumerate(cnts_np):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 1000 and hier_np[0][i][3] == -1 and y < 450:
            valid_np.append((x, y, w, h, area, i))
    valid_np.sort(key=lambda b: b[0])
    
    if len(valid_np) >= 7:
        BASE_Y_NP = 380.0
        SCALE_NP = PROV_CAP / 198.0
        
        # poplathai (idx 3)
        c_po = cnts_np[valid_np[3][5]]
        px, py, pw, ph = cv2.boundingRect(c_po)
        pts_po = c_po.squeeze()
        pts_tt_po = [(((p[0] - px) * SCALE_NP + 40), ((BASE_Y_NP - p[1]) * SCALE_NP)) for p in pts_po]
        area_po = 0.5 * sum(pts_tt_po[i][0]*pts_tt_po[(i+1)%len(pts_tt_po)][1] - pts_tt_po[(i+1)%len(pts_tt_po)][0]*pts_tt_po[i][1] for i in range(len(pts_tt_po)))
        if area_po > 0: pts_tt_po = pts_tt_po[::-1]
        tpen_po = TTGlyphPen(glyf_prov)
        tpen_po.moveTo(pts_tt_po[0])
        for p in pts_tt_po[1:]: tpen_po.lineTo(p)
        tpen_po.closePath()
        g_po = tpen_po.glyph()
        g_po.recalcBounds(glyf_prov)
        adv_po = int(round(g_po.xMax + 40))
        glyf_prov['poplathai'] = g_po
        hmtx_prov['poplathai'] = (adv_po, 40)
        print(f"  [+] Prov glyph 'poplathai' (Nakhon Pathom die): bounds=({g_po.xMin}, {g_po.yMin}, {g_po.xMax}, {g_po.yMax}), adv={adv_po}")
        
        # thothanthai (idx 4 body, idx 5 pedestal)
        c_tho_body = cnts_np[valid_np[4][5]]
        c_tho_ped = cnts_np[valid_np[5][5]]
        bx_body, by_body, bw_body, bh_body = cv2.boundingRect(c_tho_body)
        
        pts_body = [(((p[0] - bx_body) * SCALE_NP + 40), ((BASE_Y_NP - p[1]) * SCALE_NP)) for p in c_tho_body.squeeze()]
        area_b = 0.5 * sum(pts_body[i][0]*pts_body[(i+1)%len(pts_body)][1] - pts_body[(i+1)%len(pts_body)][0]*pts_body[i][1] for i in range(len(pts_body)))
        if area_b > 0: pts_body = pts_body[::-1]
        
        pts_ped = [(((p[0] - bx_body) * SCALE_NP + 40), ((BASE_Y_NP - p[1]) * SCALE_NP)) for p in c_tho_ped.squeeze()]
        area_p = 0.5 * sum(pts_ped[i][0]*pts_ped[(i+1)%len(pts_ped)][1] - pts_ped[(i+1)%len(pts_ped)][0]*pts_ped[i][1] for i in range(len(pts_ped)))
        if area_p > 0: pts_ped = pts_ped[::-1]
        
        tpen_tho = TTGlyphPen(glyf_prov)
        tpen_tho.moveTo(pts_body[0])
        for p in pts_body[1:]: tpen_tho.lineTo(p)
        tpen_tho.closePath()
        tpen_tho.moveTo(pts_ped[0])
        for p in pts_ped[1:]: tpen_tho.lineTo(p)
        tpen_tho.closePath()
        g_tho = tpen_tho.glyph()
        g_tho.recalcBounds(glyf_prov)
        adv_tho = int(round(g_tho.xMax + 40))
        glyf_prov['thothanthai'] = g_tho
        hmtx_prov['thothanthai'] = (adv_tho, 40)
        print(f"  [+] Prov glyph 'thothanthai' (Nakhon Pathom die + pedestal): bounds=({g_tho.xMin}, {g_tho.yMin}, {g_tho.xMax}, {g_tho.yMax}), adv={adv_tho}")

# Elevate Upper Vowels in Province Font: yMin starts at 1320
upper_vowels_list = ['saraithai', 'saraiithai', 'sarauethai', 'saraueethai', 'maihanakatthai', 'maitaikhuthai', 'nikhahitthai']
for vname in upper_vowels_list:
    if vname in glyf_prov:
        g = glyf_prov[vname]
        pen = RecordingPen()
        g.draw(pen, glyf_prov)
        v_shift_y = 1320 - g.yMin
        tpen = TTGlyphPen(glyf_prov)
        for cmd, args in pen.value:
            if cmd == 'moveTo':
                tpen.moveTo((args[0][0] * prov_sx, args[0][1] + v_shift_y))
            elif cmd == 'lineTo':
                tpen.lineTo((args[0][0] * prov_sx, args[0][1] + v_shift_y))
            elif cmd == 'qCurveTo':
                new_pts = [(p[0] * prov_sx, p[1] + v_shift_y) for p in args]
                tpen.qCurveTo(*new_pts)
            elif cmd == 'closePath':
                tpen.closePath()
        new_g = tpen.glyph()
        new_g.recalcBounds(glyf_prov)
        glyf_prov[vname] = new_g
        hmtx_prov[vname] = (0, int(new_g.xMin))

# Elevate Tone Marks in Province Font: yMin starts at 1800
tone_marks_list = ['maiekthai', 'maithothai', 'maitrithai', 'maichattawathai', 'thanthakhatthai']
for tname in tone_marks_list:
    if tname in glyf_prov:
        g = glyf_prov[tname]
        pen = RecordingPen()
        g.draw(pen, glyf_prov)
        t_shift_y2 = 1800 - g.yMin
        tpen = TTGlyphPen(glyf_prov)
        for cmd, args in pen.value:
            if cmd == 'moveTo':
                tpen.moveTo((args[0][0] * prov_sx, args[0][1] + t_shift_y2))
            elif cmd == 'lineTo':
                tpen.lineTo((args[0][0] * prov_sx, args[0][1] + t_shift_y2))
            elif cmd == 'qCurveTo':
                new_pts = [(p[0] * prov_sx, p[1] + t_shift_y2) for p in args]
                tpen.qCurveTo(*new_pts)
            elif cmd == 'closePath':
                tpen.closePath()
        new_g = tpen.glyph()
        new_g.recalcBounds(glyf_prov)
        glyf_prov[tname] = new_g
        hmtx_prov[tname] = (0, int(new_g.xMin))

# 4. Vectorize authentic Chiang Mai (เชียงใหม่) province dies from media_1793035561680_f2bee60d.png
# Providing physical stamped dies for:
# - yoyakthai (ย): authentic dual-notch cheek die
# - saraaimaimuanthai (ใ): authentic looping ascender die
# - saraiithai (ี): authentic upper vowel die
# - maiekthai (่): authentic tone mark die
p_cm = CROPS_DIR / "img_chiangmai_strip.png"
if not p_cm.exists():
    p_cm_user = BRAIN_DIR / ".user_uploaded" / "media_1793035561680_f2bee60d.png"
    if p_cm_user.exists():
        im_u_cm = cv2.imread(str(p_cm_user))
        h_cm, w_cm = im_u_cm.shape[:2]
        strip_cm = im_u_cm[int(h_cm*0.53):int(h_cm*0.77), int(w_cm*0.34):int(w_cm*0.66)]
        cv2.imwrite(str(p_cm), strip_cm)

if p_cm.exists():
    im_cm = cv2.imread(str(p_cm))
    up_cm = cv2.resize(im_cm, (im_cm.shape[1] * 4, im_cm.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    gray_cm = cv2.cvtColor(up_cm, cv2.COLOR_BGR2GRAY)
    smooth_cm = cv2.GaussianBlur(gray_cm, (5, 5), 0)
    _, th_cm = cv2.threshold(smooth_cm, 110, 255, cv2.THRESH_BINARY_INV)
    cnts_cm, hier_cm = cv2.findContours(th_cm, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    valid_cm = []
    for i, c in enumerate(cnts_cm):
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area > 100 and hier_cm[0][i][3] == -1 and y < 270 and w < 600:
            valid_cm.append((x, y, w, h, area, i))
    valid_cm.sort(key=lambda b: b[0])
    
    if len(valid_cm) >= 9:
        BASE_Y_CM = 233.0
        SCALE_CM = PROV_CAP / 119.0
        
        # saraiithai (idx 1: ี) — shifted right by +200 font units to align directly over consonant right stem (e.g. บ, ช)
        c_ii = cnts_cm[valid_cm[1][5]]
        bx_ii, by_ii, bw_ii, bh_ii = cv2.boundingRect(c_ii)
        pen_after_ch = 203.3  # Shifted right by 22.7px (+200 font units in 1047-cap space)
        pts_ii = c_ii.squeeze()
        pts_tt_ii = [(((p[0] - pen_after_ch) * SCALE_CM), ((BASE_Y_CM - p[1]) * SCALE_CM + (1320.0 - (BASE_Y_CM - (by_ii + bh_ii)) * SCALE_CM))) for p in pts_ii]
        area_ii = 0.5 * sum(pts_tt_ii[i][0]*pts_tt_ii[(i+1)%len(pts_tt_ii)][1] - pts_tt_ii[(i+1)%len(pts_tt_ii)][0]*pts_tt_ii[i][1] for i in range(len(pts_tt_ii)))
        if area_ii > 0: pts_tt_ii = pts_tt_ii[::-1]
        tpen_ii = TTGlyphPen(glyf_prov)
        tpen_ii.moveTo(pts_tt_ii[0])
        for p in pts_tt_ii[1:]: tpen_ii.lineTo(p)
        tpen_ii.closePath()
        g_ii = tpen_ii.glyph()
        g_ii.recalcBounds(glyf_prov)
        glyf_prov['saraiithai'] = g_ii
        hmtx_prov['saraiithai'] = (0, int(g_ii.xMin))
        print(f"  [+] Prov glyph 'saraiithai' (Chiang Mai die): bounds=({g_ii.xMin}, {g_ii.yMin}, {g_ii.xMax}, {g_ii.yMax}), adv=0")
        
        # yoyakthai (idx 3: ย)
        c_yo = cnts_cm[valid_cm[3][5]]
        bx_yo, by_yo, bw_yo, bh_yo = cv2.boundingRect(c_yo)
        pts_yo = c_yo.squeeze()
        pts_tt_yo = [(((p[0] - bx_yo) * SCALE_CM + 40), ((BASE_Y_CM - p[1]) * SCALE_CM)) for p in pts_yo]
        area_yo = 0.5 * sum(pts_tt_yo[i][0]*pts_tt_yo[(i+1)%len(pts_tt_yo)][1] - pts_tt_yo[(i+1)%len(pts_tt_yo)][0]*pts_tt_yo[i][1] for i in range(len(pts_tt_yo)))
        if area_yo > 0: pts_tt_yo = pts_tt_yo[::-1]
        tpen_yo = TTGlyphPen(glyf_prov)
        tpen_yo.moveTo(pts_tt_yo[0])
        for p in pts_tt_yo[1:]: tpen_yo.lineTo(p)
        tpen_yo.closePath()
        g_yo = tpen_yo.glyph()
        g_yo.recalcBounds(glyf_prov)
        adv_yo = int(round(g_yo.xMax + 40))
        glyf_prov['yoyakthai'] = g_yo
        hmtx_prov['yoyakthai'] = (adv_yo, 40)
        print(f"  [+] Prov glyph 'yoyakthai' (Chiang Mai die): bounds=({g_yo.xMin}, {g_yo.yMin}, {g_yo.xMax}, {g_yo.yMax}), adv={adv_yo}")
        
        # saraaimaimuanthai (idx 5: ใ)
        c_muan = cnts_cm[valid_cm[5][5]]
        bx_muan, by_muan, bw_muan, bh_muan = cv2.boundingRect(c_muan)
        pts_muan = c_muan.squeeze()
        pts_tt_muan = [(((p[0] - bx_muan) * SCALE_CM + 40), ((BASE_Y_CM - p[1]) * SCALE_CM)) for p in pts_muan]
        area_muan = 0.5 * sum(pts_tt_muan[i][0]*pts_tt_muan[(i+1)%len(pts_tt_muan)][1] - pts_tt_muan[(i+1)%len(pts_tt_muan)][0]*pts_tt_muan[i][1] for i in range(len(pts_tt_muan)))
        if area_muan > 0: pts_tt_muan = pts_tt_muan[::-1]
        tpen_muan = TTGlyphPen(glyf_prov)
        tpen_muan.moveTo(pts_tt_muan[0])
        for p in pts_tt_muan[1:]: tpen_muan.lineTo(p)
        tpen_muan.closePath()
        g_muan = tpen_muan.glyph()
        g_muan.recalcBounds(glyf_prov)
        adv_muan = int(round(g_muan.xMax + 40))
        glyf_prov['saraaimaimuanthai'] = g_muan
        hmtx_prov['saraaimaimuanthai'] = (adv_muan, 40)
        print(f"  [+] Prov glyph 'saraaimaimuanthai' (Chiang Mai die): bounds=({g_muan.xMin}, {g_muan.yMin}, {g_muan.xMax}, {g_muan.yMax}), adv={adv_muan}")
        
        # maiekthai (idx 8: ่)
        c_ek = cnts_cm[valid_cm[8][5]]
        bx_ek, by_ek, bw_ek, bh_ek = cv2.boundingRect(c_ek)
        pen_after_mo = 737.0
        pts_ek = c_ek.squeeze()
        pts_tt_ek = [(((p[0] - pen_after_mo) * SCALE_CM), ((BASE_Y_CM - p[1]) * SCALE_CM + (1800.0 - (BASE_Y_CM - (by_ek + bh_ek)) * SCALE_CM))) for p in pts_ek]
        area_ek = 0.5 * sum(pts_tt_ek[i][0]*pts_tt_ek[(i+1)%len(pts_tt_ek)][1] - pts_tt_ek[(i+1)%len(pts_tt_ek)][0]*pts_tt_ek[i][1] for i in range(len(pts_tt_ek)))
        if area_ek > 0: pts_tt_ek = pts_tt_ek[::-1]
        tpen_ek = TTGlyphPen(glyf_prov)
        tpen_ek.moveTo(pts_tt_ek[0])
        for p in pts_tt_ek[1:]: tpen_ek.lineTo(p)
        tpen_ek.closePath()
        g_ek = tpen_ek.glyph()
        g_ek.recalcBounds(glyf_prov)
        glyf_prov['maiekthai'] = g_ek
        hmtx_prov['maiekthai'] = (0, int(g_ek.xMin))
        print(f"  [+] Prov glyph 'maiekthai' (Chiang Mai die): bounds=({g_ek.xMin}, {g_ek.yMin}, {g_ek.xMax}, {g_ek.yMax}), adv=0")

set_font_names(font_prov, "rbz-licenseplate-province")
print("  [+] rbz-licenseplate-province name table configured.")

# -----------------------------------------------------------------------------
# 4. Build rbz-licenseplate-yellow.ttf (Commercial Transport Yellow Plate Font)
# Combines:
#   - Authentic commercial yellow transport numerals (0-9, hyphen)
#   - Authentic physical stamped English header THAILAND dies (T, H, A, I, L, N, D)
#   - Full suite of calibrated province glyphs (so it handles ID, province, & header)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[*] 4. Building rbz-licenseplate-yellow.ttf from Commercial Yellow Bus Ground Truth")
print("=" * 80)

# Start from font_prov to inherit all calibrated province typography, elevated vowels, and tone marks
font_yellow = copy.deepcopy(font_prov)
glyf_yellow = font_yellow['glyf']
hmtx_yellow = font_yellow['hmtx']

def vectorize_yellow_digit(im_path: Path):
    im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Could not load image: {im_path}")
    smooth = cv2.GaussianBlur(im, (5, 5), 0)
    _, th = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    major = [c for c in cnts if cv2.contourArea(c) > 300]
    outer = max(major, key=cv2.contourArea)
    ox, oy, ow, oh = cv2.boundingRect(outer)
    scale = CAP_HEIGHT / float(oh)
    tpen = TTGlyphPen(glyf_yellow)
    for c in major:
        pts = c.squeeze()
        if len(pts) < 3: continue
        pts = [(((p[0] - ox) * scale + 50), (CAP_HEIGHT - (p[1] - oy) * scale)) for p in pts]
        area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
        is_outer_cnt = (c is outer)
        if is_outer_cnt and area > 0:
            pts = pts[::-1]
        elif not is_outer_cnt and area < 0:
            pts = pts[::-1]
        tpen.moveTo(pts[0])
        for p in pts[1:]: tpen.lineTo(p)
        tpen.closePath()
    g = tpen.glyph()
    g.recalcBounds(glyf_yellow)
    adv = int(round(g.xMax + 55))
    return g, (adv, 50)

yellow_digits_map = [
    ('zero', 'yellow_char_0.png'),
    ('one', 'yellow_char_1.png'),
    ('three', 'yellow_char_3.png'),
    ('four', 'yellow_char_4.png'),
    ('six', 'yellow_char_6.png'),
    ('seven', 'yellow_char_7.png'),
]

for d_name, fname in yellow_digits_map:
    g, h = vectorize_yellow_digit(YELLOW_CROPS_DIR / fname)
    glyf_yellow[d_name] = g
    hmtx_yellow[d_name] = h
    print(f"  [+] Yellow Digit '{d_name}': bounds=({g.xMin}, {g.yMin}, {g.xMax}, {g.yMax}), adv={h[0]}")

# Derive 9 by rotating 6 by 180 degrees
g6 = glyf_yellow['six']
pen6 = RecordingPen()
g6.draw(pen6, glyf_yellow)
tpen9 = TTGlyphPen(glyf_yellow)
cx = (g6.xMin + g6.xMax) / 2.0
cy = (g6.yMin + g6.yMax) / 2.0
for cmd, args in pen6.value:
    if cmd == 'moveTo':
        tpen9.moveTo((2 * cx - args[0][0], 2 * cy - args[0][1]))
    elif cmd == 'lineTo':
        tpen9.lineTo((2 * cx - args[0][0], 2 * cy - args[0][1]))
    elif cmd == 'qCurveTo':
        new_pts = [(2 * cx - p[0], 2 * cy - p[1]) for p in args]
        tpen9.qCurveTo(*new_pts)
    elif cmd == 'closePath':
        tpen9.closePath()
g9 = tpen9.glyph()
g9.recalcBounds(glyf_yellow)
glyf_yellow['nine'] = g9
hmtx_yellow['nine'] = hmtx_yellow['six']
print(f"  [+] Yellow Digit 'nine': bounds=({g9.xMin}, {g9.yMin}, {g9.xMax}, {g9.yMax}), adv={hmtx_yellow['nine'][0]}")

# Yellow hyphen from commercial bus plate ground truth
im_hyphen = cv2.imread(str(YELLOW_CROPS_DIR / 'yellow_char_hyphen.png'), cv2.IMREAD_GRAYSCALE)
_, th_h = cv2.threshold(im_hyphen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_h, _ = cv2.findContours(th_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
c_hyphen = max(cnts_h, key=cv2.contourArea)
hx, hy, hw, hh = cv2.boundingRect(c_hyphen)
scale_h = 180.0 / float(hh)
tpen_hyphen = TTGlyphPen(glyf_yellow)
pts_h = c_hyphen.squeeze()
pts_tt_h = [(((p[0] - hx) * scale_h + 50), (600.0 - (p[1] - hy) * scale_h)) for p in pts_h]
area_h = 0.5 * sum(pts_tt_h[i][0]*pts_tt_h[(i+1)%len(pts_tt_h)][1] - pts_tt_h[(i+1)%len(pts_tt_h)][0]*pts_tt_h[i][1] for i in range(len(pts_tt_h)))
if area_h > 0: pts_tt_h = pts_tt_h[::-1]
tpen_hyphen.moveTo(pts_tt_h[0])
for p in pts_tt_h[1:]: tpen_hyphen.lineTo(p)
tpen_hyphen.closePath()
g_hyphen = tpen_hyphen.glyph()
g_hyphen.recalcBounds(glyf_yellow)
glyf_yellow['hyphen'] = g_hyphen
hmtx_yellow['hyphen'] = (int(g_hyphen.xMax + 55), 50)
print(f"  [+] Yellow 'hyphen': bounds=({g_hyphen.xMin}, {g_hyphen.yMin}, {g_hyphen.xMax}, {g_hyphen.yMax}), adv={hmtx_yellow['hyphen'][0]}")

# Scale 2, 5, 8 to authentic yellow plate width proportions
target_w_yellow = 654.0
for d_other in ['two', 'five', 'eight']:
    g_orig = glyf_yellow[d_other]
    orig_w = g_orig.xMax - g_orig.xMin
    scale_x_d = target_w_yellow / float(orig_w)
    pen = RecordingPen()
    g_orig.draw(pen, glyf_yellow)
    tpen = TTGlyphPen(glyf_yellow)
    for cmd, args in pen.value:
        if cmd == 'moveTo':
            tpen.moveTo(((args[0][0] - g_orig.xMin) * scale_x_d + 50, args[0][1]))
        elif cmd == 'lineTo':
            tpen.lineTo(((args[0][0] - g_orig.xMin) * scale_x_d + 50, args[0][1]))
        elif cmd == 'qCurveTo':
            new_pts = [((p[0] - g_orig.xMin) * scale_x_d + 50, p[1]) for p in args]
            tpen.qCurveTo(*new_pts)
        elif cmd == 'closePath':
            tpen.closePath()
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf_yellow)
    glyf_yellow[d_other] = new_g
    hmtx_yellow[d_other] = (int(round(new_g.xMax + 55)), 50)
    print(f"  [+] Yellow Digit '{d_other}': bounds=({new_g.xMin}, {new_g.yMin}, {new_g.xMax}, {new_g.yMax}), adv={hmtx_yellow[d_other][0]}")

# Vectorize English THAILAND dies directly from user reference crop (yellow_selection_2.png)
p_thailand = CROPS_DIR / "yellow_selection_2.png"
if not p_thailand.exists():
    p_thailand = MEDIA_DIR / "step_3209_media_2.png"

if p_thailand.exists():
    im_th_raw = cv2.imread(str(p_thailand))
    gray_th = cv2.cvtColor(im_th_raw, cv2.COLOR_BGR2GRAY)
    up_th = cv2.resize(gray_th, (gray_th.shape[1] * 4, gray_th.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
    smooth_th = cv2.GaussianBlur(up_th, (3, 3), 0)
    _, th_bin = cv2.threshold(smooth_th, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts_th, hier_th = cv2.findContours(th_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    top_th_cnts = []
    for i, c in enumerate(cnts_th):
        if hier_th[0][i][3] == -1:
            c_area = cv2.contourArea(c)
            bx, by, bw, bh = cv2.boundingRect(c)
            if c_area > 100 and bh > 30:
                top_th_cnts.append((bx, by, bw, bh, c_area, i))
    top_th_cnts.sort(key=lambda b: b[0])

    if len(top_th_cnts) >= 8:
        BASE_Y_TH = 98.5
        H_REF_TH = 85.5
        scale_th = CAP_HEIGHT / H_REF_TH
        char_labels_th = ['T', 'H', 'A', 'I', 'L', 'A2', 'N', 'D']

        for idx, (ox, oy, ow, oh, oarea, ci) in enumerate(top_th_cnts):
            label = char_labels_th[idx]
            if label == 'A2':
                continue  # A1 already processed for 'A'
            children = [child_i for child_i, h_elem in enumerate(hier_th[0]) if h_elem[3] == ci and cv2.contourArea(cnts_th[child_i]) > 50]
            
            tpen_th = TTGlyphPen(glyf_yellow)
            
            # Outer contour
            c_out = cnts_th[ci]
            pts = c_out.squeeze()
            pts_tt = [(((p[0] - ox) * scale_th + 50), ((BASE_Y_TH - p[1]) * scale_th)) for p in pts]
            area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
            if area > 0:
                pts_tt = pts_tt[::-1]
            tpen_th.moveTo(pts_tt[0])
            for p in pts_tt[1:]: tpen_th.lineTo(p)
            tpen_th.closePath()
            
            # Inner holes (for A and D)
            for ch_i in children:
                c_hole = cnts_th[ch_i]
                pts_h = c_hole.squeeze()
                pts_tt_h = [(((p[0] - ox) * scale_th + 50), ((BASE_Y_TH - p[1]) * scale_th)) for p in pts_h]
                area_h = 0.5 * sum(pts_tt_h[i][0]*pts_tt_h[(i+1)%len(pts_tt_h)][1] - pts_tt_h[(i+1)%len(pts_tt_h)][0]*pts_tt_h[i][1] for i in range(len(pts_tt_h)))
                if area_h < 0:
                    pts_tt_h = pts_tt_h[::-1]
                tpen_th.moveTo(pts_tt_h[0])
                for p in pts_tt_h[1:]: tpen_th.lineTo(p)
                tpen_th.closePath()
                
            g_th = tpen_th.glyph()
            g_th.recalcBounds(glyf_yellow)
            adv_th = int(round(g_th.xMax + 55))
            glyf_yellow[label] = g_th
            hmtx_yellow[label] = (adv_th, 50)
            print(f"  [+] Yellow Header Glyph '{label}': bounds=({g_th.xMin}, {g_th.yMin}, {g_th.xMax}, {g_th.yMax}), adv={adv_th}")

set_font_names(font_yellow, "rbz-licenseplate-yellow")
print("  [+] rbz-licenseplate-yellow name table configured.")

# -----------------------------------------------------------------------------
# 4b. Height Normalization across Font Families
# Normalizes every character's height and aligns baseline to 0.0:
#   - Digits (0-9) share exact baseline y=0.0 and cap-height y=1309.0
#   - Standard Thai consonants share exact baseline y=0.0 and cap-height
#   - Province typography normalized to exact province cap-height (1047.0)
#   - THAILAND uppercase Latin letters normalized to exact cap-height (1309.0)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[*] 4b. Normalizing Character Heights and Flat Baselines Across Font Suite")
print("=" * 80)

def normalize_font_heights(font, target_cap: float = 1309.0, is_prov: bool = False):
    glyf = font['glyf']
    hmtx = font['hmtx']
    pad_adv = 35 if is_prov else 55
    
    standard_chars = [
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
        'kokaithai', 'khokhaithai', 'khokhuatthai', 'khokhwaithai', 'khokhonthai', 'khorakhangthai',
        'ngonguthai', 'chochanthai', 'chochingthai', 'chochangthai', 'chochoethai',
        'thonangmonthothai', 'thophuthaothai', 'nonenthai', 'dodekthai', 'totaothai',
        'thothungthai', 'thothahanthai', 'thothongthai', 'nonuthai', 'bobaimaithai',
        'phophungthai', 'phophanthai', 'phosamphaothai', 'momathai', 'yoyakthai',
        'roruathai', 'lolingthai', 'wowaenthai', 'sorusithai', 'hohipthai',
        'oangthai', 'honokhukthai'
    ]
    if not is_prov:
        standard_chars += ['T', 'H', 'A', 'I', 'L', 'N', 'D']
        
    for name in standard_chars:
        if name not in glyf: continue
        g = glyf[name]
        orig_h = float(g.yMax - g.yMin)
        if orig_h <= 0: continue
        scale = target_cap / orig_h
        ymin_orig = float(g.yMin)
        xmin_orig = float(g.xMin)
        
        pen = RecordingPen()
        g.draw(pen, glyf)
        tpen = TTGlyphPen(glyf)
        for cmd, args in pen.value:
            if cmd == 'moveTo':
                tpen.moveTo(((args[0][0] - xmin_orig) * scale + xmin_orig, (args[0][1] - ymin_orig) * scale))
            elif cmd == 'lineTo':
                tpen.lineTo(((args[0][0] - xmin_orig) * scale + xmin_orig, (args[0][1] - ymin_orig) * scale))
            elif cmd == 'qCurveTo':
                new_pts = [((p[0] - xmin_orig) * scale + xmin_orig, (p[1] - ymin_orig) * scale) for p in args]
                tpen.qCurveTo(*new_pts)
            elif cmd == 'closePath':
                tpen.closePath()
        new_g = tpen.glyph()
        new_g.recalcBounds(glyf)
        glyf[name] = new_g
        new_adv = int(round(new_g.xMax + pad_adv))
        hmtx[name] = (new_adv, int(new_g.xMin))

    # Dochadathai (ฎ): align top to target_cap
    if 'dochadathai' in glyf:
        g = glyf['dochadathai']
        shift = target_cap - g.yMax
        if abs(shift) > 0.5:
            pen = RecordingPen(); g.draw(pen, glyf); tpen = TTGlyphPen(glyf)
            for cmd, args in pen.value:
                if cmd == 'moveTo': tpen.moveTo((args[0][0], args[0][1] + shift))
                elif cmd == 'lineTo': tpen.lineTo((args[0][0], args[0][1] + shift))
                elif cmd == 'qCurveTo': tpen.qCurveTo(*[(p[0], p[1] + shift) for p in args])
                elif cmd == 'closePath': tpen.closePath()
            new_g = tpen.glyph(); new_g.recalcBounds(glyf); glyf['dochadathai'] = new_g

    # Poplathai (ป): shift yMin to 0.0 so bowl sits on flat baseline
    if 'poplathai' in glyf:
        g = glyf['poplathai']
        shift = -g.yMin
        if abs(shift) > 0.5:
            pen = RecordingPen(); g.draw(pen, glyf); tpen = TTGlyphPen(glyf)
            for cmd, args in pen.value:
                if cmd == 'moveTo': tpen.moveTo((args[0][0], args[0][1] + shift))
                elif cmd == 'lineTo': tpen.lineTo((args[0][0], args[0][1] + shift))
                elif cmd == 'qCurveTo': tpen.qCurveTo(*[(p[0], p[1] + shift) for p in args])
                elif cmd == 'closePath': tpen.closePath()
            new_g = tpen.glyph(); new_g.recalcBounds(glyf); glyf['poplathai'] = new_g

    # Thothanthai (ฐ): scale so body reaches target_cap
    if 'thothanthai' in glyf:
        g = glyf['thothanthai']
        scale = target_cap / float(g.yMax)
        if abs(scale - 1.0) > 0.005:
            pen = RecordingPen(); g.draw(pen, glyf); tpen = TTGlyphPen(glyf)
            for cmd, args in pen.value:
                if cmd == 'moveTo': tpen.moveTo((args[0][0] * scale, args[0][1] * scale))
                elif cmd == 'lineTo': tpen.lineTo((args[0][0] * scale, args[0][1] * scale))
                elif cmd == 'qCurveTo': tpen.qCurveTo(*[(p[0] * scale, p[1] * scale) for p in args])
                elif cmd == 'closePath': tpen.closePath()
            new_g = tpen.glyph(); new_g.recalcBounds(glyf); glyf['thothanthai'] = new_g

    # Other Ascender characters: shift yMin to 0.0
    for asc in ['sosueathai', 'sosalathai', 'sosothai', 'fofathai', 'fofanthai', 'lochulathai']:
        if asc in glyf:
            g = glyf[asc]
            shift = -g.yMin
            if abs(shift) > 0.5:
                pen = RecordingPen(); g.draw(pen, glyf); tpen = TTGlyphPen(glyf)
                for cmd, args in pen.value:
                    if cmd == 'moveTo': tpen.moveTo((args[0][0], args[0][1] + shift))
                    elif cmd == 'lineTo': tpen.lineTo((args[0][0], args[0][1] + shift))
                    elif cmd == 'qCurveTo': tpen.qCurveTo(*[(p[0], p[1] + shift) for p in args])
                    elif cmd == 'closePath': tpen.closePath()
                new_g = tpen.glyph(); new_g.recalcBounds(glyf); glyf[asc] = new_g

# Sarauthai in master and id fonts
if 'sarauthai' in glyf_prov:
    f_master['glyf']['sarauthai'] = glyf_prov['sarauthai']
    f_master['hmtx']['sarauthai'] = (0, int(glyf_prov['sarauthai'].xMin))
    font_id['glyf']['sarauthai'] = glyf_prov['sarauthai']
    font_id['hmtx']['sarauthai'] = (0, int(glyf_prov['sarauthai'].xMin))

normalize_font_heights(f_master, target_cap=1309.0, is_prov=False)
normalize_font_heights(font_id, target_cap=1309.0, is_prov=False)
normalize_font_heights(font_prov, target_cap=1047.0, is_prov=True)

# For yellow font: normalize digits and THAILAND letters to 1309.0
yellow_chars = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'T', 'H', 'A', 'I', 'L', 'N', 'D']
for name in yellow_chars:
    if name in glyf_yellow:
        g = glyf_yellow[name]
        orig_h = float(g.yMax - g.yMin)
        if orig_h > 0:
            scale = 1309.0 / orig_h
            ymin_orig = float(g.yMin)
            xmin_orig = float(g.xMin)
            pen = RecordingPen(); g.draw(pen, glyf_yellow); tpen = TTGlyphPen(glyf_yellow)
            for cmd, args in pen.value:
                if cmd == 'moveTo': tpen.moveTo(((args[0][0] - xmin_orig) * scale + xmin_orig, (args[0][1] - ymin_orig) * scale))
                elif cmd == 'lineTo': tpen.lineTo(((args[0][0] - xmin_orig) * scale + xmin_orig, (args[0][1] - ymin_orig) * scale))
                elif cmd == 'qCurveTo': tpen.qCurveTo(*[((p[0] - xmin_orig) * scale + xmin_orig, (p[1] - ymin_orig) * scale) for p in args])
                elif cmd == 'closePath': tpen.closePath()
            new_g = tpen.glyph(); new_g.recalcBounds(glyf_yellow); glyf_yellow[name] = new_g
            hmtx_yellow[name] = (int(round(new_g.xMax + 55)), int(new_g.xMin))

print("  [+] All font families normalized with uniform cap-height and flat baseline at 0.0.")

# -----------------------------------------------------------------------------
# 5. Save all fonts to all target locations
# -----------------------------------------------------------------------------
font_items = [
    (f_master, "rbz-licenseplate.ttf"),
    (font_id, "rbz-licenseplate-id.ttf"),
    (font_prov, "rbz-licenseplate-province.ttf"),
    (font_prov, "rbz-licenseplate-prov.ttf"),
    (font_yellow, "rbz-licenseplate-yellow.ttf"),
]

CURR_BRAIN_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\dead7011-d0b0-4ee7-ab6f-11e77224b85f")
for font_obj, fname in font_items:
    for dest_dir in [MODELS_DIR, DATASET_DIR, MODEL_ALT_DIR, BRAIN_DIR, CURR_BRAIN_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_p = dest_dir / fname
        font_obj.save(str(out_p))
        print(f"[+] Saved: {out_p}")

print("\n[*] Font Suite v10 successfully compiled and installed across all target directories!")
