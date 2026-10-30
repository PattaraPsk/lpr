#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual Font Builder v7 — Ground Truth Stamped Die System
Calibrated directly from user-specified real road camera DLT plate crops:
1. cur_AY98 -> ฌ (chochoethai) and ฎ (dochadathai)
2. cur_CO06 -> ก (kokaithai) and ธ (thothongthai)
3. cur_CO35 -> ภ (phosamphaothai) and ว (wowaenthai)
4. cur_CO31 -> 3 (three), ก (kokaithai), and ส (sosuathai)
5. cur_CM72 -> 7 (seven), ก (kokaithai), and ฒ (thophuthaothai)
6. cur_CM74 -> ษ (sorusithai) and ณ (nonenthai)
7. cur_CL88 -> ญ (yoyingthai) and ต (totaothai)
8. cur_CK47 -> ฎ (dochadathai) and ป (poplathai) / บ (bobaimaithai)
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
BRAIN_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")
CROPS_DIR = BRAIN_DIR / "user_requested_chars"

# Load base font
f_master = TTFont(str(MODELS_DIR / "rbz-licenseplate.ttf"))
glyf = f_master['glyf']
hmtx = f_master['hmtx']
cmap = f_master.getBestCmap()

CAP_HEIGHT = 1309.0

def vectorize_crop(
    im_path: Path,
    has_tail_down: bool = False,
    tail_down_units: int = 195,
    has_tail_up: bool = False,
    tail_up_units: int = 240,
    lsb: int = 50,
    r_margin: int = 55,
    upsample_f: int = 16,
    morph_close: bool = False
):
    im = cv2.imdecode(np.fromfile(str(im_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise ValueError(f"Could not load image: {im_path}")
    h, w = im.shape
    
    # Upsample with Lanczos4 for subpixel contour fidelity
    up = cv2.resize(im, (w * upsample_f, h * upsample_f), interpolation=cv2.INTER_LANCZOS4)
    smooth = cv2.GaussianBlur(up, (9, 9), 0)
    _, th = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if morph_close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
        
    ys, xs = np.where(th > 0)
    tight = th[ys.min():ys.max()+1, xs.min():xs.max()+1]
    th_h, th_w = tight.shape
    
    # Coordinate scaling to TrueType em-square
    if has_tail_down:
        total_target_h = CAP_HEIGHT + tail_down_units
        scale_y = total_target_h / float(th_h)
        scale_x = scale_y
        y_top = CAP_HEIGHT
    elif has_tail_up:
        total_target_h = CAP_HEIGHT + tail_up_units
        scale_y = total_target_h / float(th_h)
        scale_x = scale_y
        y_top = CAP_HEIGHT + tail_up_units
    else:
        scale_y = CAP_HEIGHT / float(th_h)
        scale_x = scale_y
        y_top = CAP_HEIGHT
        
    cnts, _ = cv2.findContours(tight, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    tpen = TTGlyphPen(glyf)
    for cnt in cnts:
        if cv2.contourArea(cnt) < 120: continue
        pts = cnt.squeeze()
        if len(pts) < 3: continue
        pts = [((p[0] * scale_x + lsb), (y_top - p[1] * scale_y)) for p in pts]
        area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
        if area > 0:
            pts = pts[::-1]
        tpen.moveTo(pts[0])
        for p in pts[1:]:
            tpen.lineTo(p)
        tpen.closePath()
        
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf)
    adv = int(round(new_g.xMax + r_margin))
    return new_g, (adv, lsb)

print("=" * 80)
print("[*] Building rbz-licenseplate v7 directly from User-Specified Ground Truth Plates")
print("=" * 80)

# 1. cur_AY98: ฌ and ฎ
print("--- 1. cur_AY98: ฌ and ฎ ---")
g_cho, h_cho = vectorize_crop(CROPS_DIR / "char_cho_cur_AY98.png")
glyf['chochoethai'] = g_cho
hmtx['chochoethai'] = h_cho
print(f"  [+] ฌ (chochoethai): bounds=({g_cho.xMin}, {g_cho.yMin}, {g_cho.xMax}, {g_cho.yMax}), adv={h_cho[0]}")

g_do, h_do = vectorize_crop(CROPS_DIR / "char_do_cur_AY98.png", has_tail_down=True, tail_down_units=195)
glyf['dochadathai'] = g_do
hmtx['dochadathai'] = h_do
print(f"  [+] ฎ (dochadathai): bounds=({g_do.xMin}, {g_do.yMin}, {g_do.xMax}, {g_do.yMax}), adv={h_do[0]}")

# 2. cur_CO06: ก and ธ
print("--- 2. cur_CO06: ก and ธ ---")
g_ko, h_ko = vectorize_crop(CROPS_DIR / "char_ko_cur_CO06.png")
glyf['kokaithai'] = g_ko
hmtx['kokaithai'] = h_ko
print(f"  [+] ก (kokaithai): bounds=({g_ko.xMin}, {g_ko.yMin}, {g_ko.xMax}, {g_ko.yMax}), adv={h_ko[0]}")

g_tho_thong, h_tho_thong = vectorize_crop(CROPS_DIR / "char_tho_cur_CO06.png")
glyf['thothongthai'] = g_tho_thong
hmtx['thothongthai'] = h_tho_thong
print(f"  [+] ธ (thothongthai): bounds=({g_tho_thong.xMin}, {g_tho_thong.yMin}, {g_tho_thong.xMax}, {g_tho_thong.yMax}), adv={h_tho_thong[0]}")

# 3. cur_CO35: ภ and ว
print("--- 3. cur_CO35: ภ and ว ---")
g_pho, h_pho = vectorize_crop(CROPS_DIR / "char_pho_cur_CO35.png")
glyf['phosamphaothai'] = g_pho
hmtx['phosamphaothai'] = h_pho
print(f"  [+] ภ (phosamphaothai): bounds=({g_pho.xMin}, {g_pho.yMin}, {g_pho.xMax}, {g_pho.yMax}), adv={h_pho[0]}")

g_wo, h_wo = vectorize_crop(CROPS_DIR / "char_wo_cur_CO35.png")
glyf['wowaenthai'] = g_wo
hmtx['wowaenthai'] = h_wo
print(f"  [+] ว (wowaenthai): bounds=({g_wo.xMin}, {g_wo.yMin}, {g_wo.xMax}, {g_wo.yMax}), adv={h_wo[0]}")

# 4. cur_CO31: 3, ก, and ส
print("--- 4. cur_CO31: 3 and ส ---")
g_3, h_3 = vectorize_crop(CROPS_DIR / "char_d3_cur_CO31.png")
glyf['three'] = g_3
hmtx['three'] = h_3
print(f"  [+] 3 (three): bounds=({g_3.xMin}, {g_3.yMin}, {g_3.xMax}, {g_3.yMax}), adv={h_3[0]}")

g_so, h_so = vectorize_crop(CROPS_DIR / "char_so_cur_CO31.png", has_tail_up=True, tail_up_units=210)
glyf['sosuathai'] = g_so
hmtx['sosuathai'] = h_so
print(f"  [+] ส (sosuathai): bounds=({g_so.xMin}, {g_so.yMin}, {g_so.xMax}, {g_so.yMax}), adv={h_so[0]}")

# 5. cur_CM72: 7, ก, and ฒ
print("--- 5. cur_CM72: 7 and ฒ ---")
g_7, h_7 = vectorize_crop(CROPS_DIR / "char_d7_cur_CM72.png")
glyf['seven'] = g_7
hmtx['seven'] = h_7
print(f"  [+] 7 (seven): bounds=({g_7.xMin}, {g_7.yMin}, {g_7.xMax}, {g_7.yMax}), adv={h_7[0]}")

g_tho_phu, h_tho_phu = vectorize_crop(CROPS_DIR / "char_tho_cm72.png")
glyf['thophuthaothai'] = g_tho_phu
hmtx['thophuthaothai'] = h_tho_phu
print(f"  [+] ฒ (thophuthaothai): bounds=({g_tho_phu.xMin}, {g_tho_phu.yMin}, {g_tho_phu.xMax}, {g_tho_phu.yMax}), adv={h_tho_phu[0]}")

# 6. cur_CM74: ษ and ณ (Flat Baseline)
print("--- 6. cur_CM74: ษ and ณ (Flat Base) ---")
g_so_ru, h_so_ru = vectorize_crop(CROPS_DIR / "char_so_cur_CM74.png")
glyf['sorusithai'] = g_so_ru
hmtx['sorusithai'] = h_so_ru
print(f"  [+] ษ (sorusithai): bounds=({g_so_ru.xMin}, {g_so_ru.yMin}, {g_so_ru.xMax}, {g_so_ru.yMax}), adv={h_so_ru[0]}")

g_no, h_no = vectorize_crop(CROPS_DIR / "char_no_cur_CM74.png")
glyf['nonenthai'] = g_no
hmtx['nonenthai'] = h_no
print(f"  [+] ณ (nonenthai): bounds=({g_no.xMin}, {g_no.yMin}, {g_no.xMax}, {g_no.yMax}), adv={h_no[0]}")

# 7. cur_CL88: ญ and ต
print("--- 7. cur_CL88: ญ and ต ---")
g_yo, h_yo = vectorize_crop(CROPS_DIR / "char_yo_cur_CL88.png")
glyf['yoyingthai'] = g_yo
hmtx['yoyingthai'] = h_yo
print(f"  [+] ญ (yoyingthai): bounds=({g_yo.xMin}, {g_yo.yMin}, {g_yo.xMax}, {g_yo.yMax}), adv={h_yo[0]}")

g_to, h_to = vectorize_crop(CROPS_DIR / "char_to_cur_CL88.png")
glyf['totaothai'] = g_to
hmtx['totaothai'] = h_to
print(f"  [+] ต (totaothai): bounds=({g_to.xMin}, {g_to.yMin}, {g_to.xMax}, {g_to.yMax}), adv={h_to[0]}")

# 8. cur_CK47: บ and ป (Flat Base)
print("--- 8. cur_CK47: บ and ป ---")
g_bo, h_bo = vectorize_crop(CROPS_DIR / "char_bo_cur_CK47.png")
glyf['bobaimaithai'] = g_bo
hmtx['bobaimaithai'] = h_bo
print(f"  [+] บ (bobaimaithai): bounds=({g_bo.xMin}, {g_bo.yMin}, {g_bo.xMax}, {g_bo.yMax}), adv={h_bo[0]}")

# Create ป (poplathai) from บ with right stem extended upward to y = 1599 (+290 units)
pen_bo = RecordingPen()
g_bo.draw(pen_bo, glyf)
tpen_po = TTGlyphPen(glyf)
stem_x_threshold = g_bo.xMax - 250
for cmd, args in pen_bo.value:
    if cmd == 'moveTo':
        pt = args[0]
        new_y = pt[1] + 290 if (pt[1] > 1100 and pt[0] > stem_x_threshold) else pt[1]
        tpen_po.moveTo((pt[0], new_y))
    elif cmd == 'lineTo':
        pt = args[0]
        new_y = pt[1] + 290 if (pt[1] > 1100 and pt[0] > stem_x_threshold) else pt[1]
        tpen_po.lineTo((pt[0], new_y))
    elif cmd == 'qCurveTo':
        new_pts = [(p[0], p[1] + 290 if (p[1] > 1100 and p[0] > stem_x_threshold) else p[1]) for p in args]
        tpen_po.qCurveTo(*new_pts)
    elif cmd == 'closePath':
        tpen_po.closePath()

g_po = tpen_po.glyph()
g_po.recalcBounds(glyf)
glyf['poplathai'] = g_po
hmtx['poplathai'] = h_bo
print(f"  [+] ป (poplathai): bounds=({g_po.xMin}, {g_po.yMin}, {g_po.xMax}, {g_po.yMax}), adv={h_bo[0]}")

# -----------------------------------------------------------------------------
# Generate Dual Fonts: rbz-licenseplate-id.ttf & rbz-licenseplate-prov.ttf
# -----------------------------------------------------------------------------
font_id = copy.deepcopy(f_master)
for record in font_id['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-id").encode(record.getEncoding())

font_prov = copy.deepcopy(f_master)
glyf_prov = font_prov['glyf']
hmtx_prov = font_prov['hmtx']
cmap_prov = font_prov.getBestCmap()

prov_sx = 0.88
prov_sy = 0.80

consonants_list = 'กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'
for ch in consonants_list:
    code = ord(ch)
    if code in cmap_prov:
        gname = cmap_prov[code]
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

# Elevate Level 1 Tone Marks: yMin = 1320
l1_tones = ['maiekthai', 'maithothai', 'maitrithai', 'maichattawathai', 'thanthakhatthai']
for tname in l1_tones:
    if tname in glyf_prov:
        g = glyf_prov[tname]
        pen = RecordingPen()
        g.draw(pen, glyf_prov)
        t_shift_y = 1320 - g.yMin
        tpen = TTGlyphPen(glyf_prov)
        for cmd, args in pen.value:
            if cmd == 'moveTo':
                tpen.moveTo((args[0][0] * prov_sx, args[0][1] + t_shift_y))
            elif cmd == 'lineTo':
                tpen.lineTo((args[0][0] * prov_sx, args[0][1] + t_shift_y))
            elif cmd == 'qCurveTo':
                new_pts = [(p[0] * prov_sx, p[1] + t_shift_y) for p in args]
                tpen.qCurveTo(*new_pts)
            elif cmd == 'closePath':
                tpen.closePath()
        new_g = tpen.glyph()
        new_g.recalcBounds(glyf_prov)
        glyf_prov[tname] = new_g

# Elevate Level 2 Tone Marks: yMin = 1800
l2_tones = ['maiekupperthai', 'maithoupperthai', 'maitriupperthai', 'maichattawaupperthai', 'thanthakhatupperthai']
for tname in l2_tones:
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

for record in font_prov['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-prov").encode(record.getEncoding())

# Save to all font destinations
for font_obj, fname in [(f_master, "rbz-licenseplate.ttf"), (font_id, "rbz-licenseplate-id.ttf"), (font_prov, "rbz-licenseplate-prov.ttf")]:
    for dest_dir in [MODELS_DIR, DATASET_DIR, BRAIN_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_p = dest_dir / fname
        font_obj.save(str(out_p))
        print(f"[+] Successfully saved {out_p}")

print("\n[*] Dual Font System v7 successfully compiled and installed!")
