#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Font Suite Builder v9 — User Reference Stamped Die Typography System
Directly integrates all characters from the 5 user-uploaded plate photographs:
1. Image 1: 8, ก, ณ, 2, 0
2. Image 2: ฎ, บ (and derived ป)
3. Image 3: ษ, 8, 6
4. Image 4: ง, ข, 8, 9
5. Image 5: ช, ฐ (with pedestal), 5, 7, 5, and Province Font (ก, ร, ุ, ง, เ, ท, พ, ม, ห, า, น, ค)
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
CROPS_DIR = BRAIN_DIR / "new_user_crops"
YELLOW_CROPS_DIR = BRAIN_DIR / "user_requested_chars" / "yellow_plate"

# Load base master font
f_master = TTFont(str(MODELS_DIR / "rbz-licenseplate.ttf"))
glyf = f_master['glyf']
hmtx = f_master['hmtx']
cmap = f_master.getBestCmap()

CAP_HEIGHT = 1309.0

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
print("[*] 1. Calibrating Characters from 5 User Uploaded Real Plate Images")
print("=" * 80)

# --- Image 1: 8, ก, ณ, 2, 0 ---
print("\n--- Image 1 (8กณ 2020): ก, ณ, 2, 0 ---")
g_ko, h_ko = vectorize_crop(CROPS_DIR / "img1_ko.png")
glyf['kokaithai'] = g_ko; hmtx['kokaithai'] = h_ko
print(f"  [+] ก (kokaithai): bounds=({g_ko.xMin}, {g_ko.yMin}, {g_ko.xMax}, {g_ko.yMax}), adv={h_ko[0]}")

g_no, h_no = vectorize_crop(CROPS_DIR / "img1_no.png")
glyf['nonenthai'] = g_no; hmtx['nonenthai'] = h_no
print(f"  [+] ณ (nonenthai): bounds=({g_no.xMin}, {g_no.yMin}, {g_no.xMax}, {g_no.yMax}), adv={h_no[0]}")

g_2, h_2 = vectorize_crop(CROPS_DIR / "img1_2.png")
glyf['two'] = g_2; hmtx['two'] = h_2
print(f"  [+] 2 (two): bounds=({g_2.xMin}, {g_2.yMin}, {g_2.xMax}, {g_2.yMax}), adv={h_2[0]}")

# 0 directly from media_1793030925861_d37e572c.jpg (img_zero.png)
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

# ป directly from media_1793030925861_d37e572c.jpg (img_po_pla.png)
im_po = cv2.imread(str(CROPS_DIR / 'img_po_pla.png'), cv2.IMREAD_GRAYSCALE)
smooth_po = cv2.GaussianBlur(im_po, (5, 5), 0)
_, th_po = cv2.threshold(smooth_po, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_po, _ = cv2.findContours(th_po, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
outer_po = max(cnts_po, key=cv2.contourArea)
px, py, pw, ph = cv2.boundingRect(outer_po)

# Plate numeral height in this image is ~170px -> scale = CAP_HEIGHT / 170.0
scale_po = CAP_HEIGHT / 170.0
tpen_po = TTGlyphPen(glyf)
pts_po = outer_po.squeeze()
pts_tt_po = [(((p[0] - px) * scale_po + 50), ((py + ph - p[1]) * scale_po)) for p in pts_po]
area = 0.5 * sum(pts_tt_po[i][0]*pts_tt_po[(i+1)%len(pts_tt_po)][1] - pts_tt_po[(i+1)%len(pts_tt_po)][0]*pts_tt_po[i][1] for i in range(len(pts_tt_po)))
if area > 0: pts_tt_po = pts_tt_po[::-1]
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

# ฐ (thothanthai) with body + pedestal
im_tho = cv2.imread(str(CROPS_DIR / "img5_tho_than_with_pedestal.png"), cv2.IMREAD_GRAYSCALE)
_, th_tho = cv2.threshold(im_tho, 70, 255, cv2.THRESH_BINARY_INV)
cnts_tho, _ = cv2.findContours(th_tho, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_tho = [c for c in cnts_tho if cv2.contourArea(c) > 500]

# Digits in Image 5 have height 182 px -> scale = 1309.0 / 182.0
scale_tho = CAP_HEIGHT / 182.0
# Crop coordinate offsets: y=6 is cap height, baseline is at y=188
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

# -----------------------------------------------------------------------------
# 2. Build rbz-licenseplate-id.ttf
# -----------------------------------------------------------------------------
font_id = copy.deepcopy(f_master)
for record in font_id['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-id").encode(record.getEncoding())

# -----------------------------------------------------------------------------
# 3. Build rbz-licenseplate-prov.ttf (Directly calibrated from Image 5 Province Strip)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[*] 3. Calibrating rbz-licenseplate-prov.ttf from Image 5 Province Strip")
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

# Vectorize exact province glyphs from Image 5
p_prov = CROPS_DIR / "img5_province_strip.png"
im_prov = cv2.imread(str(p_prov))
gray_prov = cv2.cvtColor(im_prov, cv2.COLOR_BGR2GRAY)
_, th_p = cv2.threshold(gray_prov, 70, 255, cv2.THRESH_BINARY_INV)
th_p[105:, :] = 0  # strip bottom edge

cnts_p, hier_p = cv2.findContours(th_p, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
major_p = [(c, hier_p[0][idx]) for idx, c in enumerate(cnts_p) if cv2.contourArea(c) > 200]
major_p.sort(key=lambda item: cv2.boundingRect(item[0])[0])

prov_names = [
    'kokaithai', 'roruathai', 'sarauthai', 'ngonguthai', 'saraethai',
    'thothahanthai', 'phophanthai', 'momathai', 'hohipthai', 'saraaathai',
    'nonuthai', 'khokhwaithai'
]

PROV_CAP = 1047.0 # 1309 * 0.80
scale_prov_h = PROV_CAP / 58.0

# First vectorize roruathai to get its geometry and metrics
c_ro = major_p[1][0]
bx_ro, by_ro, bw_ro, bh_ro = cv2.boundingRect(c_ro)
tpen_ro = TTGlyphPen(glyf_prov)
pts_ro = [(((p[0] - bx_ro) * scale_prov_h + 40), (PROV_CAP - (p[1] - 18) * scale_prov_h)) for p in c_ro.squeeze()]
area_ro = 0.5 * sum(pts_ro[i][0]*pts_ro[(i+1)%len(pts_ro)][1] - pts_ro[(i+1)%len(pts_ro)][0]*pts_ro[i][1] for i in range(len(pts_ro)))
if area_ro > 0: pts_ro = pts_ro[::-1]
tpen_ro.moveTo(pts_ro[0])
for p in pts_ro[1:]: tpen_ro.lineTo(p)
tpen_ro.closePath()
g_ro = tpen_ro.glyph()
g_ro.recalcBounds(glyf_prov)
adv_ro = int(round(g_ro.xMax + 35)) # Authentic tight kerning
glyf_prov['roruathai'] = g_ro
hmtx_prov['roruathai'] = (adv_ro, 40)
print(f"  [+] Prov glyph 'roruathai': bounds=({g_ro.xMin}, {g_ro.yMin}, {g_ro.xMax}, {g_ro.yMax}), adv={adv_ro}")

# Now vectorize all glyphs (handling zero-advance centered sarauthai under roruathai)
for idx in range(min(12, len(major_p))):
    if idx == 1: continue # already done roruathai
    c, h_node = major_p[idx]
    bx, by, bw, bh = cv2.boundingRect(c)
    gname = prov_names[idx]
    
    tpen = TTGlyphPen(glyf_prov)
    pts = c.squeeze()
    if idx == 2:  # ุ (sarauthai, lower vowel)
        # Shift X to sit centered directly underneath roruathai (when positioned following roruathai)
        u_center_init = (bw * scale_prov_h) / 2.0 + 40
        ro_center = (g_ro.xMin + g_ro.xMax) / 2.0
        shift_back = ro_center - adv_ro - u_center_init
        pts_tt = [(((p[0] - bx) * scale_prov_h + 40 + shift_back), ((76 - p[1]) * scale_prov_h)) for p in pts]
    else:
        pts_tt = [(((p[0] - bx) * scale_prov_h + 40), (PROV_CAP - (p[1] - 18) * scale_prov_h)) for p in pts]
        
    area = 0.5 * sum(pts_tt[i][0]*pts_tt[(i+1)%len(pts_tt)][1] - pts_tt[(i+1)%len(pts_tt)][0]*pts_tt[i][1] for i in range(len(pts_tt)))
    if area > 0: pts_tt = pts_tt[::-1]
    tpen.moveTo(pts_tt[0])
    for p in pts_tt[1:]: tpen.lineTo(p)
    tpen.closePath()
    
    new_gp = tpen.glyph()
    new_gp.recalcBounds(glyf_prov)
    if idx == 2:
        # Zero advance for lower combining vowel
        adv_p = 0
        hmtx_prov[gname] = (0, int(new_gp.xMin))
    else:
        adv_p = int(round(new_gp.xMax + 45))
        hmtx_prov[gname] = (adv_p, 40)
    glyf_prov[gname] = new_gp
    print(f"  [+] Prov glyph '{gname}': bounds=({new_gp.xMin}, {new_gp.yMin}, {new_gp.xMax}, {new_gp.yMax}), adv={adv_p}")

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

# -----------------------------------------------------------------------------
# 4. Save Fonts
# -----------------------------------------------------------------------------
font_items = [
    (f_master, "rbz-licenseplate.ttf"),
    (font_id, "rbz-licenseplate-id.ttf"),
    (font_prov, "rbz-licenseplate-prov.ttf"),
]

for font_obj, fname in font_items:
    for dest_dir in [MODELS_DIR, DATASET_DIR, BRAIN_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_p = dest_dir / fname
        font_obj.save(str(out_p))
        print(f"[+] Saved: {out_p}")

print("\n[*] Font Suite v9 successfully compiled and installed!")
