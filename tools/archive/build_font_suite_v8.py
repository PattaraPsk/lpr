#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Font Suite Builder v8 — Ground Truth Stamped Die Typography System
Includes:
1. rbz-licenseplate.ttf (Master font with 16 DLT calibrated glyphs + complete ญ with pedestal)
2. rbz-licenseplate-id.ttf (Vehicle Registration ID font)
3. rbz-licenseplate-prov.ttf (Shorter, leaner Province font with elevated upper vowels & tone marks)
4. rbz-licenseplate-yellow.ttf (Commercial Transport Yellow Plate font with authentic stadium 0, vertical bar 1, commercial 3, 4, 6, 7, 9, and hyphen)
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
YELLOW_CROPS_DIR = CROPS_DIR / "yellow_plate"

# Load base font
f_master = TTFont(str(MODELS_DIR / "rbz-licenseplate.ttf"))
glyf = f_master['glyf']
hmtx = f_master['hmtx']
cmap = f_master.getBestCmap()

CAP_HEIGHT = 1309.0

# -----------------------------------------------------------------------------
# 1. Update ญ (yoyingthai) with complete body + pedestal from cur_CL88
# -----------------------------------------------------------------------------
print("=" * 80)
print("[*] 1. Calibrating complete ญ with Pedestal (เชิง ญ) from cur_CL88")
print("=" * 80)

yo_img_path = CROPS_DIR / "char_yo_cur_CL88_with_pedestal.png"
im_yo = cv2.imdecode(np.fromfile(str(yo_img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
h_yo, w_yo = im_yo.shape
up_yo = cv2.resize(im_yo, (w_yo * 16, h_yo * 16), interpolation=cv2.INTER_LANCZOS4)
smooth_yo = cv2.GaussianBlur(up_yo, (9, 9), 0)
_, th_yo = cv2.threshold(smooth_yo, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_yo, _ = cv2.findContours(th_yo, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

body_cnt = max(cnts_yo, key=cv2.contourArea)
bx, by, bw, bh = cv2.boundingRect(body_cnt)

scale_yo = CAP_HEIGHT / float(bh)
y_top_yo = CAP_HEIGHT

tpen_yo = TTGlyphPen(glyf)
for cnt in cnts_yo:
    if cv2.contourArea(cnt) < 120: continue
    pts = cnt.squeeze()
    if len(pts) < 3: continue
    pts = [(((p[0] - bx) * scale_yo + 50), (y_top_yo - (p[1] - by) * scale_yo)) for p in pts]
    area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
    if area > 0:
        pts = pts[::-1]
    tpen_yo.moveTo(pts[0])
    for p in pts[1:]: tpen_yo.lineTo(p)
    tpen_yo.closePath()

new_g_yo = tpen_yo.glyph()
new_g_yo.recalcBounds(glyf)
adv_yo = int(round(new_g_yo.xMax + 55))
glyf['yoyingthai'] = new_g_yo
hmtx['yoyingthai'] = (adv_yo, 50)
print(f"  [+] ญ (yoyingthai): bounds=({new_g_yo.xMin}, {new_g_yo.yMin}, {new_g_yo.xMax}, {new_g_yo.yMax}), adv={adv_yo}")

# -----------------------------------------------------------------------------
# 2. Build rbz-licenseplate-id.ttf
# -----------------------------------------------------------------------------
font_id = copy.deepcopy(f_master)
for record in font_id['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-id").encode(record.getEncoding())

# -----------------------------------------------------------------------------
# 3. Build rbz-licenseplate-prov.ttf
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 4. Build rbz-licenseplate-yellow.ttf (Commercial Transport Yellow Plate Font)
# -----------------------------------------------------------------------------
print("=" * 80)
print("[*] 4. Building rbz-licenseplate-yellow.ttf from Commercial Yellow Bus Ground Truth")
print("=" * 80)

font_yellow = copy.deepcopy(f_master)
glyf_yellow = font_yellow['glyf']
hmtx_yellow = font_yellow['hmtx']

def vectorize_yellow_digit(im_path: Path):
    im = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
    smooth = cv2.GaussianBlur(im, (5, 5), 0)
    _, th = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 1500]
    outer_cnt = max(valid_cnts, key=cv2.contourArea)
    ox, oy, ow, oh = cv2.boundingRect(outer_cnt)
    scale = CAP_HEIGHT / float(oh)
    scale_x = scale
    scale_y = scale
    y_top = CAP_HEIGHT
    tpen = TTGlyphPen(glyf_yellow)
    for c in valid_cnts:
        pts = c.squeeze()
        if len(pts) < 3: continue
        pts = [(((p[0] - ox) * scale_x + 50), (y_top - (p[1] - oy) * scale_y)) for p in pts]
        area = 0.5 * sum(pts[i][0]*pts[(i+1)%len(pts)][1] - pts[(i+1)%len(pts)][0]*pts[i][1] for i in range(len(pts)))
        is_outer = (c is outer_cnt)
        if is_outer and area > 0:
            pts = pts[::-1]
        elif not is_outer and area < 0:
            pts = pts[::-1]
        tpen.moveTo(pts[0])
        for p in pts[1:]: tpen.lineTo(p)
        tpen.closePath()
    g = tpen.glyph()
    g.recalcBounds(glyf_yellow)
    adv = int(round(g.xMax + 55))
    return g, (adv, 50)

# Vectorize 0, 1, 3, 4, 6, 7 directly from plate_009 ground truth
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

# 9 is rotated 180 of 6 (identical physical stamped die)
g6 = glyf_yellow['six']
pen6 = RecordingPen()
g6.draw(pen6, glyf_yellow)
tpen9 = TTGlyphPen(glyf_yellow)
cx = (g6.xMin + g6.xMax)
cy = (g6.yMin + g6.yMax)
for cmd, args in pen6.value:
    if cmd == 'moveTo':
        tpen9.moveTo((cx - args[0][0], cy - args[0][1]))
    elif cmd == 'lineTo':
        tpen9.lineTo((cx - args[0][0], cy - args[0][1]))
    elif cmd == 'qCurveTo':
        tpen9.qCurveTo(*[(cx - p[0], cy - p[1]) for p in args])
    elif cmd == 'closePath':
        tpen9.closePath()
g9 = tpen9.glyph()
g9.recalcBounds(glyf_yellow)
glyf_yellow['nine'] = g9
hmtx_yellow['nine'] = hmtx_yellow['six']
print(f"  [+] Yellow Digit 'nine': bounds=({g9.xMin}, {g9.yMin}, {g9.xMax}, {g9.yMax}), adv={hmtx_yellow['nine'][0]}")

# Stamped rectangular hyphen from plate_009
im_hyphen = cv2.imread(str(YELLOW_CROPS_DIR / 'yellow_char_hyphen.png'), cv2.IMREAD_GRAYSCALE)
_, th_h = cv2.threshold(im_hyphen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cnts_h, _ = cv2.findContours(th_h, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
cnt_h = max(cnts_h, key=cv2.contourArea)
hx, hy, hw, hh = cv2.boundingRect(cnt_h)
tpen_hyphen = TTGlyphPen(glyf_yellow)
scale_h_y = 230.0 / float(hh)
scale_h_x = 360.0 / float(hw)
pts_h = [(((p[0] - hx) * scale_h_x + 50), (655 + 115 - (p[1] - hy) * scale_h_y)) for p in cnt_h.squeeze()]
area = 0.5 * sum(pts_h[i][0]*pts_h[(i+1)%len(pts_h)][1] - pts_h[(i+1)%len(pts_h)][0]*pts_h[i][1] for i in range(len(pts_h)))
if area > 0: pts_h = pts_h[::-1]
tpen_hyphen.moveTo(pts_h[0])
for p in pts_h[1:]: tpen_hyphen.lineTo(p)
tpen_hyphen.closePath()
g_hyphen = tpen_hyphen.glyph()
g_hyphen.recalcBounds(glyf_yellow)
glyf_yellow['hyphen'] = g_hyphen
hmtx_yellow['hyphen'] = (int(g_hyphen.xMax + 55), 50)
print(f"  [+] Yellow 'hyphen': bounds=({g_hyphen.xMin}, {g_hyphen.yMin}, {g_hyphen.xMax}, {g_hyphen.yMax}), adv={hmtx_yellow['hyphen'][0]}")

# Calibrate 2, 5, 8 to stadium aspect ratio 0.500 (w = 654)
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

for record in font_yellow['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-yellow").encode(record.getEncoding())

# -----------------------------------------------------------------------------
# 5. Save all 4 fonts to all locations
# -----------------------------------------------------------------------------
font_items = [
    (f_master, "rbz-licenseplate.ttf"),
    (font_id, "rbz-licenseplate-id.ttf"),
    (font_prov, "rbz-licenseplate-prov.ttf"),
    (font_yellow, "rbz-licenseplate-yellow.ttf"),
]

for font_obj, fname in font_items:
    for dest_dir in [MODELS_DIR, DATASET_DIR, BRAIN_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_p = dest_dir / fname
        font_obj.save(str(out_p))
        print(f"[+] Saved: {out_p}")

print("\n[*] Font Suite v8 (Master, ID, Province, Yellow) successfully compiled and installed!")
