import sys, copy
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
import numpy as np
import cv2
sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"d:\Projects-anti\lpr")
MODELS_DIR = ROOT_DIR / "models" / "fonts"
DATASET_DIR = ROOT_DIR / "dataset" / "fonts"
BRAIN_DIR = Path(r"C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b")
LEELAWADEE_PATH = Path(r"C:\Windows\Fonts\leelawad.ttf")

f_rbz = TTFont(str(MODELS_DIR / "rbz-licenseplate.ttf"))
f_leela = TTFont(str(LEELAWADEE_PATH))

glyf_rbz = f_rbz['glyf']
glyf_leela = f_leela['glyf']
hmtx_rbz = f_rbz['hmtx']
hmtx_leela = f_leela['hmtx']

cap_h_rbz = glyf_rbz['kokaithai'].yMax - glyf_rbz['kokaithai'].yMin
cap_h_leela = glyf_leela['uni0E01'].yMax - glyf_leela['uni0E01'].yMin
scale_f = cap_h_rbz / float(cap_h_leela)

print(f"Cap height rbz: {cap_h_rbz}, scale_f: {scale_f:.3f}")

# -------------------------------------------------------------------------
# 1. ฎ (dochadathai): Bolder (+35 units stroke) & Vertically Shorter (tail -205)
# -------------------------------------------------------------------------
from PIL import Image, ImageFont, ImageDraw
f_temp = ImageFont.truetype(str(LEELAWADEE_PATH), 400)
im = Image.new('L', (600, 700), 0)
draw = ImageDraw.Draw(im)
draw.text((100, 100), 'ฎ', fill=255, font=f_temp)
arr = np.array(im)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
dilated = cv2.dilate(arr, kernel)
ys, xs = np.where(dilated > 100)
cropped = dilated[ys.min():ys.max()+1, xs.min():xs.max()+1]

# Compress bottom tail vertically by 0.52
ch, cw = cropped.shape
split_y = int(ch * 0.65)
top_part = cropped[:split_y, :]
bot_part = cropped[split_y:, :]
bot_compressed = cv2.resize(bot_part, (cw, int(bot_part.shape[0] * 0.52)), interpolation=cv2.INTER_LANCZOS4)
shortened_do = np.vstack([top_part, bot_compressed])

_, binary_do = cv2.threshold(shortened_do, 128, 255, cv2.THRESH_BINARY)
contours_do, _ = cv2.findContours(binary_do, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

target_total_h = 1515 # 1309 cap height + 206 tail
sh_h, sh_w = shortened_do.shape
scale_y = target_total_h / float(sh_h)
scale_x = scale_y * 0.88 # lean

tpen_do = TTGlyphPen(glyf_rbz)
for cnt in contours_do:
    pts = cnt.squeeze()
    if len(pts) < 3: continue
    pts = [((p[0] * scale_x + 55), (1309 - p[1] * scale_y)) for p in pts]
    tpen_do.moveTo(pts[0])
    for p in pts[1:]:
        tpen_do.lineTo(p)
    tpen_do.closePath()

new_g_do = tpen_do.glyph()
new_g_do.recalcBounds(glyf_rbz)
glyf_rbz['dochadathai'] = new_g_do
hmtx_rbz['dochadathai'] = (int(round(sh_w * scale_x + 110)), 55)
print(f"1. ฎ fixed: bounds=({new_g_do.xMin}, {new_g_do.yMin}, {new_g_do.xMax}, {new_g_do.yMax}), adv={hmtx_rbz['dochadathai'][0]}")

# -------------------------------------------------------------------------
# 2. ณ (nonenthai): Authentic Plate Reference from plate_024 (aspect 0.82)
# -------------------------------------------------------------------------
im_no = cv2.imread('real_no_char.png', cv2.IMREAD_GRAYSCALE)
_, th_no = cv2.threshold(im_no, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ys_no, xs_no = np.where(th_no > 0)
tight_no = th_no[ys_no.min():ys_no.max()+1, xs_no.min():xs_no.max()+1]
# Smooth
k_no = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_no = cv2.morphologyEx(tight_no, cv2.MORPH_OPEN, k_no)
cleaned_no = cv2.GaussianBlur(cleaned_no, (3, 3), 0)
_, b_no = cv2.threshold(cleaned_no, 128, 255, cv2.THRESH_BINARY)
cnts_no, _ = cv2.findContours(b_no, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

nh_h, nh_w = b_no.shape
scale_no_y = 1309.0 / float(nh_h)
scale_no_x = scale_no_y * 1.00 # maintains authentic aspect ratio ~0.83!

tpen_no = TTGlyphPen(glyf_rbz)
for cnt in cnts_no:
    if cv2.contourArea(cnt) < 300: continue
    pts = cnt.squeeze()
    if len(pts) < 3: continue
    pts = [((p[0] * scale_no_x + 50), (1309 - p[1] * scale_no_y)) for p in pts]
    tpen_no.moveTo(pts[0])
    for p in pts[1:]:
        tpen_no.lineTo(p)
    tpen_no.closePath()

new_g_no = tpen_no.glyph()
new_g_no.recalcBounds(glyf_rbz)
gname_no = f_rbz.getBestCmap()[ord('ณ')]
glyf_rbz[gname_no] = new_g_no
hmtx_rbz[gname_no] = (int(round(new_g_no.xMax + 60)), 50)
w_no = new_g_no.xMax - new_g_no.xMin
h_no = new_g_no.yMax - new_g_no.yMin
print(f"2. ณ fixed: bounds=({new_g_no.xMin}, {new_g_no.yMin}, {new_g_no.xMax}, {new_g_no.yMax}), w={w_no}, h={h_no}, aspect={w_no/h_no:.3f}, adv={hmtx_rbz[gname_no][0]}")

# -------------------------------------------------------------------------
# 3. ฒ (thophuthaothai): Authentic Plate Reference from plate_012 (aspect ~0.95)
# -------------------------------------------------------------------------
im_tho = cv2.imread('real_tho_char.png', cv2.IMREAD_GRAYSCALE)
ys_tho, xs_tho = np.where(im_tho > 0)
tight_tho = im_tho[ys_tho.min():ys_tho.max()+1, xs_tho.min():xs_tho.max()+1]
k_tho = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned_tho = cv2.morphologyEx(tight_tho, cv2.MORPH_OPEN, k_tho)
cleaned_tho = cv2.GaussianBlur(cleaned_tho, (3, 3), 0)
_, b_tho = cv2.threshold(cleaned_tho, 128, 255, cv2.THRESH_BINARY)
cnts_tho, _ = cv2.findContours(b_tho, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

th_h, th_w = b_tho.shape
scale_tho_y = 1309.0 / float(th_h)
scale_tho_x = scale_tho_y * 0.90 # condense slightly to 0.88 aspect

tpen_tho = TTGlyphPen(glyf_rbz)
for cnt in cnts_tho:
    if cv2.contourArea(cnt) < 500: continue
    pts = cnt.squeeze()
    if len(pts) < 3: continue
    pts = [((p[0] * scale_tho_x + 50), (1309 - p[1] * scale_tho_y)) for p in pts]
    tpen_tho.moveTo(pts[0])
    for p in pts[1:]:
        tpen_tho.lineTo(p)
    tpen_tho.closePath()

new_g_tho = tpen_tho.glyph()
new_g_tho.recalcBounds(glyf_rbz)
gname_tho = f_rbz.getBestCmap()[ord('ฒ')]
glyf_rbz[gname_tho] = new_g_tho
hmtx_rbz[gname_tho] = (int(round(new_g_tho.xMax + 60)), 50)
w_tho = new_g_tho.xMax - new_g_tho.xMin
h_tho = new_g_tho.yMax - new_g_tho.yMin
print(f"3. ฒ fixed: bounds=({new_g_tho.xMin}, {new_g_tho.yMin}, {new_g_tho.xMax}, {new_g_tho.yMax}), w={w_tho}, h={h_tho}, aspect={w_tho/h_tho:.3f}, adv={hmtx_rbz[gname_tho][0]}")

# -------------------------------------------------------------------------
# 4. ร (roruathai): Solid, Clean Authentic License Plate Shape
# -------------------------------------------------------------------------
g_src_ro = glyf_leela['uni0E23']
pen_ro = RecordingPen()
g_src_ro.draw(pen_ro, glyf_leela)
sx_ro = scale_f * 0.95
sy_ro = scale_f * 1.00
shift_ro_x = 40
tpen_ro = TTGlyphPen(glyf_rbz)
for cmd, args in pen_ro.value:
    if cmd == 'moveTo':
        tpen_ro.moveTo((args[0][0] * sx_ro + shift_ro_x, args[0][1] * sy_ro))
    elif cmd == 'lineTo':
        tpen_ro.lineTo((args[0][0] * sx_ro + shift_ro_x, args[0][1] * sy_ro))
    elif cmd == 'qCurveTo':
        new_pts = [(p[0] * sx_ro + shift_ro_x, p[1] * sy_ro) for p in args]
        tpen_ro.qCurveTo(*new_pts)
    elif cmd == 'closePath':
        tpen_ro.closePath()

new_g_ro = tpen_ro.glyph()
new_g_ro.recalcBounds(glyf_rbz)
gname_ro = f_rbz.getBestCmap()[ord('ร')]
glyf_rbz[gname_ro] = new_g_ro
hmtx_rbz[gname_ro] = (int(round(new_g_ro.xMax + 50)), 40)
w_ro = new_g_ro.xMax - new_g_ro.xMin
h_ro = new_g_ro.yMax - new_g_ro.yMin
print(f"4. ร fixed: bounds=({new_g_ro.xMin}, {new_g_ro.yMin}, {new_g_ro.xMax}, {new_g_ro.yMax}), w={w_ro}, h={h_ro}, aspect={w_ro/h_ro:.3f}, adv={hmtx_rbz[gname_ro][0]}")

# -------------------------------------------------------------------------
# 5. Build rbz-licenseplate-id.ttf (Plate ID Font)
# -------------------------------------------------------------------------
font_id = copy.deepcopy(f_rbz)
for record in font_id['name'].names:
    if record.nameID in (1, 4, 6):
        val = record.toUnicode()
        record.string = val.replace("rbz-licenseplate", "rbz-licenseplate-id").encode(record.getEncoding())

# -------------------------------------------------------------------------
# 6. Build rbz-licenseplate-prov.ttf (Province Font)
# With Shorter Consonants (cap ~1050), Elevated Vowels (yMin >= 1320), Elevated Tone Marks (yMin >= 1800)
# -------------------------------------------------------------------------
font_prov = copy.deepcopy(f_rbz)
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

# Elevate Upper Vowels in Province Font:
# In prov font, consonant cap height is 1309 * 0.80 = 1047 units.
# Upper vowels start at yMin = 1320 (leaving a massive 273-unit white gap above consonants!)
upper_vowels_list = ['saraithai', 'saraiithai', 'sarauethai', 'saraueethai', 'maihanakatthai', 'maitaikhuthai', 'nikhahitthai']
for vname in upper_vowels_list:
    if vname in glyf_prov:
        g = glyf_prov[vname]
        pen = RecordingPen()
        g.draw(pen, glyf_prov)
        # In base font, vowel yMin was 1370. We scale x by prov_sx and shift y so yMin starts at 1320
        # g.yMin is 1370. Target yMin = 1320
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

# Elevate Level 1 Tone Marks (when on consonant without upper vowel): yMin = 1320
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

# Elevate Level 2 Tone Marks (when on top of upper vowel): yMin = 1800 (leaving 180-unit gap above vowel arc!)
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

# -------------------------------------------------------------------------
# 7. Save All Fonts
# -------------------------------------------------------------------------
configs = [
    (f_rbz, "rbz-licenseplate.ttf"),
    (font_id, "rbz-licenseplate-id.ttf"),
    (font_prov, "rbz-licenseplate-prov.ttf")
]

for font_obj, fname in configs:
    for dest_dir in [MODELS_DIR, DATASET_DIR, BRAIN_DIR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_p = dest_dir / fname
        font_obj.save(str(out_p))
        print(f"[+] Successfully saved {out_p}")

print("[*] Complete font system v6 successfully built!")
