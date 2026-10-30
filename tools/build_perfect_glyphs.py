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

# Consonant cap height in rbz
cap_h_rbz = glyf_rbz['kokaithai'].yMax - glyf_rbz['kokaithai'].yMin
cap_h_leela = glyf_leela['uni0E01'].yMax - glyf_leela['uni0E01'].yMin
scale_f = cap_h_rbz / float(cap_h_leela)

print(f"Cap height rbz: {cap_h_rbz}, scale_f: {scale_f:.3f}")

# 1. FIX ฎ (dochadathai): Bolder (+35 units) and Vertically Shorter (tail compressed)
# In Leelawadee: uni0E0E has top head and center tail.
# We extract uni0E0E, embolden stroke by morphological dilation, and shorten tail so yMin >= -200.
def build_bolder_shorter_dochada():
    # Render uni0E0E at high resolution image, dilate stroke, compress bottom tail, and vectorize
    from PIL import Image, ImageFont, ImageDraw
    f = ImageFont.truetype(str(LEELAWADEE_PATH), 400)
    # Render with offset
    im = Image.new('L', (600, 700), 0)
    draw = ImageDraw.Draw(im)
    draw.text((100, 100), 'ฎ', fill=255, font=f)
    arr = np.array(im)
    
    # Measure baseline: where is bottom of main body vs tail
    # In Leelawadee at 400pt, cap height ~200px
    # Embolden stroke with dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(arr, kernel)
    
    # Crop tight
    ys, xs = np.where(dilated > 100)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    cropped = dilated[y1:y2+1, x1:x2+1]
    
    # Vertically shorten: compress lower 35% of the glyph vertically by 0.55
    ch, cw = cropped.shape
    split_y = int(ch * 0.65)
    top_part = cropped[:split_y, :]
    bot_part = cropped[split_y:, :]
    bot_compressed = cv2.resize(bot_part, (cw, int(bot_part.shape[0] * 0.55)), interpolation=cv2.INTER_LANCZOS4)
    shortened = np.vstack([top_part, bot_compressed])
    
    # Now convert shortened binary mask into TrueType contours
    _, binary = cv2.threshold(shortened, 128, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # Scale to 2048 EM box
    # Target height: cap height (1309) + tail (220) = ~1529
    target_total_h = 1520
    sh_h, sh_w = shortened.shape
    scale_y = target_total_h / float(sh_h)
    scale_x = scale_y * 0.90 # slightly condensed
    
    # y base: yMax = 1309, yMin = 1309 - target_total_h = -211
    tpen = TTGlyphPen(glyf_rbz)
    for i, cnt in enumerate(contours):
        pts = cnt.squeeze()
        if len(pts) < 3: continue
        # smooth slightly
        pts = [((p[0] * scale_x + 60), (1309 - p[1] * scale_y)) for p in pts]
        tpen.moveTo(pts[0])
        for p in pts[1:]:
            tpen.lineTo(p)
        tpen.closePath()
        
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf_rbz)
    glyf_rbz['dochadathai'] = new_g
    hmtx_rbz['dochadathai'] = (int(round(sh_w * scale_x + 120)), 60)
    print(f"Built Bolder Shorter ฎ: bounds=({new_g.xMin}, {new_g.yMin}, {new_g.xMax}, {new_g.yMax}), adv={hmtx_rbz['dochadathai'][0]}")

build_bolder_shorter_dochada()

# 2. FIX ณ (nonenthai): Choose New Reference & Condense Width
# In Leelawadee: uni0E13 has authentic shape. We condense it horizontally to aspect ratio 0.82
def build_condensed_nonen():
    g_src = glyf_leela['uni0E13']
    pen = RecordingPen()
    g_src.draw(pen, glyf_leela)
    
    # Scale x to make aspect ratio ~0.82
    sx = scale_f * 0.70  # condensed width ~950 units
    sy = scale_f * 1.00  # full cap height ~1309 units
    shift_x = 50
    
    tpen = TTGlyphPen(glyf_rbz)
    for cmd, args in pen.value:
        if cmd == 'moveTo':
            tpen.moveTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'lineTo':
            tpen.lineTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'qCurveTo':
            new_pts = [(p[0] * sx + shift_x, p[1] * sy) for p in args]
            tpen.qCurveTo(*new_pts)
        elif cmd == 'closePath':
            tpen.closePath()
            
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf_rbz)
    gname = f_rbz.getBestCmap()[ord('ณ')]
    glyf_rbz[gname] = new_g
    new_adv = int(round(new_g.xMax + 60))
    hmtx_rbz[gname] = (new_adv, 50)
    w = new_g.xMax - new_g.xMin
    h = new_g.yMax - new_g.yMin
    print(f"Built Condensed ณ ({gname}): bounds=({new_g.xMin}, {new_g.yMin}, {new_g.xMax}, {new_g.yMax}), w={w}, h={h}, aspect={w/h:.3f}, adv={new_adv}")

build_condensed_nonen()

# 3. FIX ฒ (thophuthaothai): Condense Width to Aspect Ratio ~0.85
def build_condensed_thophuthao():
    g_src = glyf_leela['uni0E12']
    pen = RecordingPen()
    g_src.draw(pen, glyf_leela)
    
    sx = scale_f * 0.64  # condensed width ~1020 units
    sy = scale_f * 1.00  # full cap height ~1309 units
    shift_x = 50
    
    tpen = TTGlyphPen(glyf_rbz)
    for cmd, args in pen.value:
        if cmd == 'moveTo':
            tpen.moveTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'lineTo':
            tpen.lineTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'qCurveTo':
            new_pts = [(p[0] * sx + shift_x, p[1] * sy) for p in args]
            tpen.qCurveTo(*new_pts)
        elif cmd == 'closePath':
            tpen.closePath()
            
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf_rbz)
    gname = f_rbz.getBestCmap()[ord('ฒ')]
    glyf_rbz[gname] = new_g
    new_adv = int(round(new_g.xMax + 60))
    hmtx_rbz[gname] = (new_adv, 50)
    w = new_g.xMax - new_g.xMin
    h = new_g.yMax - new_g.yMin
    print(f"Built Condensed ฒ ({gname}): bounds=({new_g.xMin}, {new_g.yMin}, {new_g.xMax}, {new_g.yMax}), w={w}, h={h}, aspect={w/h:.3f}, adv={new_adv}")

build_condensed_thophuthao()

# 4. FIX ร (roruathai): Solid, Clean Authentic License Plate Shape
def build_clean_rorua():
    g_src = glyf_leela['uni0E23']
    pen = RecordingPen()
    g_src.draw(pen, glyf_leela)
    
    sx = scale_f * 0.95
    sy = scale_f * 1.00
    shift_x = 40
    
    tpen = TTGlyphPen(glyf_rbz)
    for cmd, args in pen.value:
        if cmd == 'moveTo':
            tpen.moveTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'lineTo':
            tpen.lineTo((args[0][0] * sx + shift_x, args[0][1] * sy))
        elif cmd == 'qCurveTo':
            new_pts = [(p[0] * sx + shift_x, p[1] * sy) for p in args]
            tpen.qCurveTo(*new_pts)
        elif cmd == 'closePath':
            tpen.closePath()
            
    new_g = tpen.glyph()
    new_g.recalcBounds(glyf_rbz)
    gname = f_rbz.getBestCmap()[ord('ร')]
    glyf_rbz[gname] = new_g
    new_adv = int(round(new_g.xMax + 50))
    hmtx_rbz[gname] = (new_adv, 40)
    w = new_g.xMax - new_g.xMin
    h = new_g.yMax - new_g.yMin
    print(f"Built Clean ร ({gname}): bounds=({new_g.xMin}, {new_g.yMin}, {new_g.xMax}, {new_g.yMax}), w={w}, h={h}, aspect={w/h:.3f}, adv={new_adv}")

build_clean_rorua()

f_rbz.save("test_perfect_glyphs.ttf")
print("Saved test_perfect_glyphs.ttf")
