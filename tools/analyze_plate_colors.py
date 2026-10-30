import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
import numpy as np
import json

mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')
art_dir = Path(r'C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\inspect_crops')
art_dir.mkdir(parents=True, exist_ok=True)

test_files = [
    "cur_AH64_WhiteGreen_2ฆฎ_9898_กรุงเทพมหานคร.jpg",
    "cur_AU65_-_5กธ_2347_กรุงเทพมหานคร.jpg",
    "Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163007_WhiteBlackCar_14_3112_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163017_YellowBlack_701_6466_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163123_WhiteGreen_3ฒล_7019_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163125_WhiteBlackCar_3ฒฐ_795_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163127_WhiteBlackCar_4ฒฌ_5525_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163138_WhiteBlackCar_5กฉ_8934_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png",
    "Screenshot 2026-10-27 163225_WhiteBlackCar_72_1135_นครปฐม.png",
    "Screenshot 2026-10-27 163235_WhiteBlackCar_72_1135_นครปฐม.png",
    "Screenshot 2026-10-27 163246_WhiteBlackCar_3ฒษ_3543_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163253_WhiteBlackCar_52_8343_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163303_WhiteBlackCar_ผก_6290_พระนครศรีอยุธยา.png",
    "cur_AD04_-_กฮ_4332_กรุงเทพมหานคร.jpg",
    "cur_AE04_-_บพ_4567_นนทบุรี.jpg",
    "cur_AF69_-_ผม_2222_กรุงเทพมหานคร.jpg",
    "cur_AK98_-_4ขบ_6000_กรุงเทพมหานคร.jpg"
]

print("Analyzing pixel colors and lines...")
for fn in test_files:
    p = mobile_dir / fn
    if not p.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w, c = img.shape
    
    # Convert to HSV to detect plate background and ink color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    
    # Check for yellow plate: Hue 15-35, Saturation > 60, Value > 120
    yellow_mask = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 60) & (v_channel > 100)
    yellow_pct = np.mean(yellow_mask) * 100.0
    
    # Check for green ink: Hue 35-85, Saturation > 40
    green_mask = (h_channel >= 35) & (h_channel <= 85) & (s_channel > 40)
    green_pct = np.mean(green_mask) * 100.0
    
    # Check for blue ink: Hue 90-130, Saturation > 40
    blue_mask = (h_channel >= 90) & (h_channel <= 130) & (s_channel > 40)
    blue_pct = np.mean(blue_mask) * 100.0
    
    # Save a resized copy for visual check
    disp = cv2.resize(img, (240, int(240 * h / w))) if w > 0 else img
    cv2.imencode('.png', disp)[1].tofile(str(art_dir / (fn[:30] + ".png")))
    
    print(f"{fn[:35]:<35} | size:{w:4d}x{h:<4d} (ratio:{w/h:.2f}) | YellowBg:{yellow_pct:5.1f}% | GreenInk:{green_pct:5.1f}% | BlueInk:{blue_pct:5.1f}%")
