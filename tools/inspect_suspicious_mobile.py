import sys
sys.stdout.reconfigure(encoding='utf-8')
import cv2
import numpy as np
from pathlib import Path
import json

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})
mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')

suspicious = [
    'cur_AD04_-_กฮ_4332_กรุงเทพมหานคร.jpg',
    'cur_AD15_-_ฎฌ_34_กรุงเทพมหานคร.jpg',
    'cur_AD19_-_5ขร_150_กรุงเทพมหานคร.jpg',
    'cur_AE03_-_ชล_299_กรุงเทพมหานคร.jpg',
    'cur_AE04_-_บพ_4567_นนทบุรี.jpg',
    'cur_AF45_-_2กด_9696_กรุงเทพมหานคร.jpg',
    'cur_AF69_-_ผม_2222_กรุงเทพมหานคร.jpg',
    'cur_AK98_-_4ขบ_6000_กรุงเทพมหานคร.jpg',
    'cur_AL00_-_4ขบ_4000_กรุงเทพมหานคร.jpg',
    'cur_AU65_-_5กธ_2347_กรุงเทพมหานคร.jpg',
    'cur_AH64_WhiteGreen_2ฆฎ_9898_กรุงเทพมหานคร.jpg',
    'Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163007_WhiteBlackCar_14_3112_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163017_YellowBlack_701_6466_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163123_WhiteGreen_3ฒล_7019_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163125_WhiteBlackCar_3ฒฐ_795_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163127_WhiteBlackCar_4ฒฌ_5525_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163225_WhiteBlackCar_72_1135_นครปฐม.png',
    'Screenshot 2026-10-27 163235_WhiteBlackCar_72_1135_นครปฐม.png',
    'Screenshot 2026-10-27 163246_WhiteBlackCar_3ฒษ_3543_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163253_WhiteBlackCar_52_8343_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163303_WhiteBlackCar_ผก_6290_พระนครศรีอยุธยา.png',
]

for name in suspicious:
    p = mobile_dir / name
    if not p.exists():
        # Find partial
        matches = list(mobile_dir.glob(f"*{name[:15]}*"))
        if matches:
            p = matches[0]
    
    if p.exists():
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        # Calculate mean color in BGR
        mean_bgr = img.mean(axis=(0, 1))
        # Center crop mean
        ch, cw = h // 4, w // 4
        center_crop = img[ch:h-ch, cw:w-cw]
        center_mean = center_crop.mean(axis=(0, 1))
        
        entry = items.get(p.name, {})
        d3 = entry.get('d3', '-')
        pt = entry.get('plate_type', '-')
        print(f"{p.name[:50]:<50} | d3: {d3:<4} | curr_pt: {pt:<14} | size: {w}x{h} ({w/h:.2f}) | center BGR: {center_mean.astype(int)}")
