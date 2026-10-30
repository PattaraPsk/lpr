import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
import numpy as np

mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')

files_to_check = [
    "cur_AH64_WhiteGreen_2ฆฎ_9898_กรุงเทพมหานคร.jpg",
    "cur_AU65_-_5กธ_2347_กรุงเทพมหานคร.jpg",
    "Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163007_WhiteBlackCar_14_3112_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163017_YellowBlack_701_6466_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163138_WhiteBlackCar_5กฉ_8934_กรุงเทพมหานคร.png",
    "Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png",
    "Screenshot 2026-10-27 163225_WhiteBlackCar_72_1135_นครปฐม.png",
    "Screenshot 2026-10-27 163253_WhiteBlackCar_52_8343_กรุงเทพมหานคร.png",
]

for fn in files_to_check:
    p = mobile_dir / fn
    if not p.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    # mean colors
    b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    print(f"File: {fn}")
    print(f"  Shape: {w}x{h} (ratio: {w/h:.2f}) | Mean BGR: ({b:.1f}, {g:.1f}, {r:.1f})")
