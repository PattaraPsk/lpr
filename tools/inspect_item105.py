import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
import numpy as np

mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')
fn = "Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png"
p = mobile_dir / fn
img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
print("Image shape:", img.shape)

# Let's upscale and save to artifact directory
art_dir = Path(r'C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b')
cv2.imwrite(str(art_dir / "inspect_105_กกบ.png"), cv2.resize(img, (img.shape[1]*4, img.shape[0]*4), interpolation=cv2.INTER_NEAREST))
print("Saved inspect_105_กกบ.png")
