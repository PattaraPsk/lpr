import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2
import numpy as np
import json

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})
mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')
art_dir = Path(r'C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b\mobile_inspect')
art_dir.mkdir(parents=True, exist_ok=True)

for idx, (k, v) in enumerate(items.items(), 1):
    d3 = v.get('d3', '-')
    pt = v.get('plate_type', '-')
    p = mobile_dir / k
    if p.exists() and (d3 != '-' or pt != '-'):
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            # save to art_dir with index
            safe_k = f"{idx:03d}_{k}"
            cv2.imencode('.png', img)[1].tofile(str(art_dir / safe_k))

print(f"Exported images to {art_dir}")
