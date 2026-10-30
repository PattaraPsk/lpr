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
art_dir = Path(r'C:\Users\WS4090\.gemini\antigravity\brain\a97c3b33-fd86-4034-a27c-d9b63ea4a13b')

# Let's create a visual contact sheet of all 69 labeled items
cards = []
target_w = 200
target_h = 100

for k, v in items.items():
    d3 = v.get('d3', '-')
    d4 = v.get('d4', '-')
    prov = v.get('province', '-')
    pt = v.get('plate_type', '-')
    status = v.get('status', '-')
    
    if d3 == '-' and pt == '-':
        continue
        
    p = mobile_dir / k
    if not p.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
        
    h, w = img.shape[:2]
    # Resize preserving aspect ratio or padded
    aspect = w / float(h)
    resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # Create card with label text
    card = np.zeros((target_h + 50, target_w, 3), dtype=np.uint8)
    card[:target_h, :target_w] = resized
    
    # Write text info
    cv2.putText(card, f"PT: {pt}", (5, target_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(card, f"d3:{d3} d4:{d4}", (5, target_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)
    cv2.putText(card, f"ar:{aspect:.2f}", (5, target_h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
    cards.append((k, card, pt, d3, aspect))

print(f"Total cards created: {len(cards)}")

# Arrange in a grid: 5 columns
cols = 5
rows = (len(cards) + cols - 1) // cols
grid_w = cols * (target_w + 10) + 10
grid_h = rows * (target_h + 60) + 10

sheet = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
sheet[:] = (30, 30, 30)

for idx, (k, card, pt, d3, aspect) in enumerate(cards):
    r = idx // cols
    c = idx % cols
    x = 10 + c * (target_w + 10)
    y = 10 + r * (target_h + 60)
    sheet[y:y+target_h+50, x:x+target_w] = card

contact_path = art_dir / "mobile_labels_contact_sheet.jpg"
cv2.imwrite(str(contact_path), sheet)
print(f"Saved contact sheet to {contact_path}")
