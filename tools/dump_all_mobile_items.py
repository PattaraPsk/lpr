import sys
from pathlib import Path
import json

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})

out_path = Path('mobile_items_dump.txt')
with open(out_path, 'w', encoding='utf-8') as out:
    out.write(f"Total items in session: {len(items)}\n\n")
    for idx, (k, v) in enumerate(items.items(), 1):
        out.write(f"[{idx:03d}] {k}\n")
        out.write(f"  plate_type: {v.get('plate_type')}\n")
        out.write(f"  d3: {v.get('d3')}\n")
        out.write(f"  d4: {v.get('d4')}\n")
        out.write(f"  province: {v.get('province')}\n")
        out.write(f"  status: {v.get('status')}\n")
        out.write(f"  ai_d3: {v.get('ai_d3')}\n")
        out.write(f"  ai_d4: {v.get('ai_d4')}\n")
        out.write(f"  ai_province: {v.get('ai_province')}\n")
        out.write(f"  ai_confidence: {v.get('ai_confidence')}\n\n")

print("Dumped successfully to mobile_items_dump.txt")
