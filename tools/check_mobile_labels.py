import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})
print(f"Total items: {len(items)}")

for idx, (k, v) in enumerate(items.items(), 1):
    pt = v.get('plate_type')
    d3 = v.get('d3')
    d4 = v.get('d4')
    prov = v.get('province')
    status = v.get('status')
    if d3 != '-' or pt != '-':
        print(f"[{idx:03d}] {k}")
        print(f"      plate_type: '{pt}' | d3: '{d3}' | d4: '{d4}' | prov: '{prov}' | status: '{status}'")
