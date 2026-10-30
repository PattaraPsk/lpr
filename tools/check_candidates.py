import json
import sys
import re
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

session_p = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_p, 'r', encoding='utf-8') as f:
    session = json.load(f)
items = session.get('items', {})

PICKUP_CHARS = set('บผยรลปณตถฒ')
VAN_CHARS = set('นฬอฮ')
TAXI_CHARS = set('ทม')
CAR_CHARS = set('กขคฆงจฉชซฌญฎฐฑธพฟภวศษส')

candidates_for_update = []

for fn, data in sorted(items.items()):
    curr_pt = data.get('plate_type', '-')
    d3 = data.get('d3', '-')
    d4 = data.get('d4', '-')
    prov = data.get('province', '-')
    status = data.get('status', '-')
    
    # Try getting d3 from filename if session has '-'
    eff_d3 = d3
    if eff_d3 == '-':
        parts = fn.rsplit('.', 1)[0].split('_')
        if len(parts) >= 4 and parts[-3] != '-':
            eff_d3 = parts[-3]
            
    # Classify based on DLT front character criteria
    dlt_type = None
    reason = None
    
    if eff_d3 != '-':
        clean_d3 = eff_d3.strip()
        if clean_d3.isdigit():
            num = int(clean_d3)
            if 40 <= num <= 59 or 80 <= num <= 99:
                # White/Black heavy
                dlt_type = 'WhiteBlackCar'
                reason = f'Numeric series {clean_d3} (Private Heavy - White/Black)'
            else:
                # Commercial heavy / yellow
                dlt_type = 'YellowBlack'
                reason = f'Numeric series {clean_d3} (Commercial Heavy - Yellow/Black)'
        else:
            letters = re.sub(r'^[0-9]+', '', clean_d3)
            if letters:
                c1 = letters[0]
                if c1 in PICKUP_CHARS:
                    dlt_type = 'WhiteGreen'
                    reason = f'DLT pickup consonant "{c1}"'
                elif c1 in VAN_CHARS:
                    dlt_type = 'WhiteBlue'
                    reason = f'DLT van consonant "{c1}"'
                elif c1 in TAXI_CHARS:
                    dlt_type = 'YellowBlack'
                    reason = f'DLT taxi consonant "{c1}"'
                elif c1 in CAR_CHARS:
                    dlt_type = 'WhiteBlackCar'
                    reason = f'DLT passenger car consonant "{c1}"'
                    
    # Special cases:
    # 1) Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png:
    # Plate is physically yellow with black letters, currently YellowBlack -> keep YellowBlack
    if fn == 'Screenshot 2026-10-27 162926_YellowBlack_2ขร_547_กรุงเทพมหานคร.png':
        dlt_type = 'YellowBlack'
        reason = 'Public yellow plate (keep YellowBlack)'
        
    # 2) Motorcycle taxi (วินมอเตอร์ไซค์)
    if fn == 'Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png':
        dlt_type = 'YellowBlack'
        reason = 'Motorcycle taxi (วินมอเตอร์ไซค์) - Yellow plate'

    if dlt_type and dlt_type != curr_pt:
        candidates_for_update.append((fn, curr_pt, dlt_type, eff_d3, d4, prov, reason))

print(f'Total candidates for update: {len(candidates_for_update)}')
for item in candidates_for_update:
    print(f'{item[0]}')
    print(f'   Current: {item[1]} -> Proposed: {item[2]} (d3: {item[3]}, d4: {item[4]}, prov: {item[5]}) | {item[6]}')
