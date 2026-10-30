import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import re

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})

# DLT Rules:
# Passenger Car (<= 7 seats) -> WhiteBlackCar:
#   2-letter: ก, ข, ค, ฆ, ง, จ, ฉ, ช, ซ, ฌ, ญ, ฎ, ฐ, ฑ, ธ, พ, ฟ, ภ, ว, ศ, ษ, ส
#   1-digit + 2-letter: 1กข - 9ฮฮ (for passenger cars)
# Van / Microbus (> 7 seats) -> WhiteBlue:
#   First consonant is strictly: น, ฬ, อ, ฮ (e.g. นข, นง, 1นข, ฬอ, อร, ฮข)
# Private Pickup / Truck (<= 2,200 kg) -> WhiteGreen:
#   First consonant is strictly: บ, ผ, ย, ร, ล, ป, ณ, ต, ถ, ฒ (e.g. บพ, ผม, ผก, 1บข, 3ฒล, 3ฒฐ, 4ฒฌ, 3ฒษ)
# Taxi (<= 7 seats) -> YellowBlack:
#   First consonant is strictly: ท (taxi), ม (service car/airport taxi) (e.g. ทฬ, 1มข)
# Motorcycle -> WhiteBlackMotor:
#   3-line layout, aspect ratio ~ 1.28
# Large Plate / Heavy Transport (Land Transport Act) -> YellowBlack (commercial) or WhiteBlackCar (private heavy):
#   Leading numbers like 10-39, 70-79, 701-799, 14, 72, etc.

for idx, (k, v) in enumerate(items.items(), 1):
    d3 = v.get('d3', '-')
    d4 = v.get('d4', '-')
    prov = v.get('province', '-')
    curr_pt = v.get('plate_type', '-')
    status = v.get('status', '-')
    
    if d3 == '-' and curr_pt == '-':
        continue
        
    print(f"[{idx:03d}] {k}")
    print(f"      d3: '{d3}', d4: '{d4}', prov: '{prov}', curr_pt: '{curr_pt}'")
