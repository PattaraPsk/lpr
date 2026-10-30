import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import json
import re
import cv2
import numpy as np

session_file = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_file, 'r', encoding='utf-8') as f:
    session = json.load(f)

items = session.get('items', {})
mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')

# DLT Consonant Sets
# WhiteGreen: Pickups, small trucks (บ, ผ, ย, ร, ล, ป, ณ, ต, ถ, ฒ)
PICKUP_CHARS = set("บผยรลปณตถฒ")
# WhiteBlue: Vans, microbuses (น, ฬ, อ, ฮ)
VAN_CHARS = set("นฬอฮ")
# YellowBlack: Taxis, commercial passenger (ท, ม)
TAXI_CHARS = set("ทม")
# WhiteBlackCar: Private passenger <= 7 seats (ก, ข, ค, ฆ, ง, จ, ฉ, ช, ซ, ฌ, ญ, ฎ, ฐ, ฑ, ธ, พ, ฟ, ภ, ว, ศ, ษ, ส)
CAR_CHARS = set("กขคฆงจฉชซฌญฎฐฑธพฟภวศษส")

def classify_dlt(d3: str, aspect_ratio: float, img: np.ndarray = None) -> tuple:
    """
    Returns (predicted_type, reason)
    """
    if not d3 or d3 == '-':
        return (None, "No d3")
    
    clean_d3 = d3.strip()
    
    # 1. Numeric series: Land Transport Act (Large Plate)
    if clean_d3.isdigit():
        num_val = int(clean_d3)
        # Commercial series (10-39, 60-79, 700-799) are YellowBlack
        # Private heavy series (40-59, 80-99) are WhiteBlackCar / WhiteHeavy
        if 40 <= num_val <= 59 or 80 <= num_val <= 99:
            return ("WhiteBlackCar", f"Numeric series {clean_d3} (Private Heavy/Large Plate)")
        else:
            return ("YellowBlack", f"Numeric series {clean_d3} (Commercial Heavy/Large Plate)")
            
    # 2. Motorcycle check: aspect ratio < 1.5
    if aspect_ratio < 1.55:
        return ("WhiteBlackMotor", f"Aspect ratio {aspect_ratio:.2f} < 1.55 (Motorcycle)")
        
    # 3. Extract the Thai consonant pair from d3
    # Handles: 'กข' (2 letters), '1กข' (1 digit + 2 letters), 'กกบ' (vanity/auction)
    # Remove leading digits if any
    letters_only = re.sub(r'^[0-9]+', '', clean_d3)
    if not letters_only:
        return (None, "No letters")
        
    first_char = letters_only[0]
    
    # DLT Rule 1: Taxi (ท, ม)
    if first_char in TAXI_CHARS:
        return ("YellowBlack", f"First consonant '{first_char}' is Taxi (YellowBlack)")
        
    # DLT Rule 2: Van (น, ฬ, อ, ฮ)
    if first_char in VAN_CHARS:
        return ("WhiteBlue", f"First consonant '{first_char}' is Van (WhiteBlue)")
        
    # DLT Rule 3: Pickup / Truck (บ, ผ, ย, ร, ล, ป, ณ, ต, ถ, ฒ)
    if first_char in PICKUP_CHARS:
        return ("WhiteGreen", f"First consonant '{first_char}' is Pickup (WhiteGreen)")
        
    # DLT Rule 4: Passenger Car (ก, ข, จ, ฉ, ช, ฌ, ญ, ฎ, ฐ, ธ, พ, ภ, ว, ศ, ษ, ส, etc.)
    if first_char in CAR_CHARS:
        return ("WhiteBlackCar", f"First consonant '{first_char}' is Private Car (WhiteBlackCar)")
        
    return ("WhiteBlackCar", f"Default fallback for '{first_char}'")

print(f"{'Filename':<45} | {'d3':<5} | {'Curr Plate Type':<16} | {'New DLT Plate Type':<18} | {'Reason'}")
print("-" * 115)

changes = []
for k, v in items.items():
    d3 = v.get('d3', '-')
    curr_pt = v.get('plate_type', '-')
    p = mobile_dir / k
    aspect = 2.2
    img = None
    if p.exists():
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            aspect = img.shape[1] / float(img.shape[0])
            
    # Also check if d3 is '-' but filename has d3
    eff_d3 = d3
    if eff_d3 == '-':
        parts = k.split('_')
        # format: Screenshot ..._PlateType_d3_d4_prov.png or cur_..._PlateType_d3_d4_prov.jpg
        if len(parts) >= 4 and parts[-3] != '-':
            eff_d3 = parts[-3]
            
    predicted_pt, reason = classify_dlt(eff_d3, aspect, img)
    
    if predicted_pt and predicted_pt != curr_pt:
        changes.append((k, curr_pt, predicted_pt, eff_d3, reason))
        print(f"{k:<45} | {eff_d3:<5} | {curr_pt:<16} -> {predicted_pt:<18} | {reason}")

print(f"\nTotal potential changes: {len(changes)}")
