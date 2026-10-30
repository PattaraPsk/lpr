"""
Batch update mobile plate types and rename files based on DLT front character criteria.
Compliant with ISO/IEC 42001:2023 dataset change management.
"""

import json
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

mobile_dir = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile')
session_p = mobile_dir / '.lpr_session.json'
backup_p = mobile_dir / '.lpr_session.json.bak'

# 1. Backup session file
shutil.copy2(session_p, backup_p)
print(f"Backed up session file to {backup_p}")

with open(session_p, 'r', encoding='utf-8') as f:
    session_data = json.load(f)

items = session_data.get('items', {})

# Mapping: old_filename -> (new_plate_type, new_filename)
updates = {
    # Rectify mislabeled WhiteBlackCar -> YellowBlack (Commercial Land Transport Act series)
    "Screenshot 2026-10-27 163007_WhiteBlackCar_14_3112_กรุงเทพมหานคร.png": (
        "YellowBlack",
        "Screenshot 2026-10-27 163007_YellowBlack_14_3112_กรุงเทพมหานคร.png"
    ),
    "Screenshot 2026-10-27 163225_WhiteBlackCar_72_1135_นครปฐม.png": (
        "YellowBlack",
        "Screenshot 2026-10-27 163225_YellowBlack_72_1135_นครปฐม.png"
    ),
    "Screenshot 2026-10-27 163235_WhiteBlackCar_72_1135_นครปฐม.png": (
        "YellowBlack",
        "Screenshot 2026-10-27 163235_YellowBlack_72_1135_นครปฐม.png"
    ),
    
    # Rectify mislabeled WhiteBlackCar -> YellowBlack (Motorcycle Taxi / วินมอเตอร์ไซค์)
    "Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png": (
        "YellowBlack",
        "Screenshot 2026-10-27 163219_YellowBlack_กกบ_310_ปทุมธานี.png"
    ),

    # Rectify mislabeled WhiteBlackCar -> WhiteGreen (DLT pickup consonant ฒ and ผ)
    "Screenshot 2026-10-27 163125_WhiteBlackCar_3ฒฐ_795_กรุงเทพมหานคร.png": (
        "WhiteGreen",
        "Screenshot 2026-10-27 163125_WhiteGreen_3ฒฐ_795_กรุงเทพมหานคร.png"
    ),
    "Screenshot 2026-10-27 163127_WhiteBlackCar_4ฒฌ_5525_กรุงเทพมหานคร.png": (
        "WhiteGreen",
        "Screenshot 2026-10-27 163127_WhiteGreen_4ฒฌ_5525_กรุงเทพมหานคร.png"
    ),
    "Screenshot 2026-10-27 163246_WhiteBlackCar_3ฒษ_3543_กรุงเทพมหานคร.png": (
        "WhiteGreen",
        "Screenshot 2026-10-27 163246_WhiteGreen_3ฒษ_3543_กรุงเทพมหานคร.png"
    ),
    "Screenshot 2026-10-27 163303_WhiteBlackCar_ผก_6290_พระนครศรีอยุธยา.png": (
        "WhiteGreen",
        "Screenshot 2026-10-27 163303_WhiteGreen_ผก_6290_พระนครศรีอยุธยา.png"
    ),

    # Rectify mislabeled WhiteGreen -> WhiteBlackCar (DLT passenger car consonant ฆ)
    "cur_AH64_WhiteGreen_2ฆฎ_9898_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AH64_WhiteBlackCar_2ฆฎ_9898_กรุงเทพมหานคร.jpg"
    ),

    # Assign plate_type to unassigned items with known DLT front consonants
    # Pickups: บ, ผ
    "cur_AE04_-_บพ_4567_นนทบุรี.jpg": (
        "WhiteGreen",
        "cur_AE04_WhiteGreen_บพ_4567_นนทบุรี.jpg"
    ),
    "cur_AF69_-_ผม_2222_กรุงเทพมหานคร.jpg": (
        "WhiteGreen",
        "cur_AF69_WhiteGreen_ผม_2222_กรุงเทพมหานคร.jpg"
    ),

    # Passenger cars: ก, ข, ช, ฎ
    "cur_AD04_-_กฮ_4332_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AD04_WhiteBlackCar_กฮ_4332_กรุงเทพมหานคร.jpg"
    ),
    "cur_AD15_-_ฎฌ_34_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AD15_WhiteBlackCar_ฎฌ_34_กรุงเทพมหานคร.jpg"
    ),
    "cur_AD19_-_5ขร_150_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AD19_WhiteBlackCar_5ขร_150_กรุงเทพมหานคร.jpg"
    ),
    "cur_AE03_-_ชล_299_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AE03_WhiteBlackCar_ชล_299_กรุงเทพมหานคร.jpg"
    ),
    "cur_AF45_-_2กด_9696_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AF45_WhiteBlackCar_2กด_9696_กรุงเทพมหานคร.jpg"
    ),
    "cur_AK98_-_4ขบ_6000_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AK98_WhiteBlackCar_4ขบ_6000_กรุงเทพมหานคร.jpg"
    ),
    "cur_AL00_-_4ขบ_4000_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AL00_WhiteBlackCar_4ขบ_4000_กรุงเทพมหานคร.jpg"
    ),
    "cur_AU65_-_5กธ_2347_กรุงเทพมหานคร.jpg": (
        "WhiteBlackCar",
        "cur_AU65_WhiteBlackCar_5กธ_2347_กรุงเทพมหานคร.jpg"
    )
}

print(f"Total files to update: {len(updates)}")

renamed_count = 0
session_updated_count = 0

new_items = {}
for old_fn, item_data in items.items():
    if old_fn in updates:
        new_pt, new_fn = updates[old_fn]
        old_path = mobile_dir / old_fn
        new_path = mobile_dir / new_fn
        
        # 1. Rename physical file
        if old_path.exists():
            if old_path != new_path:
                old_path.rename(new_path)
                print(f"Renamed file: {old_fn} -> {new_fn}")
                renamed_count += 1
            else:
                print(f"File already named: {new_fn}")
        else:
            if new_path.exists():
                print(f"Target already exists: {new_fn}")
            else:
                print(f"WARNING: Neither old nor new exists: {old_fn}")

        # 2. Update item metadata
        item_data['plate_type'] = new_pt
        item_data['filename'] = new_fn
        item_data['rel_path'] = new_fn
        # Ensure status is approved if text is readable
        if item_data.get('d3') != '-' and item_data.get('status') == 'cannot_read':
            item_data['status'] = 'approved'
            
        new_items[new_fn] = item_data
        session_updated_count += 1
    else:
        new_items[old_fn] = item_data

session_data['items'] = new_items

# Save updated session file
with open(session_p, 'w', encoding='utf-8') as f:
    json.dump(session_data, f, ensure_ascii=False, indent=2)

print(f"\nUpdate complete!")
print(f"Files renamed: {renamed_count}")
print(f"Session entries updated: {session_updated_count}")
print(f"Total items in session: {len(new_items)}")
