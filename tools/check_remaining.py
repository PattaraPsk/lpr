import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

session_p = Path(r'D:\Projects-anti\lpr\dataset\snapshot\mobile\.lpr_session.json')
with open(session_p, 'r', encoding='utf-8') as f:
    session = json.load(f)
items = session.get('items', {})

updated_keys = {
    'Screenshot 2026-10-27 163007_WhiteBlackCar_14_3112_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163125_WhiteBlackCar_3ฒฐ_795_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163127_WhiteBlackCar_4ฒฌ_5525_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163219_WhiteBlackCar_กกบ_310_ปทุมธานี.png',
    'Screenshot 2026-10-27 163225_WhiteBlackCar_72_1135_นครปฐม.png',
    'Screenshot 2026-10-27 163235_WhiteBlackCar_72_1135_นครปฐม.png',
    'Screenshot 2026-10-27 163246_WhiteBlackCar_3ฒษ_3543_กรุงเทพมหานคร.png',
    'Screenshot 2026-10-27 163303_WhiteBlackCar_ผก_6290_พระนครศรีอยุธยา.png',
    'cur_AD04_-_กฮ_4332_กรุงเทพมหานคร.jpg',
    'cur_AD15_-_ฎฌ_34_กรุงเทพมหานคร.jpg',
    'cur_AD19_-_5ขร_150_กรุงเทพมหานคร.jpg',
    'cur_AE03_-_ชล_299_กรุงเทพมหานคร.jpg',
    'cur_AE04_-_บพ_4567_นนทบุรี.jpg',
    'cur_AF45_-_2กด_9696_กรุงเทพมหานคร.jpg',
    'cur_AF69_-_ผม_2222_กรุงเทพมหานคร.jpg',
    'cur_AH64_WhiteGreen_2ฆฎ_9898_กรุงเทพมหานคร.jpg',
    'cur_AK98_-_4ขบ_6000_กรุงเทพมหานคร.jpg',
    'cur_AL00_-_4ขบ_4000_กรุงเทพมหานคร.jpg',
    'cur_AU65_-_5กธ_2347_กรุงเทพมหานคร.jpg'
}

print(f"Remaining items with labels:")
for k, v in sorted(items.items()):
    if k in updated_keys:
        continue
    pt = v.get('plate_type', '-')
    d3 = v.get('d3', '-')
    d4 = v.get('d4', '-')
    prov = v.get('province', '-')
    if pt != '-' or d3 != '-':
        print(f"{k:<55} | pt={pt:<15} | d3={d3:<6} | d4={d4:<6} | prov={prov:<15}")
