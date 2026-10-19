"""
Generate mock test dataset for Thai LPR annotator testing.
Creates a folder named 'sample_images' with structure:
{name}_{plate_type}_{d3}_{d4}_{province}.jpg
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def generate_samples():
    sample_dir = Path(__file__).parent / "sample_images"
    sample_dir.mkdir(exist_ok=True)

    
    # Samples with (filename, plate_text, prov_text, plate_type_desc, plate_bg, plate_text_color)
    samples = [
        # 1. White Background with Black Letters: Private passenger cars (<= 7 seats) / motorcycles
        (
            "car01_white_black_1กก_1234_กรุงเทพมหานคร.jpg",
            "1กก 1234", "กรุงเทพมหานคร",
            "White/Black (Car <= 7 seats)",
            (255, 255, 255), (0, 0, 0)
        ),
        # 2. White Background with Blue Letters: Vans, microbuses (> 7 seats)
        (
            "van02_white_blue_2ขค_5678_เชียงใหม่.jpg",
            "2ขค 5678", "เชียงใหม่",
            "White/Blue (Van > 7 seats)",
            (255, 255, 255), (20, 60, 220)
        ),
        # 3. White Background with Green Letters: Private pickup trucks
        (
            "pickup03_white_green_1ผผ_7890_ชลบุรี.jpg",
            "1ผผ 7890", "ชลบุรี",
            "White/Green (Pickup truck)",
            (255, 255, 255), (20, 140, 40)
        ),
        # 4. Yellow Background with Black Letters: Taxis, commercial hire
        (
            "taxi04_yellow_black_ทส_9999_กรุงเทพมหานคร.jpg",
            "ทส 9999", "กรุงเทพมหานคร",
            "Yellow/Black (Taxi)",
            (250, 204, 21), (0, 0, 0)
        ),
        # 5. Yellow Background with Blue Letters: Fixed-route local public buses / songthaews
        (
            "bus05_yellow_blue_10_4567_ขอนแก่น.jpg",
            "10-4567", "ขอนแก่น",
            "Yellow/Blue (Songthaew/Bus)",
            (250, 204, 21), (20, 60, 220)
        ),
        # 6. Yellow Background with Green Letters: Public tuk-tuks
        (
            "tuktuk06_yellow_green_สกร_123_ภูเก็ต.jpg",
            "สกร 123", "ภูเก็ต",
            "Yellow/Green (Tuk-Tuk)",
            (250, 204, 21), (20, 140, 40)
        ),
        # 7. Green Background with White Letters: Limousine / hotel / tour service
        (
            "limo07_green_white_ณข_8888_สุราษฎร์ธานี.jpg",
            "ณข 8888", "สุราษฎร์ธานี",
            "Green/White (Limo/Tour)",
            (20, 140, 40), (255, 255, 255)
        ),
        # 8. Red Background with Black Letters: Temporary / dealer plate
        (
            "newcar08_red_black_ก_0001_กรุงเทพมหานคร.jpg",
            "ก-0001", "กรุงเทพมหานคร",
            "Red/Black (Temporary/Dealer)",
            (220, 38, 38), (0, 0, 0)
        ),
        # 9. Orange Background with Black Letters: Agricultural vehicles, tractors
        (
            "tractor09_orange_black_ตค_3333_นครราชสีมา.jpg",
            "ตค 3333", "นครราชสีมา",
            "Orange/Black (Tractor/Agricultural)",
            (234, 88, 12), (0, 0, 0)
        ),
        # Multi-underscore name
        (
            "camera_gate_in_car10_white_black_5กภ_6789_นนทบุรี.jpg",
            "5กภ 6789", "นนทบุรี",
            "White/Black (Gate Camera In)",
            (255, 255, 255), (0, 0, 0)
        ),
        # Pre-labeled Not Plate
        (
            "obstacle11_+_+_+_+.jpg",
            "NOT A LICENSE PLATE", "+",
            "Not Plate",
            (200, 200, 200), (80, 80, 80)
        ),
        # Pre-labeled Cannot Read
        (
            "blurred12_-_-_-_-.jpg",
            "BLURRED / UNREADABLE", "-",
            "Cannot Read",
            (160, 160, 160), (40, 40, 40)
        ),

    ]
    
    for filename, plate_text, prov_text, desc, plate_bg, plate_fg in samples:
        img = Image.new("RGB", (640, 360), color=(235, 238, 242))
        draw = ImageDraw.Draw(img)
        
        # Outer border
        draw.rectangle([70, 70, 570, 290], outline=(50, 50, 60), width=3, fill=plate_bg)
        # Inner plate line
        draw.rectangle([80, 80, 560, 280], outline=plate_fg, width=2)
        
        # Texts
        draw.text((150, 110), plate_text, fill=plate_fg)
        draw.text((150, 170), prov_text, fill=plate_fg)
        draw.text((150, 230), f"[{desc}]", fill=plate_fg)
        
        out_path = sample_dir / filename
        img.save(out_path, quality=90)
        print(f"Created: {out_path.name}")

if __name__ == "__main__":
    generate_samples()
