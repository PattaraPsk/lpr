"""
Generate mock test dataset for Thai LPR annotator testing.
Creates a folder named 'sample_images' with nested subfolders and duplicate files:
- cam_front/
- cam_rear/
- gate_in/
- Root files
- Duplicate '_dup' files for testing removal
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
    
    # Subfolders to create
    subfolders = ["cam_front", "cam_rear", "gate_in"]
    for sf in subfolders:
        (sample_dir / sf).mkdir(exist_ok=True)
    
    # Samples with (subfolder, filename, plate_text, prov_text, plate_type_desc, plate_bg, plate_text_color)
    samples = [
        # Root folder samples
        (
            "",
            "car01_WhiteBlackCar_1กก_1234_กรุงเทพมหานคร.jpg",
            "1กก 1234", "กรุงเทพมหานคร",
            "White/Black - Car (<= 7 seats)",
            (255, 255, 255), (0, 0, 0)
        ),
        (
            "",
            "motor02_WhiteBlackMotor_1กข_999_เชียงใหม่.jpg",
            "1กข 999", "เชียงใหม่",
            "White/Black - Motorcycle",
            (255, 255, 255), (0, 0, 0)
        ),
        (
            "",
            "test_dup_WhiteBlackCar_1กก_1234_กรุงเทพมหานคร.jpg",
            "1กก 1234 (DUP)", "กรุงเทพมหานคร",
            "Duplicate Sample",
            (255, 255, 255), (180, 0, 0)
        ),
        # cam_front subfolder
        (
            "cam_front",
            "van03_WhiteBlue_2ขค_5678_เชียงใหม่.jpg",
            "2ขค 5678", "เชียงใหม่",
            "White/Blue (Van > 7 seats)",
            (255, 255, 255), (20, 60, 220)
        ),
        (
            "cam_front",
            "pickup04_WhiteGreen_1ผผ_7890_ชลบุรี.jpg",
            "1ผผ 7890", "ชลบุรี",
            "White/Green (Pickup truck)",
            (255, 255, 255), (20, 140, 40)
        ),
        (
            "cam_front",
            "pickup04_dup_WhiteGreen_1ผผ_7890_ชลบุรี.jpg",
            "1ผผ 7890 (DUP)", "ชลบุรี",
            "Duplicate Pickup",
            (255, 255, 255), (180, 0, 0)
        ),
        # cam_rear subfolder
        (
            "cam_rear",
            "taxi05_YellowBlack_ทส_9999_กรุงเทพมหานคร.jpg",
            "ทส 9999", "กรุงเทพมหานคร",
            "Yellow/Black (Taxi)",
            (250, 204, 21), (0, 0, 0)
        ),
        (
            "cam_rear",
            "bus06_YellowBlue_10_4567_ขอนแก่น.jpg",
            "10-4567", "ขอนแก่น",
            "Yellow/Blue (Songthaew/Bus)",
            (250, 204, 21), (20, 60, 220)
        ),
        (
            "cam_rear",
            "tuktuk07_YellowGreen_สกร_123_ภูเก็ต.jpg",
            "สกร 123", "ภูเก็ต",
            "Yellow/Green (Tuk-Tuk)",
            (250, 204, 21), (20, 140, 40)
        ),
        # gate_in subfolder
        (
            "gate_in",
            "limo08_GreenWhite_ณข_8888_สุราษฎร์ธานี.jpg",
            "ณข 8888", "สุราษฎร์ธานี",
            "Green/White (Limo/Tour)",
            (20, 140, 40), (255, 255, 255)
        ),
        (
            "gate_in",
            "newcar09_RedBlack_ก_0001_กรุงเทพมหานคร.jpg",
            "ก-0001", "กรุงเทพมหานคร",
            "Red/Black (Temporary/Dealer)",
            (220, 38, 38), (0, 0, 0)
        ),
        (
            "gate_in",
            "tractor10_OrangeBlack_ตค_3333_นครราชสีมา.jpg",
            "ตค 3333", "นครราชสีมา",
            "Orange/Black (Tractor/Agricultural)",
            (234, 88, 12), (0, 0, 0)
        ),
        (
            "gate_in",
            "obstacle12_+_+_+_+.jpg",
            "NOT A LICENSE PLATE", "+",
            "Not Plate",
            (200, 200, 200), (80, 80, 80)
        ),
        (
            "gate_in",
            "blurred13_-_-_-_-.jpg",
            "BLURRED / UNREADABLE", "-",
            "Cannot Read",
            (160, 160, 160), (40, 40, 40)
        ),
    ]
    
    for sub, filename, plate_text, prov_text, desc, plate_bg, plate_fg in samples:
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
        
        target_dir = sample_dir / sub if sub else sample_dir
        out_path = target_dir / filename
        img.save(out_path, quality=90)
        rel = out_path.relative_to(sample_dir)
        print(f"Created: {rel}")

if __name__ == "__main__":
    generate_samples()
