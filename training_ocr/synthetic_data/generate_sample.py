#!/usr/bin/env python3
"""
Generate realistic synthetic vehicle images and sample cropped plates for instant testing.
"""
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_vehicle(output_path: str, plate_text: str = "1AB 8920", province: str = "BANGKOK"):
    width, height = 1280, 720
    img = np.full((height, width, 3), 45, dtype=np.uint8)

    for y in range(300):
        val = int(80 - (y / 300.0) * 35)
        img[y, :] = [val + 10, val + 5, val]

    noise = np.random.normal(0, 5, (height - 300, width, 3)).astype(np.int16)
    road_sub = img[300:, :].astype(np.int16) + noise
    img[300:, :] = np.clip(road_sub, 0, 255).astype(np.uint8)

    cv2.line(img, (150, height), (450, 300), (200, 200, 200), 6)
    cv2.line(img, (width - 150, height), (width - 450, 300), (200, 200, 200), 6)

    car_color = (70, 55, 45)
    pts = np.array([
        [400, 220], [880, 220], [960, 360], [990, 560],
        [950, 600], [330, 600], [290, 560], [320, 360]
    ], np.int32)
    cv2.fillPoly(img, [pts], car_color)

    win_pts = np.array([[430, 240], [850, 240], [920, 350], [360, 350]], np.int32)
    cv2.fillPoly(img, [win_pts], (25, 20, 20))

    cv2.rectangle(img, (320, 390), (440, 420), (30, 30, 210), -1)
    cv2.rectangle(img, (840, 390), (960, 420), (30, 30, 210), -1)

    cv2.rectangle(img, (290, 550), (350, 630), (20, 20, 20), -1)
    cv2.rectangle(img, (930, 550), (990, 630), (20, 20, 20), -1)

    cv2.rectangle(img, (420, 480), (860, 590), (50, 40, 35), -1)
    cv2.rectangle(img, (510, 495), (770, 575), (25, 25, 25), -1)

    plate_x1, plate_y1 = 525, 505
    plate_x2, plate_y2 = 755, 565
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (245, 245, 245), -1)
    cv2.rectangle(img, (plate_x1, plate_y1), (plate_x2, plate_y2), (30, 30, 30), 2)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, plate_text, (plate_x1 + 22, plate_y1 + 40), font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, province, (plate_x1 + 75, plate_y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(output_path, quality=92)
    print(f"Generated sample vehicle: {output_path}")

def create_sample_cropped_plate(output_path: str, reg_text: str, province_text: str, font_path: str):
    w, h = 680, 300
    plate = Image.new("RGB", (w, h), (248, 248, 248))
    draw = ImageDraw.Draw(plate)

    margin = 12
    draw.rounded_rectangle([(margin, margin), (w - margin, h - margin)], radius=20, outline=(20, 20, 20), width=8)
    draw.rounded_rectangle([(margin + 6, margin + 6), (w - margin - 6, h - margin - 6)], radius=16, outline=(210, 210, 210), width=2)

    s_rad = 12
    draw.ellipse([(65, 45), (65 + 2 * s_rad, 45 + 2 * s_rad)], fill=(120, 120, 120), outline=(50, 50, 50), width=2)
    draw.ellipse([(w - 65 - 2 * s_rad, 45), (w - 65, 45 + 2 * s_rad)], fill=(120, 120, 120), outline=(50, 50, 50), width=2)

    font_top = ImageFont.truetype(font_path, 115)
    font_bot = ImageFont.truetype(font_path, 42)

    bb_top = draw.textbbox((0, 0), reg_text, font=font_top)
    top_w = bb_top[2] - bb_top[0]
    draw.text(((w - top_w) // 2, 40), reg_text, fill=(18, 18, 18), font=font_top)

    bb_bot = draw.textbbox((0, 0), province_text, font=font_bot)
    bot_w = bb_bot[2] - bb_bot[0]
    draw.text(((w - bot_w) // 2, 195), province_text, fill=(18, 18, 18), font=font_bot)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plate.save(output_path, quality=95)
    print(f"Generated sample cropped plate: {output_path}")

if __name__ == "__main__":
    font = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansThai-Bold.ttf")
    if not os.path.exists(font):
        font = "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"

    create_synthetic_vehicle("input_images/sample_car_01.jpg", "1AB 8920", "BANGKOK")
    create_synthetic_vehicle("input_images/sample_car_02.jpg", "9XY 4512", "THAILAND")

    create_sample_cropped_plate("cropped/sample_plate_01.jpg", "1กข 8920", "กรุงเทพมหานคร", font)
    create_sample_cropped_plate("cropped/sample_plate_02.jpg", "9วย 4512", "เชียงใหม่", font)
    create_sample_cropped_plate("cropped/sample_plate_03.jpg", "2ฒณ 7631", "ภูเก็ต", font)
