"""
Plate Palette & Color Extractor in CIE LAB Color Space.
Identifies background color and text foreground color from cropped plate images.
"""

from typing import Tuple
import cv2
import numpy as np


def extract_plate_colors(plate_bgr: np.ndarray) -> Tuple[str, str]:
    """
    Analyzes plate crop and returns (background_color_name, text_color_name).
    Possible background: 'white', 'yellow', 'orange', 'green'
    Possible text: 'black', 'green', 'blue', 'white'
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "white", "black"

    h, w = plate_bgr.shape[:2]
    # Crop central region to avoid outer plate border and vehicle bumper pixels
    margin_y = max(int(h * 0.12), 1)
    margin_x = max(int(w * 0.08), 1)
    central_crop = plate_bgr[margin_y:h - margin_y, margin_x:w - margin_x]
    if central_crop.size == 0:
        central_crop = plate_bgr

    # Convert to grayscale and apply Otsu thresholding to segment text from background
    gray = cv2.cvtColor(central_crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # In standard plates (white/yellow/orange bg), background pixels dominate (>60% of pixels)
    white_pixel_ratio = np.count_nonzero(binary) / binary.size

    # Usually background corresponds to the majority class
    if white_pixel_ratio > 0.45:
        bg_mask = (binary == 255)
        text_mask = (binary == 0)
    else:
        # Inverted plate (e.g. green background with white text)
        bg_mask = (binary == 0)
        text_mask = (binary == 255)

    # Convert to HSV and LAB
    hsv = cv2.cvtColor(central_crop, cv2.COLOR_BGR2HSV)

    # 1. Background Color Classification
    bg_pixels_hsv = hsv[bg_mask]
    if len(bg_pixels_hsv) == 0:
        bg_color = "white"
    else:
        median_h = np.median(bg_pixels_hsv[:, 0])
        median_s = np.median(bg_pixels_hsv[:, 1])
        median_v = np.median(bg_pixels_hsv[:, 2])

        # Saturation check: Low saturation -> White/Silver/Grey
        if median_s < 45 or median_v > 200 and median_s < 60:
            bg_color = "white"
        elif 18 <= median_h <= 38:  # Yellow / Orange range
            if median_h < 22 and median_s > 140:
                bg_color = "orange"
            else:
                bg_color = "yellow"
        elif 38 < median_h <= 85:   # Green range
            bg_color = "green"
        else:
            bg_color = "white"

    # 2. Text Color Classification
    text_pixels_hsv = hsv[text_mask]
    text_pixels_bgr = central_crop[text_mask]

    if len(text_pixels_hsv) == 0:
        text_color = "black"
    else:
        med_t_h = np.median(text_pixels_hsv[:, 0])
        med_t_s = np.median(text_pixels_hsv[:, 1])
        med_t_v = np.median(text_pixels_hsv[:, 2])

        # If background is green and text is very bright -> White text
        if bg_color == "green" and med_t_v > 160 and med_t_s < 50:
            text_color = "white"
        elif med_t_s < 45 or med_t_v < 65:
            # Low saturation or low brightness -> Black / Dark Gray
            text_color = "black"
        elif 95 <= med_t_h <= 130:
            # Blue hue range -> Blue text (Private Vans)
            text_color = "blue"
        elif 35 <= med_t_h <= 85:
            # Green hue range -> Green text (Pickups, Tuk-Tuks)
            text_color = "green"
        else:
            text_color = "black"

    return bg_color, text_color
