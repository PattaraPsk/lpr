#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
License Plate Image Augmentation Pipeline
==========================================
Comprehensive set of augmentations for creating realistic training data
from synthetic or real license plate images.

Each augmentation accepts a PIL Image + intensity (0.0-1.0) and returns
the augmented image. Geometric transforms also update bounding boxes.
"""

import io
import math
import random
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Individual Augmentation Functions
# ---------------------------------------------------------------------------

def perspective_transform(
    img: Image.Image, bboxes: List[Dict], intensity: float = 0.5
) -> Tuple[Image.Image, List[Dict]]:
    """Random 4-point perspective warp simulating viewing angles."""
    w, h = img.size
    max_shift = int(min(w, h) * 0.12 * intensity)
    if max_shift < 2:
        return img, bboxes

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), random.randint(0, max_shift)],
        [w - random.randint(0, max_shift), h - random.randint(0, max_shift)],
        [random.randint(0, max_shift), h - random.randint(0, max_shift)],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    arr = np.array(img)
    warped = cv2.warpPerspective(arr, M, (w, h),
                                 borderMode=cv2.BORDER_REPLICATE)

    # Transform bounding boxes
    new_bboxes = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb["bbox"]
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
        nx1 = max(0, int(transformed[:, 0].min()))
        ny1 = max(0, int(transformed[:, 1].min()))
        nx2 = min(w, int(transformed[:, 0].max()))
        ny2 = min(h, int(transformed[:, 1].max()))
        new_bboxes.append({"char": bb["char"], "bbox": (nx1, ny1, nx2, ny2)})

    return Image.fromarray(warped), new_bboxes


def rotation(
    img: Image.Image, bboxes: List[Dict], intensity: float = 0.5
) -> Tuple[Image.Image, List[Dict]]:
    """Random rotation -15° to +15° scaled by intensity."""
    max_angle = 15 * intensity
    angle = random.uniform(-max_angle, max_angle)
    if abs(angle) < 0.5:
        return img, bboxes

    w, h = img.size
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False,
                         fillcolor=(128, 128, 128))

    # Transform bboxes
    cx, cy = w / 2, h / 2
    rad = math.radians(-angle)  # PIL rotates CCW, we need CW transform
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    new_bboxes = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb["bbox"]
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        transformed = []
        for px, py in corners:
            dx, dy = px - cx, py - cy
            nx = cos_a * dx - sin_a * dy + cx
            ny = sin_a * dx + cos_a * dy + cy
            transformed.append((nx, ny))
        xs = [p[0] for p in transformed]
        ys = [p[1] for p in transformed]
        new_bboxes.append({
            "char": bb["char"],
            "bbox": (
                max(0, int(min(xs))),
                max(0, int(min(ys))),
                min(w, int(max(xs))),
                min(h, int(max(ys)))
            )
        })

    return rotated, new_bboxes


def gaussian_blur(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Simulates out-of-focus camera."""
    radius = max(0.5, 3.0 * intensity)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Simulates motion blur from moving vehicle."""
    kernel_size = max(3, int(15 * intensity))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Horizontal motion kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    arr = np.array(img)
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(blurred)


def brightness_contrast(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Random brightness and contrast adjustments."""
    brightness_range = 0.4 * intensity
    contrast_range = 0.4 * intensity

    b_factor = random.uniform(1.0 - brightness_range, 1.0 + brightness_range)
    c_factor = random.uniform(1.0 - contrast_range, 1.0 + contrast_range)

    img = ImageEnhance.Brightness(img).enhance(b_factor)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def gaussian_noise(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Adds Gaussian sensor noise."""
    arr = np.array(img, dtype=np.float32)
    sigma = 25 * intensity
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def salt_pepper_noise(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Adds salt and pepper noise."""
    arr = np.array(img).copy()
    prob = 0.05 * intensity
    # Salt
    num_salt = int(prob * arr.size / arr.shape[2] if len(arr.shape) == 3 else prob * arr.size)
    coords = tuple(np.random.randint(0, d, num_salt) for d in arr.shape[:2])
    arr[coords] = 255
    # Pepper
    num_pepper = int(prob * arr.size / arr.shape[2] if len(arr.shape) == 3 else prob * arr.size)
    coords = tuple(np.random.randint(0, d, num_pepper) for d in arr.shape[:2])
    arr[coords] = 0
    return Image.fromarray(arr)


def jpeg_artifacts(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Simulates JPEG compression artifacts."""
    quality = max(5, int(50 - 40 * intensity))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def color_jitter(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Random hue/saturation shifts."""
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)

    hue_shift = random.uniform(-15, 15) * intensity
    sat_factor = random.uniform(0.7, 1.3) if intensity > 0.3 else 1.0

    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return Image.fromarray(result)


def shadow_overlay(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Random shadow bands across the plate."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    num_bands = random.randint(1, 3)
    for _ in range(num_bands):
        angle = random.uniform(0, math.pi)
        shadow_width = random.randint(int(w * 0.1), int(w * 0.4))
        offset = random.randint(0, w)
        darkness = 0.3 + 0.4 * (1 - intensity)

        mask = np.ones((h, w), dtype=np.float32)
        for y in range(h):
            x_center = offset + y * math.tan(angle - math.pi / 2)
            x1 = max(0, int(x_center - shadow_width // 2))
            x2 = min(w, int(x_center + shadow_width // 2))
            mask[y, x1:x2] = darkness

        for c in range(arr.shape[2]):
            arr[:, :, c] *= mask

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def rain_droplets(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Simulated raindrop distortion."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]

    num_drops = int(20 * intensity)
    for _ in range(num_drops):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        radius = random.randint(2, max(3, int(8 * intensity)))

        # Create subtle lens-like distortion
        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)

        region = arr[y1:y2, x1:x2].copy()
        if region.size == 0:
            continue

        # Slight blur + brightness increase for droplet
        blurred = cv2.GaussianBlur(region, (3, 3), 0)
        highlight = np.clip(blurred.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        arr[y1:y2, x1:x2] = highlight

    return Image.fromarray(arr)


def dirt_overlay(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Partial occlusion with brown/gray dirt patches."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    num_patches = int(5 * intensity) + 1
    for _ in range(num_patches):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(5, max(6, int(w * 0.15 * intensity)))
        ry = random.randint(3, max(4, int(h * 0.15 * intensity)))

        # Brownish/grayish color
        r = random.randint(100, 160)
        g = random.randint(80, 130)
        b = random.randint(60, 100)
        alpha = random.randint(80, 180)

        draw.ellipse(
            [(cx - rx, cy - ry), (cx + rx, cy + ry)],
            fill=(r, g, b, alpha) if overlay.mode == "RGBA" else (r, g, b)
        )

    # Blend with original
    arr_orig = np.array(img, dtype=np.float32)
    arr_overlay = np.array(overlay, dtype=np.float32)
    blend = 0.3 + 0.5 * intensity
    result = np.clip(arr_orig * (1 - blend) + arr_overlay * blend, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


def sunlight_glare(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Bright glare spot overlay."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    radius = int(min(w, h) * (0.2 + 0.3 * intensity))

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.clip(1.0 - dist / max(radius, 1), 0, 1)
    strength = 100 * intensity

    for c in range(arr.shape[2]):
        arr[:, :, c] = np.clip(arr[:, :, c] + mask * strength, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def resize_upscale(
    img: Image.Image, intensity: float = 0.5
) -> Image.Image:
    """Downsample then upsample to simulate low-res camera."""
    w, h = img.size
    factor = max(0.15, 1.0 - 0.75 * intensity)
    small_w = max(8, int(w * factor))
    small_h = max(4, int(h * factor))

    small = img.resize((small_w, small_h), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


# ---------------------------------------------------------------------------
# Augmentation Registry
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY = {
    "perspective": {
        "func": perspective_transform,
        "geometric": True,
        "group": "Geometric",
        "display": "Perspective Warp",
    },
    "rotation": {
        "func": rotation,
        "geometric": True,
        "group": "Geometric",
        "display": "Rotation",
    },
    "gaussian_blur": {
        "func": gaussian_blur,
        "geometric": False,
        "group": "Blur",
        "display": "Gaussian Blur",
    },
    "motion_blur": {
        "func": motion_blur,
        "geometric": False,
        "group": "Blur",
        "display": "Motion Blur",
    },
    "brightness_contrast": {
        "func": brightness_contrast,
        "geometric": False,
        "group": "Lighting",
        "display": "Brightness / Contrast",
    },
    "gaussian_noise": {
        "func": gaussian_noise,
        "geometric": False,
        "group": "Noise",
        "display": "Gaussian Noise",
    },
    "salt_pepper": {
        "func": salt_pepper_noise,
        "geometric": False,
        "group": "Noise",
        "display": "Salt & Pepper",
    },
    "jpeg_artifacts": {
        "func": jpeg_artifacts,
        "geometric": False,
        "group": "Noise",
        "display": "JPEG Artifacts",
    },
    "color_jitter": {
        "func": color_jitter,
        "geometric": False,
        "group": "Lighting",
        "display": "Color Jitter",
    },
    "shadow": {
        "func": shadow_overlay,
        "geometric": False,
        "group": "Lighting",
        "display": "Shadow Overlay",
    },
    "rain": {
        "func": rain_droplets,
        "geometric": False,
        "group": "Weather",
        "display": "Rain Droplets",
    },
    "dirt": {
        "func": dirt_overlay,
        "geometric": False,
        "group": "Weather",
        "display": "Dirt / Mud",
    },
    "glare": {
        "func": sunlight_glare,
        "geometric": False,
        "group": "Lighting",
        "display": "Sunlight Glare",
    },
    "resize_upscale": {
        "func": resize_upscale,
        "geometric": False,
        "group": "Resolution",
        "display": "Low-Res Simulation",
    },
}

# Preset configurations
PRESETS = {
    "None": {},
    "Light": {
        "brightness_contrast": {"enabled": True, "intensity": 0.3},
        "gaussian_noise": {"enabled": True, "intensity": 0.2},
        "rotation": {"enabled": True, "intensity": 0.2},
    },
    "Medium": {
        "perspective": {"enabled": True, "intensity": 0.4},
        "rotation": {"enabled": True, "intensity": 0.3},
        "gaussian_blur": {"enabled": True, "intensity": 0.3},
        "brightness_contrast": {"enabled": True, "intensity": 0.4},
        "gaussian_noise": {"enabled": True, "intensity": 0.3},
        "color_jitter": {"enabled": True, "intensity": 0.3},
    },
    "Heavy": {
        "perspective": {"enabled": True, "intensity": 0.6},
        "rotation": {"enabled": True, "intensity": 0.5},
        "gaussian_blur": {"enabled": True, "intensity": 0.4},
        "motion_blur": {"enabled": True, "intensity": 0.4},
        "brightness_contrast": {"enabled": True, "intensity": 0.5},
        "gaussian_noise": {"enabled": True, "intensity": 0.4},
        "salt_pepper": {"enabled": True, "intensity": 0.3},
        "jpeg_artifacts": {"enabled": True, "intensity": 0.4},
        "shadow": {"enabled": True, "intensity": 0.4},
        "dirt": {"enabled": True, "intensity": 0.3},
    },
    "Extreme": {
        "perspective": {"enabled": True, "intensity": 0.8},
        "rotation": {"enabled": True, "intensity": 0.7},
        "gaussian_blur": {"enabled": True, "intensity": 0.6},
        "motion_blur": {"enabled": True, "intensity": 0.6},
        "brightness_contrast": {"enabled": True, "intensity": 0.7},
        "gaussian_noise": {"enabled": True, "intensity": 0.6},
        "salt_pepper": {"enabled": True, "intensity": 0.5},
        "jpeg_artifacts": {"enabled": True, "intensity": 0.6},
        "color_jitter": {"enabled": True, "intensity": 0.5},
        "shadow": {"enabled": True, "intensity": 0.6},
        "rain": {"enabled": True, "intensity": 0.5},
        "dirt": {"enabled": True, "intensity": 0.5},
        "glare": {"enabled": True, "intensity": 0.5},
        "resize_upscale": {"enabled": True, "intensity": 0.5},
    },
}


# ---------------------------------------------------------------------------
# Augmentation Pipeline Class
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """
    Configurable augmentation pipeline for license plate images.

    Usage:
        config = {
            "perspective": {"enabled": True, "intensity": 0.5},
            "gaussian_blur": {"enabled": True, "intensity": 0.3},
        }
        pipeline = AugmentationPipeline(config)
        aug_img, aug_bboxes = pipeline.apply(image, bboxes)
    """

    def __init__(self, augmentation_config: Optional[dict] = None):
        self.config = augmentation_config or {}

    def update_config(self, new_config: dict):
        """Replace the augmentation configuration."""
        self.config = new_config

    def set_preset(self, preset_name: str):
        """Load a preset configuration."""
        if preset_name in PRESETS:
            self.config = PRESETS[preset_name].copy()

    def apply(
        self, image: Image.Image, bboxes: Optional[List[Dict]] = None
    ) -> Tuple[Image.Image, List[Dict]]:
        """
        Apply enabled augmentations randomly (each has a 70% chance of being applied).
        Geometric augmentations are applied first.
        """
        if bboxes is None:
            bboxes = []

        img = image.copy()
        current_bboxes = [b.copy() for b in bboxes]

        # Sort: geometric first, then pixel-level
        enabled = [
            (name, cfg)
            for name, cfg in self.config.items()
            if cfg.get("enabled", False) and name in AUGMENTATION_REGISTRY
        ]

        geometric = [(n, c) for n, c in enabled if AUGMENTATION_REGISTRY[n]["geometric"]]
        pixel = [(n, c) for n, c in enabled if not AUGMENTATION_REGISTRY[n]["geometric"]]

        for name, cfg in geometric:
            if random.random() < 0.7:
                intensity = cfg.get("intensity", 0.5)
                reg = AUGMENTATION_REGISTRY[name]
                img, current_bboxes = reg["func"](img, current_bboxes, intensity)

        for name, cfg in pixel:
            if random.random() < 0.7:
                intensity = cfg.get("intensity", 0.5)
                reg = AUGMENTATION_REGISTRY[name]
                img = reg["func"](img, intensity)

        return img, current_bboxes

    def apply_all(
        self, image: Image.Image, bboxes: Optional[List[Dict]] = None
    ) -> Tuple[Image.Image, List[Dict]]:
        """Apply ALL enabled augmentations deterministically in sequence."""
        if bboxes is None:
            bboxes = []

        img = image.copy()
        current_bboxes = [b.copy() for b in bboxes]

        enabled = [
            (name, cfg)
            for name, cfg in self.config.items()
            if cfg.get("enabled", False) and name in AUGMENTATION_REGISTRY
        ]

        geometric = [(n, c) for n, c in enabled if AUGMENTATION_REGISTRY[n]["geometric"]]
        pixel = [(n, c) for n, c in enabled if not AUGMENTATION_REGISTRY[n]["geometric"]]

        for name, cfg in geometric:
            intensity = cfg.get("intensity", 0.5)
            reg = AUGMENTATION_REGISTRY[name]
            img, current_bboxes = reg["func"](img, current_bboxes, intensity)

        for name, cfg in pixel:
            intensity = cfg.get("intensity", 0.5)
            reg = AUGMENTATION_REGISTRY[name]
            img = reg["func"](img, intensity)

        return img, current_bboxes
