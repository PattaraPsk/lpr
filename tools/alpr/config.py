#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALPR Configuration & Model Resolver.
Compliant with ISO/IEC 42001 and ISO/IEC 25012.
"""

from pathlib import Path
from typing import List, Optional

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Default Candidate Paths for Neural Inference Engines
DETECTOR_CANDIDATES: List[Path] = [
    ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt",
    ROOT_DIR / "models" / "detector" / "vehicle_plate_detector.pt",
    ROOT_DIR / "vehicle_plate_detector.pt",
]

VEHICLE_CANDIDATES: List[Path] = [
    ROOT_DIR / "models" / "detector" / "yolov8s.pt",
    ROOT_DIR / "yolov8s.pt",
]

READER_CANDIDATES: List[Path] = [
    ROOT_DIR / "models" / "reader" / "thai_plate_reader.pth",
    ROOT_DIR / "models" / "reader" / "selen-lpr-0.0.1" / "best_reader.pth",
    ROOT_DIR / "thai_plate_reader.pth",
]

THAI_FONT_CANDIDATES: List[str] = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\leelawad.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/thai/Loma.ttf",
    "/usr/share/fonts/truetype/tlwg/TlwgTypo.ttf",
]

DEFAULT_CONF_THRESH = 0.25
DEFAULT_VEH_CONF_THRESH = 0.35
DEFAULT_DISPLAY_WIDTH = 1280
DEFAULT_DISPLAY_HEIGHT = 720
DEFAULT_SKIP_OCR_FRAMES = 5


def resolve_model_path(user_path: Optional[str | Path], default_candidates: List[Path]) -> Path:
    """
    Resolves model checkpoint file from user input or candidate paths.
    Prioritizes models/ directory over root fallbacks.
    """
    if user_path:
        p = Path(user_path).resolve()
        if p.is_file():
            return p
    for cand in default_candidates:
        if cand.is_file():
            return cand.resolve()
    return default_candidates[0]
