#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Edge-Optimized Thai License Plate Recognition (ALPR/ANPR) Package.
Compliant with ISO 8601, ISO/IEC 25012, ISO/IEC 27001, and ISO 3166-2:TH.
"""

from tools.alpr.config import (
    ROOT_DIR,
    resolve_model_path,
    DETECTOR_CANDIDATES,
    VEHICLE_CANDIDATES,
    READER_CANDIDATES,
    THAI_FONT_CANDIDATES,
)
from tools.alpr.stream import ThreadedStreamCapture
from tools.alpr.tracker import TrackedPlate, box_iou, box_center_dist
from tools.alpr.detector import (
    detect_vehicles,
    detect_plates_coarse_to_fine,
    detect_plates_direct,
    match_vehicle_and_orientation,
)
from tools.alpr.hud import (
    get_thai_font,
    render_thai_badge,
    overlay_badge,
    draw_hud,
)
from tools.alpr.demo import (
    RealtimeLPRDemo,
    find_default_sample_video,
    main,
)

__all__ = [
    "RealtimeLPRDemo",
    "TrackedPlate",
    "ThreadedStreamCapture",
    "render_thai_badge",
    "overlay_badge",
    "get_thai_font",
    "draw_hud",
    "box_iou",
    "box_center_dist",
    "detect_vehicles",
    "detect_plates_coarse_to_fine",
    "detect_plates_direct",
    "match_vehicle_and_orientation",
    "find_default_sample_video",
    "resolve_model_path",
    "main",
]
