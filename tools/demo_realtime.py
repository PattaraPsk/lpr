#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Recognition (ALPR/ANPR) - Real-Time Edge Demo.
Fully compliant with International Standards:
- ISO 8601:2019: Date, Time & Duration Representation (UTC/Local ISO Timestamps)
- ISO/IEC 25012:2008: Data Quality Requirements & Integrity
- ISO/IEC 27001:2022: Security & RTSP Credential Masking
- ISO 3166-2:TH: Thailand Province Codes (TH-10 to TH-96)
- ISO/IEC 10646: Unicode NFC Thai Script Normalization

This module re-exports components from the modularized `tools.alpr` package
for clean backward-compatibility with all scripts, launchers, and tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.alpr import (
    RealtimeLPRDemo,
    TrackedPlate,
    ThreadedStreamCapture,
    render_thai_badge,
    overlay_badge,
    get_thai_font,
    draw_hud,
    box_iou,
    box_center_dist,
    detect_vehicles,
    detect_plates_coarse_to_fine,
    detect_plates_direct,
    match_vehicle_and_orientation,
    find_default_sample_video,
    resolve_model_path,
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

if __name__ == "__main__":
    main()
