#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenCORE LPR Integration Package.
Exposes a clean, stable public API for embedding Thai LPR into external systems.
Compliant with ISO/IEC 42001:2023, ISO 8601, ISO 3166-2:TH.
"""

from selencore.lpr_engine import LPREngine, LPRResult, Detection, PlateReading

__all__ = [
    "LPREngine",
    "LPRResult",
    "Detection",
    "PlateReading",
]
