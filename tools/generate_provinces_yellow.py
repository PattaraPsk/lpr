#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backward-compatible wrapper for Thai License Plate Large Plate (แผ่นป้ายทะเบียนขนาดใหญ่) Generator.
Delegates to tools.generate_provinces_large_plate.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.generate_provinces_large_plate import generate_all_large_plates

if __name__ == "__main__":
    generate_all_large_plates()
