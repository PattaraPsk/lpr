#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Recognition (ALPR/ANPR) - Real-Time Demo Launcher.
Entry point at root repository directory.
"""

import sys
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from tools.demo_realtime import main

if __name__ == "__main__":
    main()
