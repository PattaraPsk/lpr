#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Augmentor — Windows Desktop App Launcher
"""

import os
import sys
from pathlib import Path

# Ensure root and tools are in PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(CURRENT_DIR))

# Ensure UTF-8 on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from PyQt6.QtWidgets import QApplication


def main():
    # Create QApplication instance first so all Qt engines & fonts are initialized
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Import windows after QApplication exists
    from main_window import PlateAugmentorWindow

    window = PlateAugmentorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
