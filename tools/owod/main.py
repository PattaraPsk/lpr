#!/usr/bin/env python3
"""
Launcher for OWOD License Plate Studio & YOLO Model Pruning Tool.
"""

import sys
import os
from pathlib import Path

# Add parent tools directory and owod directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from owod.ui.main_window import OWODMainWindow

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OWOD License Plate & YOLO Pruning Studio")
    
    window = OWODMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
