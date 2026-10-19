#!/usr/bin/env python3
"""
Desktop GUI for Thai License Plate (LPR) Dataset Annotation & Review.
"""

import sys
from pathlib import Path

# Add current directory (DataManipulator) and tools parent directory to sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from app.main_window import MainWindow

def main():
    # Enable High DPI scaling
    app = QApplication(sys.argv)
    app.setApplicationName("Thai LPR Annotator")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
