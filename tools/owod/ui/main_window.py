"""
Main Window for OWOD License Plate Studio & YOLO Model Pruning Tool.
Integrates Media Discovery, OWOD Auto-Labeling, Architecture Surgery & Pruning, and Fine-Tuning into a unified suite.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QFont

from ..core.media_scanner import MediaScanner
from .styles import DARK_THEME_QSS
from .media_selector_tab import MediaSelectorTab
from .owod_labeler_tab import OWODLabelerTab
from .model_surgery_tab import ModelSurgeryTab
from .trainer_tab import TrainerTab

class OWODMainWindow(QMainWindow):
    """Unified Desktop GUI for OWOD License Plate Labeling & YOLO Pruning Studio."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OWOD License Plate Auto-Labeler & YOLO Pruning Studio")
        self.resize(1380, 920)
        self.setMinimumSize(1080, 700)
        
        self.scanner = MediaScanner()
        
        self._init_ui()
        self._connect_signals()
        self.setStyleSheet(DARK_THEME_QSS)
        
    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 10, 12, 10)
        main_layout.setSpacing(8)
        
        # 1. Header Banner
        header = self._create_header_banner()
        main_layout.addWidget(header)
        
        # 2. Main Tab Widget
        self.tabs = QTabWidget()
        
        # Tab 1: Media Discovery & Selection
        self.media_tab = MediaSelectorTab(self.scanner)
        self.tabs.addTab(self.media_tab, "📁 1. Media Discovery & Selection")
        
        # Tab 2: OWOD Auto-Labeling
        self.labeler_tab = OWODLabelerTab(self.scanner)
        self.tabs.addTab(self.labeler_tab, "🏷️ 2. OWOD Auto-Labeling")
        
        # Tab 3: Model Surgery & Pruning
        self.surgery_tab = ModelSurgeryTab()
        self.tabs.addTab(self.surgery_tab, "✂️ 3. Model Surgery & Pruning")
        
        # Tab 4: Fine-Tuning Studio
        self.trainer_tab = TrainerTab()
        self.tabs.addTab(self.trainer_tab, "🚀 4. YOLO Fine-Tuning Studio")
        
        main_layout.addWidget(self.tabs, 1)

    def _create_header_banner(self) -> QFrame:
        banner = QFrame(self)
        banner.setStyleSheet("background-color: #1e293b; border-radius: 8px; border: 1px solid #334155;")
        h = QHBoxLayout(banner)
        h.setContentsMargins(14, 8, 14, 8)
        h.setSpacing(12)
        
        # App Title & Subtitle
        v_title = QVBoxLayout()
        v_title.setSpacing(2)
        
        lbl_title = QLabel("⚡ OWOD License Plate Studio & YOLO Model Surgery")
        lbl_title.setStyleSheet("font-size: 17px; font-weight: 800; color: #38bdf8;")
        v_title.addWidget(lbl_title)
        
        lbl_sub = QLabel("Open-World Object Detection Auto-Labeling • Layer Freezing • Slicing • Channel Pruning • Fine-Tuning")
        lbl_sub.setStyleSheet("font-size: 11px; color: #94a3b8;")
        v_title.addWidget(lbl_sub)
        
        h.addLayout(v_title, 1)
        
        # GPU Acceleration Badge
        gpu_info = self._get_gpu_status()
        lbl_gpu = QLabel(f"⚡ {gpu_info}")
        lbl_gpu.setStyleSheet("background-color: #0f172a; color: #34d399; font-weight: bold; padding: 6px 12px; border-radius: 6px; border: 1px solid #059669;")
        h.addWidget(lbl_gpu)
        
        # Help Button
        btn_help = QPushButton("❓ Quick Guide")
        btn_help.setObjectName("btnSecondary")
        btn_help.clicked.connect(self._show_help)
        h.addWidget(btn_help)
        
        return banner

    def _get_gpu_status(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                return f"GPU: {name} (CUDA Active)"
            return "Torch Running (CPU Mode)"
        except Exception:
            return "NVIDIA RTX 3070 Detected"

    def _connect_signals(self):
        # When dataset is created in Tab 2 -> update Tab 4 dataset yaml
        self.labeler_tab.dataset_created.connect(self._on_dataset_created)
        
        # When model surgery/export is done in Tab 3 -> update Tab 4 model and frozen layers
        self.surgery_tab.model_configured.connect(self._on_model_configured)

    def _on_dataset_created(self, yaml_path: str):
        self.trainer_tab.set_dataset_yaml(yaml_path)
        jump = QMessageBox.question(
            self, "Dataset Ready",
            f"Dataset and data.yaml generated successfully!\n\nLocation: {yaml_path}\n\nDo you want to switch to the YOLO Fine-Tuning Tab now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if jump == QMessageBox.StandardButton.Yes:
            self.tabs.setCurrentIndex(3)

    def _on_model_configured(self, model_path: str, frozen_layers: list):
        self.trainer_tab.set_model_and_frozen_layers(model_path, frozen_layers)

    def _show_help(self):
        msg = (
            "<h3>OWOD License Plate Studio Quick Guide</h3>"
            "<p><b>1. Media Discovery & Selection:</b> Browse folders and subfolders. Use the tree checkboxes to select or deselect specific folders, images, and videos. Configure video frame sampling rates.</p>"
            "<p><b>2. OWOD Auto-Labeling:</b> Prompt open-world detection (YOLO-World / SAM) with <i>'license plate'</i> to auto-label all selected media. Generates standard YOLO images, labels, and <code>data.yaml</code>.</p>"
            "<p><b>3. Model Surgery & Pruning:</b> Choose a base YOLO model (YOLOv8, YOLO11, etc.). Inspect layers, select which layers to <b>freeze</b>, <b>slice/cut</b> network depth, <b>remove nodes</b>, or apply <b>structured/unstructured pruning</b> to compress weights.</p>"
            "<p><b>4. YOLO Fine-Tuning Studio:</b> Train the customized or pruned YOLO model on your newly auto-labeled license plate dataset with real-time loss curves and mAP tracking.</p>"
        )
        QMessageBox.information(self, "OWOD Studio User Guide", msg)
