#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Augmentor — GUI Application (PyQt6)
======================================================
Comprehensive Windows desktop app for:
1. Interactive Reference Plate Editor:
   - Load any reference plate image or blank template
   - Freely move, resize, rotate, and color each character
   - Interactive mouse gizmos (rotation top handle, resize corner handle)
   - Fine-tuned inspector sliders & spinboxes
   - Save / load project layouts (.json)
2. Synthetic Thai license plate rendering & batch generator
3. Interactive multi-stage augmentation pipeline (Perspective, Motion blur, Weather, Dirt, Lighting)
4. Batch augmenting existing plate datasets
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PIL import Image

# Import sibling modules
try:
    from .plate_renderer import (
        PLATE_CONFIGS,
        PROVINCES,
        generate_random_plate,
        get_plate_types,
        render_plate,
    )
    from .augmentor import (
        AUGMENTATION_REGISTRY,
        PRESETS,
        AugmentationPipeline,
    )
    from .exporter import YOLOExporter
    from .interactive_editor_tab import InteractiveEditorTab
except ImportError:
    from plate_renderer import (
        PLATE_CONFIGS,
        PROVINCES,
        generate_random_plate,
        get_plate_types,
        render_plate,
    )
    from augmentor import (
        AUGMENTATION_REGISTRY,
        PRESETS,
        AugmentationPipeline,
    )
    from exporter import YOLOExporter
    from interactive_editor_tab import InteractiveEditorTab


# Helper to convert PIL Image to QPixmap
def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Background Worker Threads
# ---------------------------------------------------------------------------

class SyntheticBatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, float, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        count: int,
        out_dir: str,
        val_split: float,
        plate_type_choice: str,
        aug_config: dict,
        export_yolo: bool,
    ):
        super().__init__()
        self.count = count
        self.out_dir = out_dir
        self.val_split = val_split
        self.plate_type_choice = plate_type_choice
        self.aug_config = aug_config
        self.export_yolo = export_yolo
        self.is_running = True

    def run(self):
        start_time = time.time()
        pipeline = AugmentationPipeline(self.aug_config)
        exporter = YOLOExporter(self.out_dir, val_split=self.val_split)

        types_list = list(PLATE_CONFIGS.keys())
        generated = 0

        try:
            for idx in range(self.count):
                if not self.is_running:
                    break

                ptype = (
                    self.plate_type_choice
                    if self.plate_type_choice in PLATE_CONFIGS
                    else random.choice(types_list)
                )

                # Render base plate
                img, bboxes, meta = generate_random_plate(ptype)

                # Augment
                aug_img, aug_bboxes = pipeline.apply(img, bboxes)

                is_val = random.random() < self.val_split
                stem = f"aug_{ptype}_{idx:05d}_{meta['d3']}_{meta['d4']}"
                stem = "".join([c if c.isalnum() or c in "_-" else "_" for c in stem])

                if self.export_yolo:
                    exporter.export_sample(aug_img, aug_bboxes, stem, metadata=meta, is_val=is_val)
                else:
                    dest = Path(self.out_dir) / f"{stem}.jpg"
                    aug_img.save(dest, quality=95)

                generated += 1
                if (idx + 1) % 5 == 0 or idx == self.count - 1:
                    self.progress.emit(idx + 1, self.count, f"Generated {idx + 1}/{self.count} plates...")

            if self.export_yolo:
                exporter.finalize()

            elapsed = time.time() - start_time
            self.finished.emit(generated, elapsed, self.out_dir)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False


class ExistingBatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(int, float, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        image_paths: List[Path],
        out_dir: str,
        copies_per_img: int,
        aug_config: dict,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.out_dir = Path(out_dir)
        self.copies_per_img = copies_per_img
        self.aug_config = aug_config
        self.is_running = True

    def run(self):
        start_time = time.time()
        pipeline = AugmentationPipeline(self.aug_config)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        total_tasks = len(self.image_paths) * self.copies_per_img
        done = 0

        try:
            for p in self.image_paths:
                if not self.is_running:
                    break
                try:
                    src_img = Image.open(p).convert("RGB")
                except Exception:
                    continue

                for copy_idx in range(self.copies_per_img):
                    if not self.is_running:
                        break
                    aug_img, _ = pipeline.apply(src_img, [])
                    out_name = f"{p.stem}_aug_{copy_idx:02d}.jpg"
                    aug_img.save(self.out_dir / out_name, quality=95)
                    done += 1

                    if done % 5 == 0 or done == total_tasks:
                        self.progress.emit(done, total_tasks, f"Augmented {done}/{total_tasks}...")

            elapsed = time.time() - start_time
            self.finished.emit(done, elapsed, str(self.out_dir))
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class PlateAugmentorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thai LPR — Plate Augmentor & Synthetic Studio")
        self.resize(1380, 880)
        self.setMinimumSize(1080, 680)

        # State for Live Pipeline & Previews
        self.current_base_img: Optional[Image.Image] = None
        self.current_base_bboxes: List[Dict] = []
        self.current_metadata: Dict = {}
        self.current_aug_img: Optional[Image.Image] = None
        self.current_aug_bboxes: List[Dict] = []
        self.loaded_image_paths: List[Path] = []

        # Augmentation widgets map: name -> (checkbox, slider, label)
        self.aug_controls: Dict[str, Tuple[QCheckBox, QSlider, QLabel]] = {}

        self._setup_ui()
        self._apply_dark_theme()

        # Generate initial plate for live pipeline
        self.on_randomize_plate()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)

        # Top-level Tab Bar
        self.main_tabs = QTabWidget()
        main_layout.addWidget(self.main_tabs)

        # Tab 1: Interactive Plate Canvas (Move, Resize, Rotate characters on reference plate)
        self.interactive_tab = InteractiveEditorTab(self)
        self.interactive_tab.composite_rendered.connect(self._on_custom_composite_received)
        self.main_tabs.addTab(self.interactive_tab, "🎨 Interactive Canvas Editor (Move / Resize / Rotate)")

        # Tab 2: Synthetic Generator & Live Augmentation Pipeline
        pipeline_tab = self._create_pipeline_tab()
        self.main_tabs.addTab(pipeline_tab, "⚡ Live Augmentation Pipeline & Batch Generator")

        # Tab 3: Batch Augment Existing Dataset Folder
        load_tab = self._create_load_tab()
        self.main_tabs.addTab(load_tab, "📁 Batch Augment Existing Plates")

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready. Use Interactive Canvas to design plates or run batch generators.")

    # -----------------------------------------------------------------------
    # Tab 2: Synthetic Generator & Live Augmentation Pipeline
    # -----------------------------------------------------------------------
    def _create_pipeline_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_tab_widget = QTabWidget()
        left_layout.addWidget(left_tab_widget)

        # Sub-tab: Generator & Batch
        tab_gen = self._create_generate_subtab()
        left_tab_widget.addTab(tab_gen, "Generator & Batch")

        # Sub-tab: Augmentations
        tab_aug = self._create_augment_subtab()
        left_tab_widget.addTab(tab_aug, "Augmentation Filters")

        splitter.addWidget(left_panel)
        splitter.setStretchFactor(0, 4)

        # Right Viewport
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)

        # Top bar
        top_bar = QHBoxLayout()
        self.chk_show_bboxes = QCheckBox("Show Character BBoxes (YOLO)")
        self.chk_show_bboxes.setChecked(True)
        self.chk_show_bboxes.toggled.connect(self._refresh_previews)
        top_bar.addWidget(self.chk_show_bboxes)

        top_bar.addStretch()
        self.lbl_meta = QLabel("Plate: -")
        self.lbl_meta.setStyleSheet("font-weight: bold; color: #60a5fa;")
        top_bar.addWidget(self.lbl_meta)
        right_layout.addLayout(top_bar)

        # Previews
        preview_box = QGroupBox("Live Viewport")
        preview_vbox = QVBoxLayout(preview_box)

        lbl_orig_title = QLabel("Source / Base Plate:")
        lbl_orig_title.setStyleSheet("font-size: 11px; color: #aaa;")
        preview_vbox.addWidget(lbl_orig_title)

        self.lbl_preview_orig = QLabel()
        self.lbl_preview_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_orig.setMinimumHeight(140)
        self.lbl_preview_orig.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        preview_vbox.addWidget(self.lbl_preview_orig)

        lbl_aug_title = QLabel("Augmented Output (Camera / Motion / Weather Distortion):")
        lbl_aug_title.setStyleSheet("font-size: 11px; color: #38bdf8; font-weight: bold;")
        preview_vbox.addWidget(lbl_aug_title)

        self.lbl_preview_aug = QLabel()
        self.lbl_preview_aug.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview_aug.setMinimumHeight(170)
        self.lbl_preview_aug.setStyleSheet("background-color: #181818; border: 2px solid #2563eb; border-radius: 6px;")
        preview_vbox.addWidget(self.lbl_preview_aug)

        right_layout.addWidget(preview_box, stretch=3)

        # Quick variations strip
        variations_group = QGroupBox("Augmentation Variations (Quick Previews)")
        var_layout = QHBoxLayout(variations_group)
        self.var_labels = []
        for _ in range(4):
            vlbl = QLabel()
            vlbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vlbl.setFixedSize(140, 75)
            vlbl.setStyleSheet("background-color: #222; border: 1px solid #444; border-radius: 4px;")
            self.var_labels.append(vlbl)
            var_layout.addWidget(vlbl)

        btn_regen_vars = QPushButton("Re-roll Variations")
        btn_regen_vars.clicked.connect(self._generate_variation_strip)
        var_layout.addWidget(btn_regen_vars)
        right_layout.addWidget(variations_group, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 6)

        return widget

    def _create_generate_subtab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        # Plate Format Group
        grp_format = QGroupBox("1. License Plate Attributes")
        f_grid = QGridLayout(grp_format)

        f_grid.addWidget(QLabel("Plate Type:"), 0, 0)
        self.combo_plate_type = QComboBox()
        for type_id, disp_name in get_plate_types():
            self.combo_plate_type.addItem(disp_name, type_id)
        self.combo_plate_type.addItem("🎲 Random Plate Type (Mixed)", "RANDOM")
        f_grid.addWidget(self.combo_plate_type, 0, 1)

        f_grid.addWidget(QLabel("Consonants (d3):"), 1, 0)
        self.txt_d3 = QLineEdit()
        self.txt_d3.setPlaceholderText("e.g. กข or 1กข")
        f_grid.addWidget(self.txt_d3, 1, 1)

        f_grid.addWidget(QLabel("Numbers (d4):"), 2, 0)
        self.txt_d4 = QLineEdit()
        self.txt_d4.setPlaceholderText("e.g. 1234")
        f_grid.addWidget(self.txt_d4, 2, 1)

        f_grid.addWidget(QLabel("Province:"), 3, 0)
        self.combo_province = QComboBox()
        for p in PROVINCES:
            self.combo_province.addItem(p)
        f_grid.addWidget(self.combo_province, 3, 1)

        btn_row = QHBoxLayout()
        btn_rand = QPushButton("🎲 Randomize")
        btn_rand.setStyleSheet("background-color: #3b82f6; font-weight: bold;")
        btn_rand.clicked.connect(self.on_randomize_plate)
        btn_row.addWidget(btn_rand)

        btn_gen = QPushButton("⚡ Render Single")
        btn_gen.setStyleSheet("background-color: #10b981; font-weight: bold;")
        btn_gen.clicked.connect(self.on_render_single)
        btn_row.addWidget(btn_gen)
        f_grid.addLayout(btn_row, 4, 0, 1, 2)

        layout.addWidget(grp_format)

        # Batch Generation Group
        grp_batch = QGroupBox("2. Batch Dataset Generation")
        b_layout = QVBoxLayout(grp_batch)

        grid_b = QGridLayout()
        grid_b.addWidget(QLabel("Samples Count:"), 0, 0)
        self.spn_batch_count = QSpinBox()
        self.spn_batch_count.setRange(1, 50000)
        self.spn_batch_count.setValue(100)
        grid_b.addWidget(self.spn_batch_count, 0, 1)

        grid_b.addWidget(QLabel("Train / Val Split:"), 1, 0)
        self.lbl_split_display = QLabel("Train 85% / Val 15%")
        self.slider_split = QSlider(Qt.Orientation.Horizontal)
        self.slider_split.setRange(50, 95)
        self.slider_split.setValue(85)
        self.slider_split.valueChanged.connect(
            lambda v: self.lbl_split_display.setText(f"Train {v}% / Val {100 - v}%")
        )
        h_split = QHBoxLayout()
        h_split.addWidget(self.slider_split)
        h_split.addWidget(self.lbl_split_display)
        grid_b.addLayout(h_split, 1, 1)

        grid_b.addWidget(QLabel("Output Folder:"), 2, 0)
        h_dir = QHBoxLayout()
        default_out = str(Path(__file__).resolve().parent.parent.parent / "dataset" / "augmented_plates")
        self.txt_out_dir = QLineEdit(default_out)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_output_dir)
        h_dir.addWidget(self.txt_out_dir)
        h_dir.addWidget(btn_browse)
        grid_b.addLayout(h_dir, 2, 1)

        b_layout.addLayout(grid_b)

        self.chk_export_yolo = QCheckBox("Export YOLO BBoxes (LicensePlate=0, ThaiChar=1, Digit=2) + data.yaml")
        self.chk_export_yolo.setChecked(True)
        b_layout.addWidget(self.chk_export_yolo)

        self.btn_run_batch = QPushButton("🚀 Run Batch Generator")
        self.btn_run_batch.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 8px;")
        self.btn_run_batch.clicked.connect(self.on_start_batch_generation)
        b_layout.addWidget(self.btn_run_batch)

        self.progress_batch = QProgressBar()
        self.progress_batch.setVisible(False)
        b_layout.addWidget(self.progress_batch)

        layout.addWidget(grp_batch)
        layout.addStretch()

        scroll.setWidget(content)
        return scroll

    def _create_augment_subtab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        # Preset selection
        preset_box = QGroupBox("Augmentation Intensity Presets")
        p_layout = QHBoxLayout(preset_box)
        p_layout.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        for p in ["Medium", "Light", "Heavy", "Extreme", "None"]:
            self.combo_preset.addItem(p)
        self.combo_preset.setCurrentText("Medium")
        self.combo_preset.currentTextChanged.connect(self.on_preset_changed)
        p_layout.addWidget(self.combo_preset)

        btn_apply_aug = QPushButton("🔄 Re-Augment")
        btn_apply_aug.setStyleSheet("background-color: #3b82f6; font-weight: bold;")
        btn_apply_aug.clicked.connect(self.apply_augmentations_to_preview)
        p_layout.addWidget(btn_apply_aug)
        layout.addWidget(preset_box)

        # Build controls grouped by category
        groups = {}
        for name, reg in AUGMENTATION_REGISTRY.items():
            grp_name = reg["group"]
            if grp_name not in groups:
                groups[grp_name] = []
            groups[grp_name].append((name, reg))

        for grp_name, items in groups.items():
            grp = QGroupBox(f"{grp_name} Augmentations")
            g_layout = QGridLayout(grp)
            row = 0
            for name, reg in items:
                chk = QCheckBox(reg["display"])
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(1, 100)
                slider.setValue(40)
                lbl_val = QLabel("0.40")
                slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(f"{v/100:.2f}"))
                slider.valueChanged.connect(self.apply_augmentations_to_preview)
                chk.toggled.connect(self.apply_augmentations_to_preview)

                g_layout.addWidget(chk, row, 0)
                g_layout.addWidget(slider, row, 1)
                g_layout.addWidget(lbl_val, row, 2)

                self.aug_controls[name] = (chk, slider, lbl_val)
                row += 1
            layout.addWidget(grp)

        self.on_preset_changed("Medium")
        scroll.setWidget(content)
        return scroll

    # -----------------------------------------------------------------------
    # Tab 3: Load Existing Plate Images & Augment
    # -----------------------------------------------------------------------
    def _create_load_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        btn_load_files = QPushButton("📁 Load Image Files...")
        btn_load_files.clicked.connect(self._on_load_files)
        top_row.addWidget(btn_load_files)

        btn_load_dir = QPushButton("📂 Load Directory of Plates...")
        btn_load_dir.clicked.connect(self._on_load_directory)
        top_row.addWidget(btn_load_dir)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear_loaded)
        top_row.addWidget(btn_clear)
        layout.addLayout(top_row)

        self.table_loaded = QTableWidget(0, 3)
        self.table_loaded.setHorizontalHeaderLabels(["Filename", "Dimensions", "Path"])
        self.table_loaded.horizontalHeader().setStretchLastSection(True)
        self.table_loaded.cellClicked.connect(self._on_select_loaded_image)
        layout.addWidget(self.table_loaded)

        aug_load_box = QGroupBox("Batch Augmentation for Loaded Plates")
        al_grid = QGridLayout(aug_load_box)

        al_grid.addWidget(QLabel("Copies per Image:"), 0, 0)
        self.spn_copies = QSpinBox()
        self.spn_copies.setRange(1, 100)
        self.spn_copies.setValue(5)
        al_grid.addWidget(self.spn_copies, 0, 1)

        al_grid.addWidget(QLabel("Output Directory:"), 1, 0)
        h_dir = QHBoxLayout()
        default_existing_out = str(Path(__file__).resolve().parent.parent.parent / "dataset" / "augmented_existing")
        self.txt_existing_out = QLineEdit(default_existing_out)
        btn_browse_ex = QPushButton("Browse...")
        btn_browse_ex.clicked.connect(self._on_browse_existing_out_dir)
        h_dir.addWidget(self.txt_existing_out)
        h_dir.addWidget(btn_browse_ex)
        al_grid.addLayout(h_dir, 1, 1)

        self.btn_run_existing_aug = QPushButton("✨ Augment All Loaded Plates")
        self.btn_run_existing_aug.setStyleSheet("background-color: #8b5cf6; color: white; font-weight: bold; padding: 8px;")
        self.btn_run_existing_aug.clicked.connect(self.on_start_existing_batch)
        al_grid.addWidget(self.btn_run_existing_aug, 2, 0, 1, 2)

        self.progress_existing = QProgressBar()
        self.progress_existing.setVisible(False)
        al_grid.addWidget(self.progress_existing, 3, 0, 1, 2)

        layout.addWidget(aug_load_box)
        return tab

    # -----------------------------------------------------------------------
    # Dark Theme
    # -----------------------------------------------------------------------
    def _apply_dark_theme(self):
        style = """
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            font-size: 12px;
        }
        QGroupBox {
            border: 1px solid #3f3f46;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 12px;
            font-weight: bold;
            color: #93c5fd;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: #27272a;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            padding: 5px;
            color: #fafafa;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #3b82f6;
        }
        QPushButton {
            background-color: #3f3f46;
            color: #fafafa;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #52525b;
        }
        QPushButton:pressed {
            background-color: #27272a;
        }
        QTabWidget::pane {
            border: 1px solid #3f3f46;
            border-radius: 4px;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background-color: #27272a;
            color: #a1a1aa;
            padding: 9px 18px;
            font-weight: bold;
            border-top-left-radius: 5px;
            border-top-right-radius: 5px;
            margin-right: 3px;
        }
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            color: #38bdf8;
            border-top: 3px solid #38bdf8;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #3f3f46;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #3b82f6;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QTableWidget {
            background-color: #27272a;
            border: 1px solid #3f3f46;
            gridline-color: #3f3f46;
            color: #fafafa;
        }
        QHeaderView::section {
            background-color: #18181b;
            color: #a1a1aa;
            padding: 4px;
            border: 1px solid #27272a;
        }
        QProgressBar {
            border: 1px solid #3f3f46;
            border-radius: 4px;
            text-align: center;
            color: white;
            background-color: #27272a;
        }
        QProgressBar::chunk {
            background-color: #3b82f6;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(style)

    # -----------------------------------------------------------------------
    # Interactive Canvas Integration
    # -----------------------------------------------------------------------
    def _on_custom_composite_received(self, pil_img: Image.Image, bboxes: List[Dict], meta: Dict):
        """Called when user clicks 'Send to Augmentation Pipeline' from the Interactive Canvas."""
        self.current_base_img = pil_img
        self.current_base_bboxes = bboxes
        self.current_metadata = meta

        self.lbl_meta.setText(f"Custom Interactive Plate ({meta.get('width', '?')}x{meta.get('height', '?')})")
        self.apply_augmentations_to_preview()
        # Switch to pipeline tab
        self.main_tabs.setCurrentIndex(1)
        self.status_bar.showMessage("Loaded custom interactive plate into Augmentation Pipeline.")

    # -----------------------------------------------------------------------
    # Logic & Event Handlers
    # -----------------------------------------------------------------------
    def get_current_aug_config(self) -> dict:
        config = {}
        for name, (chk, slider, _) in self.aug_controls.items():
            config[name] = {
                "enabled": chk.isChecked(),
                "intensity": slider.value() / 100.0,
            }
        return config

    def on_preset_changed(self, preset_name: str):
        preset = PRESETS.get(preset_name, {})
        for name, (chk, slider, _) in self.aug_controls.items():
            if name in preset:
                chk.setChecked(preset[name].get("enabled", True))
                val = int(preset[name].get("intensity", 0.4) * 100)
                slider.setValue(val)
            else:
                chk.setChecked(False)
        self.apply_augmentations_to_preview()

    def on_randomize_plate(self):
        curr_type = self.combo_plate_type.currentData()
        if curr_type == "RANDOM" or not curr_type:
            types = list(PLATE_CONFIGS.keys())
            curr_type = random.choice(types)

        img, bboxes, meta = generate_random_plate(curr_type)
        self.txt_d3.setText(meta["d3"])
        self.txt_d4.setText(meta["d4"])
        idx = self.combo_province.findText(meta["province"])
        if idx >= 0:
            self.combo_province.setCurrentIndex(idx)

        self.current_base_img = img
        self.current_base_bboxes = bboxes
        self.current_metadata = meta

        self.lbl_meta.setText(f"Plate: {meta['d3']} {meta['d4']} ({meta['province']}) — [{curr_type}]")
        self.apply_augmentations_to_preview()

    def on_render_single(self):
        curr_type = self.combo_plate_type.currentData()
        if curr_type == "RANDOM":
            curr_type = "WhiteBlackCar2Line"

        d3 = self.txt_d3.text().strip() or "กข"
        d4 = self.txt_d4.text().strip() or "1234"
        province = self.combo_province.currentText()

        img, bboxes, meta = render_plate(curr_type, d3, d4, province)
        self.current_base_img = img
        self.current_base_bboxes = bboxes
        self.current_metadata = meta

        self.lbl_meta.setText(f"Plate: {d3} {d4} ({province}) — [{curr_type}]")
        self.apply_augmentations_to_preview()

    def apply_augmentations_to_preview(self):
        if self.current_base_img is None:
            return

        pipeline = AugmentationPipeline(self.get_current_aug_config())
        aug_img, aug_bboxes = pipeline.apply_all(self.current_base_img, self.current_base_bboxes)

        self.current_aug_img = aug_img
        self.current_aug_bboxes = aug_bboxes

        self._refresh_previews()
        self._generate_variation_strip()

    def _refresh_previews(self):
        show_boxes = self.chk_show_bboxes.isChecked()

        # Render Base Preview
        if self.current_base_img:
            pix_orig = pil_to_qpixmap(self.current_base_img)
            if show_boxes and self.current_base_bboxes:
                pix_orig = self._draw_bboxes_on_pixmap(pix_orig, self.current_base_bboxes)
            self.lbl_preview_orig.setPixmap(
                pix_orig.scaled(self.lbl_preview_orig.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

        # Render Augmented Preview
        if self.current_aug_img:
            pix_aug = pil_to_qpixmap(self.current_aug_img)
            if show_boxes and self.current_aug_bboxes:
                pix_aug = self._draw_bboxes_on_pixmap(pix_aug, self.current_aug_bboxes)
            self.lbl_preview_aug.setPixmap(
                pix_aug.scaled(self.lbl_preview_aug.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    def _draw_bboxes_on_pixmap(self, pixmap: QPixmap, bboxes: List[Dict]) -> QPixmap:
        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Full plate bounding box in green
        pen_plate = QPen(QColor(16, 185, 129), 2)
        painter.setPen(pen_plate)
        painter.drawRect(1, 1, result.width() - 2, result.height() - 2)

        # Character bboxes in cyan/gold
        pen_char = QPen(QColor(56, 189, 248), 2)
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)

        for b in bboxes:
            x1, y1, x2, y2 = b["bbox"]
            ch = b.get("char", "")
            painter.setPen(pen_char)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            if ch:
                painter.setPen(QPen(QColor(250, 204, 21)))
                painter.drawText(x1 + 2, max(12, y1 - 2), ch)

        painter.end()
        return result

    def _generate_variation_strip(self):
        if self.current_base_img is None:
            return
        pipeline = AugmentationPipeline(self.get_current_aug_config())
        for lbl in self.var_labels:
            var_img, _ = pipeline.apply(self.current_base_img, [])
            pix = pil_to_qpixmap(var_img)
            lbl.setPixmap(
                pix.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    # -----------------------------------------------------------------------
    # Batch Processing
    # -----------------------------------------------------------------------
    def _on_browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.txt_out_dir.text())
        if d:
            self.txt_out_dir.setText(d)

    def _on_browse_existing_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.txt_existing_out.text())
        if d:
            self.txt_existing_out.setText(d)

    def on_start_batch_generation(self):
        count = self.spn_batch_count.value()
        out_dir = self.txt_out_dir.text().strip()
        val_split = (100 - self.slider_split.value()) / 100.0
        plate_type = self.combo_plate_type.currentData()
        aug_cfg = self.get_current_aug_config()
        export_yolo = self.chk_export_yolo.isChecked()

        if not out_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select an output directory.")
            return

        self.btn_run_batch.setEnabled(False)
        self.progress_batch.setVisible(True)
        self.progress_batch.setRange(0, count)
        self.progress_batch.setValue(0)

        self.batch_worker = SyntheticBatchWorker(
            count=count,
            out_dir=out_dir,
            val_split=val_split,
            plate_type_choice=plate_type,
            aug_config=aug_cfg,
            export_yolo=export_yolo,
        )
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.finished.connect(self._on_batch_finished)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()

    def _on_batch_progress(self, current: int, total: int, msg: str):
        self.progress_batch.setValue(current)
        self.status_bar.showMessage(msg)

    def _on_batch_finished(self, count: int, elapsed: float, out_dir: str):
        self.btn_run_batch.setEnabled(True)
        self.progress_batch.setVisible(False)
        self.status_bar.showMessage(f"Successfully generated {count} plates in {elapsed:.1f}s.")
        QMessageBox.information(
            self,
            "Batch Complete",
            f"Successfully generated {count:,} augmented license plates in {elapsed:.2f} seconds!\n\nSaved to:\n{out_dir}",
        )

    def _on_batch_error(self, err_msg: str):
        self.btn_run_batch.setEnabled(True)
        self.progress_batch.setVisible(False)
        self.status_bar.showMessage(f"Error: {err_msg}")
        QMessageBox.critical(self, "Batch Generation Failed", f"An error occurred:\n{err_msg}")

    # -----------------------------------------------------------------------
    # Load & Augment Existing Plates
    # -----------------------------------------------------------------------
    def _on_load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Plate Images", "", "Image Files (*.jpg *.png *.jpeg *.bmp)"
        )
        if files:
            self._add_loaded_paths([Path(f) for f in files])

    def _on_load_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder of Plate Images")
        if d:
            p = Path(d)
            imgs = [f for f in p.glob("*.*") if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
            self._add_loaded_paths(imgs)

    def _add_loaded_paths(self, paths: List[Path]):
        for p in paths:
            if p not in self.loaded_image_paths:
                self.loaded_image_paths.append(p)
                row = self.table_loaded.rowCount()
                self.table_loaded.insertRow(row)
                self.table_loaded.setItem(row, 0, QTableWidgetItem(p.name))
                try:
                    with Image.open(p) as im:
                        dim_str = f"{im.width}x{im.height}"
                except Exception:
                    dim_str = "?"
                self.table_loaded.setItem(row, 1, QTableWidgetItem(dim_str))
                self.table_loaded.setItem(row, 2, QTableWidgetItem(str(p)))

        self.status_bar.showMessage(f"Loaded {len(self.loaded_image_paths)} plate image(s).")

    def _on_clear_loaded(self):
        self.loaded_image_paths.clear()
        self.table_loaded.setRowCount(0)
        self.status_bar.showMessage("Cleared loaded images.")

    def _on_select_loaded_image(self, row: int, col: int):
        if row < len(self.loaded_image_paths):
            p = self.loaded_image_paths[row]
            try:
                img = Image.open(p).convert("RGB")
                self.current_base_img = img
                self.current_base_bboxes = []
                self.lbl_meta.setText(f"Loaded Image: {p.name} ({img.width}x{img.height})")
                self.apply_augmentations_to_preview()
            except Exception as e:
                self.status_bar.showMessage(f"Failed to open {p.name}: {e}")

    def on_start_existing_batch(self):
        if not self.loaded_image_paths:
            QMessageBox.warning(self, "No Images", "Please load existing plate images first.")
            return

        out_dir = self.txt_existing_out.text().strip()
        copies = self.spn_copies.value()
        aug_cfg = self.get_current_aug_config()

        self.btn_run_existing_aug.setEnabled(False)
        self.progress_existing.setVisible(True)
        self.progress_existing.setRange(0, len(self.loaded_image_paths) * copies)
        self.progress_existing.setValue(0)

        self.existing_worker = ExistingBatchWorker(
            image_paths=self.loaded_image_paths,
            out_dir=out_dir,
            copies_per_img=copies,
            aug_config=aug_cfg,
        )
        self.existing_worker.progress.connect(lambda cur, tot, msg: (self.progress_existing.setValue(cur), self.status_bar.showMessage(msg)))
        self.existing_worker.finished.connect(self._on_existing_finished)
        self.existing_worker.error.connect(self._on_batch_error)
        self.existing_worker.start()

    def _on_existing_finished(self, count: int, elapsed: float, out_dir: str):
        self.btn_run_existing_aug.setEnabled(True)
        self.progress_existing.setVisible(False)
        self.status_bar.showMessage(f"Finished augmenting {count} images in {elapsed:.1f}s.")
        QMessageBox.information(
            self,
            "Augmentation Complete",
            f"Successfully augmented {count:,} images in {elapsed:.2f} seconds!\n\nSaved to:\n{out_dir}",
        )
