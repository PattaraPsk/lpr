"""
Main Application Window for Thai License Plate (LPR) Annotation & Review.
Updated to support AI Reader (AI Reader) inference, live field comparison (d3, d4, province),
and selective/bulk replacement of predicted plate elements.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QFrame, QGroupBox, QGridLayout, QComboBox,
    QInputDialog, QCheckBox, QScrollArea, QProgressDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QKeySequence, QIcon, QShortcut, QColor

from .session_manager import SessionManager, ImageItem
from .image_viewer import ImageViewerWidget
from .searchable_combo import SearchableProvinceComboBox
from .plate_type_combo import PlateTypeComboBox
from .check_view import CheckViewDialog
from .provinces import normalize_thai_script
from .reader_worker import AISingleReaderWorker, AIBatchReaderWorker
from .training_dialog import TrainingDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thai LPR Dataset Annotator & Reviewer [Edge AI Enabled]")
        self.resize(1400, 920)
        self.setMinimumSize(1020, 680)
        
        self.session = SessionManager()
        
        # State for active AI Reader predictions
        self._active_worker: Optional[AISingleReaderWorker] = None
        self._batch_worker: Optional[AIBatchReaderWorker] = None
        self._latest_ai_d3: str = ""
        self._latest_ai_d4: str = ""
        self._latest_ai_prov: str = ""
        self._latest_ai_conf: float = 0.0
        
        # State for YOLO bounding box selection & editing
        self.selected_box_index: Optional[int] = None
        self._syncing_coords: bool = False
        
        self._init_ui()
        self._setup_shortcuts()
        self._apply_styles()
        
    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        
        # 1. Top Header Bar
        header_bar = self._create_header_bar()
        main_layout.addWidget(header_bar)
        
        # 2. Central Splitter (Left: Viewer, Right: Scrollable Edit Panel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        
        self.image_viewer = ImageViewerWidget(self)
        self.image_viewer.box_clicked.connect(self._on_viewer_box_clicked)
        splitter.addWidget(self.image_viewer)
        
        self.edit_panel = self._create_edit_panel()
        
        # Wrap edit panel in scroll area for small display height compatibility
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(self.edit_panel)
        splitter.addWidget(scroll_area)
        
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter, 1)
        
        # 3. Bottom Status Bar & Progress
        bottom_bar = self._create_bottom_bar()
        main_layout.addWidget(bottom_bar)

    def _create_header_bar(self) -> QWidget:
        header = QFrame(self)
        header.setObjectName("headerCard")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(12, 8, 12, 8)
        h_layout.setSpacing(10)
        
        # Open Folder Button
        self.btn_open_folder = QPushButton("📁 Open Folder")
        self.btn_open_folder.setObjectName("btnOpenFolder")
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        h_layout.addWidget(self.btn_open_folder)
        
        # Subfolder Grouping Selector
        lbl_sf = QLabel("Folder:")
        lbl_sf.setStyleSheet("font-weight: bold; color: #a78bfa; margin-left: 4px;")
        h_layout.addWidget(lbl_sf)
        
        self.combo_subfolder = QComboBox()
        self.combo_subfolder.setObjectName("comboSubfolder")
        self.combo_subfolder.addItem("All Subfolders", userData="(All)")
        self.combo_subfolder.currentIndexChanged.connect(self._on_subfolder_changed)
        h_layout.addWidget(self.combo_subfolder)

        # Class Filter Selector (For YOLO Object Detection Mode)
        self.lbl_filter_class = QLabel("Filter Class:")
        self.lbl_filter_class.setStyleSheet("font-weight: bold; color: #38bdf8; margin-left: 6px;")
        h_layout.addWidget(self.lbl_filter_class)

        self.combo_class_filter = QComboBox()
        self.combo_class_filter.setObjectName("comboClassFilter")
        self.combo_class_filter.addItem("(All Classes)", userData=-1)
        self.combo_class_filter.currentIndexChanged.connect(self._on_class_filter_changed)
        h_layout.addWidget(self.combo_class_filter)
        
        # Status Filter Selector (All Images, Pending, Not Plate, Cannot Read, Approved, etc.)
        self.lbl_filter_status = QLabel("Filter:")
        self.lbl_filter_status.setStyleSheet("font-weight: bold; color: #38bdf8; margin-left: 6px;")
        h_layout.addWidget(self.lbl_filter_status)

        self.combo_status_filter = QComboBox()
        self.combo_status_filter.setObjectName("comboStatusFilter")
        self.combo_status_filter.setMinimumWidth(185)
        self.combo_status_filter.addItem("All Images", userData="all")
        self.combo_status_filter.addItem("⏳ Pending (To Annotate)", userData="pending")
        self.combo_status_filter.addItem("🚫 Not Plate (+)", userData="not_plate")
        self.combo_status_filter.addItem("❓ Cannot Read (-)", userData="cannot_read")
        self.combo_status_filter.addItem("✅ Approved", userData="approved")
        self.combo_status_filter.addItem("⚡ AI Diff", userData="ai_diff")
        self.combo_status_filter.addItem("📑 Duplicates", userData="dup")
        self.combo_status_filter.currentIndexChanged.connect(self._on_status_filter_changed)
        h_layout.addWidget(self.combo_status_filter)
        
        # Dataset Mode Indicator Badge
        self.badge_mode = QLabel("🔤 Plate OCR")
        self.badge_mode.setObjectName("badgeMode")
        self.badge_mode.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        h_layout.addWidget(self.badge_mode)

        # Mode Switch Button
        self.btn_switch_mode = QPushButton("🔄 Toggle View")
        self.btn_switch_mode.setObjectName("btnSwitchMode")
        self.btn_switch_mode.setToolTip("Toggle between YOLO Object Detection View and OCR Plate View")
        self.btn_switch_mode.clicked.connect(self._on_toggle_view_mode)
        h_layout.addWidget(self.btn_switch_mode)
        
        # Remove _dup Images Button
        self.btn_clean_dup = QPushButton("🧹 Remove _dup")
        self.btn_clean_dup.setObjectName("btnCleanDup")
        self.btn_clean_dup.setToolTip("Find and permanently remove images with '_dup' in their filename")
        self.btn_clean_dup.clicked.connect(self._on_remove_dup_clicked)
        h_layout.addWidget(self.btn_clean_dup)
        
        # Batch AI Read Button
        self.btn_batch_ai = QPushButton("🤖 AI Read All")
        self.btn_batch_ai.setObjectName("btnAiReadAll")
        self.btn_batch_ai.setToolTip("Run AI Reader reader across all images in active subfolder or folder")
        self.btn_batch_ai.clicked.connect(self._on_batch_ai_clicked)
        h_layout.addWidget(self.btn_batch_ai)

        # Train Models Studio Button
        self.btn_train_models = QPushButton("🚀 Train Models")
        self.btn_train_models.setObjectName("btnTrainModels")
        self.btn_train_models.setStyleSheet("background-color: #8b5cf6; color: white; font-weight: bold; border-radius: 6px; padding: 6px 14px;")
        self.btn_train_models.setToolTip("Open Model Training Studio to fine-tune Plate Detector (YOLO) or Plate Reader (OCR)")
        self.btn_train_models.clicked.connect(self._on_train_models_clicked)
        h_layout.addWidget(self.btn_train_models)
        
        h_layout.addStretch()
        
        # Stats Badges
        self.badge_total = QLabel("Total: 0")
        self.badge_total.setObjectName("badgeNeutral")
        h_layout.addWidget(self.badge_total)
        
        self.badge_left = QLabel("Left: 0")
        self.badge_left.setObjectName("badgeWarning")
        h_layout.addWidget(self.badge_left)
        
        self.badge_approved = QLabel("Approved: 0")
        self.badge_approved.setObjectName("badgeSuccess")
        h_layout.addWidget(self.badge_approved)
        
        self.badge_not_plate = QLabel("Not Plate: 0")
        self.badge_not_plate.setObjectName("badgeBlue")
        h_layout.addWidget(self.badge_not_plate)
        
        self.badge_cannot_read = QLabel("Cannot Read: 0")
        self.badge_cannot_read.setObjectName("badgeDanger")
        h_layout.addWidget(self.badge_cannot_read)
        
        # Interactive badge shortcuts to filter list of images
        for badge, f_code, tip in [
            (self.badge_total, "all", "Click to filter All Images (Alt+A)"),
            (self.badge_left, "pending", "Click to filter Pending / Unannotated (Alt+P)"),
            (self.badge_approved, "approved", "Click to filter Approved Plates"),
            (self.badge_not_plate, "not_plate", "Click to filter Not Plate [+] (Alt+N)"),
            (self.badge_cannot_read, "cannot_read", "Click to filter Cannot Read [-] (Alt+D)"),
        ]:
            badge.setCursor(Qt.CursorShape.PointingHandCursor)
            badge.setToolTip(tip)
            badge.mousePressEvent = lambda event, c=f_code: self._set_status_filter(c)
        
        # Check Menu Button
        self.btn_check_menu = QPushButton("📋 Check Menu")
        self.btn_check_menu.setObjectName("btnCheckMenu")
        self.btn_check_menu.setToolTip("Open Review List to inspect skipped/approved images and AI differences")
        self.btn_check_menu.clicked.connect(self._on_open_check_menu)
        h_layout.addWidget(self.btn_check_menu)

        # Dataset Staging Button
        self.btn_export_staging = QPushButton("📦 Stage Approved")
        self.btn_export_staging.setObjectName("btnExportStaging")
        self.btn_export_staging.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_export_staging.setToolTip("Export/sync all approved plates into dataset staging folder (organized by type_d3_d4_province for reading task) waiting for dataset building")
        self.btn_export_staging.clicked.connect(self._on_export_staging_clicked)
        h_layout.addWidget(self.btn_export_staging)
        
        # OWOD Studio Button
        self.btn_owod_studio = QPushButton("⚡ OWOD Studio")
        self.btn_owod_studio.setObjectName("btnOWOD")
        self.btn_owod_studio.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_owod_studio.setToolTip("Open Open-World Object Detection & Model Surgery Studio")
        self.btn_owod_studio.clicked.connect(self._on_open_owod_studio)
        h_layout.addWidget(self.btn_owod_studio)
        
        return header

    def _create_edit_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("editPanelCard")
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(520)
        
        p_layout = QVBoxLayout(panel)
        p_layout.setContentsMargins(16, 16, 16, 16)
        p_layout.setSpacing(12)
        
        # Subfolder & Filename header
        self.lbl_current_subfolder = QLabel("Folder: (None)")
        self.lbl_current_subfolder.setStyleSheet("color: #a78bfa; font-weight: bold; font-size: 12px;")
        p_layout.addWidget(self.lbl_current_subfolder)
        
        self.lbl_current_filename = QLabel("No Image Selected")
        self.lbl_current_filename.setObjectName("lblFileName")
        self.lbl_current_filename.setWordWrap(True)
        p_layout.addWidget(self.lbl_current_filename)
        
        self.lbl_current_status = QLabel("Status: Ready")
        self.lbl_current_status.setObjectName("lblStatus")
        p_layout.addWidget(self.lbl_current_status)
        
        # ---------------------------------------------------------------------
        # AI Reader (AI Reader) & Comparison Box
        # ---------------------------------------------------------------------
        self.gb_ai = self._create_ai_comparison_box()
        p_layout.addWidget(self.gb_ai)

        # ---------------------------------------------------------------------
        # YOLO Bounding Box Inspector (Object Detection Mode)
        # ---------------------------------------------------------------------
        self.gb_yolo_inspector = self._create_yolo_inspector_box()
        p_layout.addWidget(self.gb_yolo_inspector)
        
        # ---------------------------------------------------------------------
        # Plate Structure Editor
        # ---------------------------------------------------------------------
        gb_edit = QGroupBox("Plate Information")
        gb_edit.setObjectName("groupBoxEdit")
        grid = QGridLayout(gb_edit)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(8)
        grid.setContentsMargins(12, 16, 12, 12)
        
        # 1. Base Name field (Editable)
        lbl_name = QLabel("Base Name:")
        lbl_name.setStyleSheet("font-weight: 500;")
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Prefix / Identifier")
        self.txt_name.returnPressed.connect(self._on_approve_clicked)
        grid.addWidget(lbl_name, 0, 0)
        grid.addWidget(self.txt_name, 0, 1)
        
        # 2. Plate Type field
        lbl_type = QLabel("Plate Type:")
        lbl_type.setStyleSheet("font-weight: bold; color: #a78bfa;")
        self.combo_plate_type = PlateTypeComboBox()
        self.combo_plate_type.setObjectName("comboPlateType")
        grid.addWidget(lbl_type, 1, 0)
        grid.addWidget(self.combo_plate_type, 1, 1)
        
        # 3. d3 field (characters)
        lbl_d3 = QLabel("d3 (Letters/Type):")
        lbl_d3.setStyleSheet("font-weight: bold; color: #60a5fa;")
        self.txt_d3 = QLineEdit()
        self.txt_d3.setPlaceholderText("e.g. 1กข, ขก, +")
        self.txt_d3.setObjectName("txtPlateField")
        self.txt_d3.textChanged.connect(self._on_manual_field_changed)
        self.txt_d3.returnPressed.connect(self._on_approve_clicked)
        grid.addWidget(lbl_d3, 2, 0)
        grid.addWidget(self.txt_d3, 2, 1)
        
        # 4. d4 field (numbers)
        lbl_d4 = QLabel("d4 (Number):")
        lbl_d4.setStyleSheet("font-weight: bold; color: #60a5fa;")
        self.txt_d4 = QLineEdit()
        self.txt_d4.setPlaceholderText("e.g. 1234, 9999, +")
        self.txt_d4.setObjectName("txtPlateField")
        self.txt_d4.textChanged.connect(self._on_manual_field_changed)
        self.txt_d4.returnPressed.connect(self._on_approve_clicked)
        grid.addWidget(lbl_d4, 3, 0)
        grid.addWidget(self.txt_d4, 3, 1)
        
        # 5. province field
        lbl_province = QLabel("Province:")
        lbl_province.setStyleSheet("font-weight: bold; color: #60a5fa;")
        self.combo_province = SearchableProvinceComboBox()
        self.combo_province.setObjectName("comboProvince")
        self.combo_province.currentTextChanged.connect(self._on_manual_field_changed)
        grid.addWidget(lbl_province, 4, 0)
        grid.addWidget(self.combo_province, 4, 1)
        
        self.gb_plate_info = gb_edit
        p_layout.addWidget(self.gb_plate_info)
        
        # File Manage Row (Rename & Delete)
        file_actions_box = QHBoxLayout()
        file_actions_box.setSpacing(8)
        
        self.btn_rename_file = QPushButton("✏️ Rename File (F2)")
        self.btn_rename_file.setObjectName("btnRenameFile")
        self.btn_rename_file.clicked.connect(self._on_rename_clicked)
        file_actions_box.addWidget(self.btn_rename_file)
        
        self.btn_delete_file = QPushButton("🗑️ Delete Image (Del)")
        self.btn_delete_file.setObjectName("btnDeleteFile")
        self.btn_delete_file.clicked.connect(self._on_delete_clicked)
        file_actions_box.addWidget(self.btn_delete_file)
        
        p_layout.addLayout(file_actions_box)
        
        # Action Buttons Group
        lbl_actions = QLabel("Quick Actions:")
        lbl_actions.setStyleSheet("font-weight: bold; color: #a0a0b0; margin-top: 4px;")
        p_layout.addWidget(lbl_actions)

        # Dataset Staging Checkbox & Folder Link
        staging_row = QHBoxLayout()
        self.chk_auto_staging = QCheckBox("Auto-stage Approved")
        self.chk_auto_staging.setChecked(True)
        self.chk_auto_staging.setStyleSheet("color: #38bdf8; font-weight: 500; font-size: 11px;")
        self.chk_auto_staging.setToolTip("Automatically copy approved plates to dataset staging folder (folders named type_d3_d4_province for reading task) upon Approve")
        self.chk_auto_staging.stateChanged.connect(lambda s: setattr(self.session, "auto_staging", bool(s)))
        staging_row.addWidget(self.chk_auto_staging)
        
        btn_open_staging = QPushButton("📂 Open Staging")
        btn_open_staging.setStyleSheet("font-size: 11px; padding: 2px 6px; background-color: #334155;")
        btn_open_staging.setToolTip("Open dataset staging directory in Windows Explorer")
        btn_open_staging.clicked.connect(self._on_open_staging_folder)
        staging_row.addWidget(btn_open_staging)
        p_layout.addLayout(staging_row)
        
        # Correct / Approve Button
        self.btn_approve = QPushButton("✅ Correct / Approve (Enter)")
        self.btn_approve.setObjectName("btnApprove")
        self.btn_approve.clicked.connect(self._on_approve_clicked)
        p_layout.addWidget(self.btn_approve)
        
        # Not Plate Button
        self.btn_not_plate = QPushButton("🚫 Not Plate [+] (Ctrl+N)")
        self.btn_not_plate.setObjectName("btnNotPlate")
        self.btn_not_plate.setToolTip("Fill all boxes with '+' for Not Plate. Save by clicking Approve.")
        self.btn_not_plate.clicked.connect(self._on_not_plate_clicked)
        p_layout.addWidget(self.btn_not_plate)
        
        # Cannot Read Button
        self.btn_cannot_read = QPushButton("❓ Cannot Read [-] (Ctrl+D)")
        self.btn_cannot_read.setObjectName("btnCannotRead")
        self.btn_cannot_read.setToolTip("Fill all boxes with '-' for Cannot Read. Save by clicking Approve.")
        self.btn_cannot_read.clicked.connect(self._on_cannot_read_clicked)
        p_layout.addWidget(self.btn_cannot_read)
        
        # Skip Button
        self.btn_skip = QPushButton("⏭️ Skip (Ctrl+S)")
        self.btn_skip.setObjectName("btnSkip")
        self.btn_skip.clicked.connect(self._on_skip_clicked)
        p_layout.addWidget(self.btn_skip)
        
        p_layout.addStretch()
        
        # Navigation Bar
        nav_box = QHBoxLayout()
        nav_box.setSpacing(8)
        
        self.btn_back = QPushButton("◀ Back")
        self.btn_back.setObjectName("btnNav")
        self.btn_back.clicked.connect(self._on_back_clicked)
        nav_box.addWidget(self.btn_back)
        
        self.lbl_nav_counter = QLabel("0 / 0")
        self.lbl_nav_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_nav_counter.setStyleSheet("font-weight: bold; color: #ffffff;")
        nav_box.addWidget(self.lbl_nav_counter, 1)
        
        self.btn_next = QPushButton("Next ▶")
        self.btn_next.setObjectName("btnNav")
        self.btn_next.clicked.connect(self._on_next_clicked)
        nav_box.addWidget(self.btn_next)
        
        p_layout.addLayout(nav_box)
        
        # Filtered Image List Jump Selector
        jump_box = QHBoxLayout()
        jump_box.setSpacing(6)
        lbl_jump = QLabel("Image List:")
        lbl_jump.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        jump_box.addWidget(lbl_jump)
        
        self.combo_image_jump = QComboBox()
        self.combo_image_jump.setObjectName("comboImageJump")
        self.combo_image_jump.setToolTip("Quickly jump to any image in the current filter list (Not Plate, Cannot Read, etc.)")
        self.combo_image_jump.currentIndexChanged.connect(self._on_jump_image_selected)
        jump_box.addWidget(self.combo_image_jump, 1)
        p_layout.addLayout(jump_box)
        
        return panel

    def _create_ai_comparison_box(self) -> QGroupBox:
        """Creates the interactive AI Reader (AI Reader) and field comparison widget."""
        gb_ai = QGroupBox("🤖 AI Reader (AI Reader)")
        gb_ai.setObjectName("groupBoxAI")
        ai_layout = QVBoxLayout(gb_ai)
        ai_layout.setContentsMargins(12, 10, 12, 10)
        ai_layout.setSpacing(8)
        
        # Top AI Controls Bar
        ai_top = QHBoxLayout()
        self.chk_auto_ai = QCheckBox("Auto AI Read")
        self.chk_auto_ai.setChecked(True)
        self.chk_auto_ai.setToolTip("Automatically run AI Reader model on each image change")
        ai_top.addWidget(self.chk_auto_ai)
        
        ai_top.addStretch()
        
        self.btn_ai_read = QPushButton("⚡ Read Plate")
        self.btn_ai_read.setObjectName("btnAiRead")
        self.btn_ai_read.setToolTip("Run AI Reader inference on current image crop")
        self.btn_ai_read.clicked.connect(self._on_manual_ai_read_clicked)
        ai_top.addWidget(self.btn_ai_read)
        
        ai_layout.addLayout(ai_top)
        
        # Status / Confidence Badge
        self.lbl_ai_status = QLabel("AI Status: Idle")
        self.lbl_ai_status.setObjectName("lblAiStatus")
        self.lbl_ai_status.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")
        ai_layout.addWidget(self.lbl_ai_status)
        
        # Comparison Table: Checkbox | Field | Current | ➜ | AI Predicted | Action
        comp_grid = QGridLayout()
        comp_grid.setVerticalSpacing(6)
        comp_grid.setHorizontalSpacing(6)
        comp_grid.setContentsMargins(0, 4, 0, 4)
        
        hdr_style = "color: #94a3b8; font-size: 11px; font-weight: bold;"
        lbl_h_sel = QLabel("Sel")
        lbl_h_sel.setStyleSheet(hdr_style)
        lbl_h_field = QLabel("Field")
        lbl_h_field.setStyleSheet(hdr_style)
        lbl_h_curr = QLabel("Current")
        lbl_h_curr.setStyleSheet(hdr_style)
        lbl_h_ai = QLabel("AI (AI Reader)")
        lbl_h_ai.setStyleSheet(hdr_style)
        lbl_h_act = QLabel("Action")
        lbl_h_act.setStyleSheet(hdr_style)
        
        comp_grid.addWidget(lbl_h_sel, 0, 0, Qt.AlignmentFlag.AlignCenter)
        comp_grid.addWidget(lbl_h_field, 0, 1)
        comp_grid.addWidget(lbl_h_curr, 0, 2)
        comp_grid.addWidget(lbl_h_ai, 0, 3)
        comp_grid.addWidget(lbl_h_act, 0, 4, Qt.AlignmentFlag.AlignCenter)
        
        # Row 1: d3 (Letters/Prefix)
        self.chk_sel_d3 = QCheckBox()
        self.chk_sel_d3.setChecked(True)
        self.chk_sel_d3.setToolTip("Include d3 in 'Replace Selected'")
        lbl_f_d3 = QLabel("d3")
        lbl_f_d3.setStyleSheet("font-weight: bold; color: #60a5fa; font-size: 12px;")
        self.lbl_comp_curr_d3 = QLabel("-")
        self.lbl_comp_curr_d3.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        self.lbl_comp_ai_d3 = QLabel("-")
        self.lbl_comp_ai_d3.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_apply_d3 = QPushButton("⬅ d3")
        self.btn_apply_d3.setObjectName("btnApplySingle")
        self.btn_apply_d3.setToolTip("Replace current d3 with AI prediction")
        self.btn_apply_d3.clicked.connect(self._on_apply_ai_d3)
        
        comp_grid.addWidget(self.chk_sel_d3, 1, 0, Qt.AlignmentFlag.AlignCenter)
        comp_grid.addWidget(lbl_f_d3, 1, 1)
        comp_grid.addWidget(self.lbl_comp_curr_d3, 1, 2)
        comp_grid.addWidget(self.lbl_comp_ai_d3, 1, 3)
        comp_grid.addWidget(self.btn_apply_d3, 1, 4)
        
        # Row 2: d4 (Numbers)
        self.chk_sel_d4 = QCheckBox()
        self.chk_sel_d4.setChecked(True)
        self.chk_sel_d4.setToolTip("Include d4 in 'Replace Selected'")
        lbl_f_d4 = QLabel("d4")
        lbl_f_d4.setStyleSheet("font-weight: bold; color: #60a5fa; font-size: 12px;")
        self.lbl_comp_curr_d4 = QLabel("-")
        self.lbl_comp_curr_d4.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        self.lbl_comp_ai_d4 = QLabel("-")
        self.lbl_comp_ai_d4.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_apply_d4 = QPushButton("⬅ d4")
        self.btn_apply_d4.setObjectName("btnApplySingle")
        self.btn_apply_d4.setToolTip("Replace current d4 with AI prediction")
        self.btn_apply_d4.clicked.connect(self._on_apply_ai_d4)
        
        comp_grid.addWidget(self.chk_sel_d4, 2, 0, Qt.AlignmentFlag.AlignCenter)
        comp_grid.addWidget(lbl_f_d4, 2, 1)
        comp_grid.addWidget(self.lbl_comp_curr_d4, 2, 2)
        comp_grid.addWidget(self.lbl_comp_ai_d4, 2, 3)
        comp_grid.addWidget(self.btn_apply_d4, 2, 4)
        
        # Row 3: province
        self.chk_sel_prov = QCheckBox()
        self.chk_sel_prov.setChecked(True)
        self.chk_sel_prov.setToolTip("Include Province in 'Replace Selected'")
        lbl_f_prov = QLabel("Prov")
        lbl_f_prov.setStyleSheet("font-weight: bold; color: #60a5fa; font-size: 12px;")
        self.lbl_comp_curr_prov = QLabel("-")
        self.lbl_comp_curr_prov.setStyleSheet("color: #cbd5e1; font-weight: 500;")
        self.lbl_comp_ai_prov = QLabel("-")
        self.lbl_comp_ai_prov.setStyleSheet("color: #38bdf8; font-weight: bold;")
        self.btn_apply_prov = QPushButton("⬅ Prov")
        self.btn_apply_prov.setObjectName("btnApplySingle")
        self.btn_apply_prov.setToolTip("Replace current Province with AI prediction")
        self.btn_apply_prov.clicked.connect(self._on_apply_ai_prov)
        
        comp_grid.addWidget(self.chk_sel_prov, 3, 0, Qt.AlignmentFlag.AlignCenter)
        comp_grid.addWidget(lbl_f_prov, 3, 1)
        comp_grid.addWidget(self.lbl_comp_curr_prov, 3, 2)
        comp_grid.addWidget(self.lbl_comp_ai_prov, 3, 3)
        comp_grid.addWidget(self.btn_apply_prov, 3, 4)
        
        comp_grid.setColumnStretch(2, 1)
        comp_grid.setColumnStretch(3, 1)
        ai_layout.addLayout(comp_grid)
        
        # Replacement Action Buttons
        ai_action_box = QHBoxLayout()
        ai_action_box.setSpacing(6)
        
        self.btn_replace_selected = QPushButton("🔀 Replace Selected")
        self.btn_replace_selected.setObjectName("btnReplaceSelected")
        self.btn_replace_selected.setToolTip("Replace checked fields (d3, d4, and/or Prov) with AI prediction")
        self.btn_replace_selected.clicked.connect(self._on_replace_selected_ai)
        ai_action_box.addWidget(self.btn_replace_selected, 1)
        
        self.btn_replace_all = QPushButton("⚡ Replace All (Ctrl+R)")
        self.btn_replace_all.setObjectName("btnReplaceAll")
        self.btn_replace_all.setToolTip("Replace all fields (d3 + d4 + Prov) with AI prediction (Ctrl+R)")
        self.btn_replace_all.clicked.connect(self._on_replace_all_ai)
        ai_action_box.addWidget(self.btn_replace_all, 1)
        
        ai_layout.addLayout(ai_action_box)
        
        return gb_ai

    def _create_yolo_inspector_box(self) -> QGroupBox:
        """Creates the interactive YOLO Bounding Box & Class inspector widget."""
        gb = QGroupBox("🎯 YOLO Bounding Box Inspector")
        gb.setObjectName("groupBoxYolo")
        y_layout = QVBoxLayout(gb)
        y_layout.setContentsMargins(12, 12, 12, 12)
        y_layout.setSpacing(10)
        
        # 1. Header with box counter and Save button
        h_box = QHBoxLayout()
        self.lbl_yolo_box_count = QLabel("Boxes: 0")
        self.lbl_yolo_box_count.setStyleSheet("font-weight: bold; color: #38bdf8; font-size: 13px;")
        h_box.addWidget(self.lbl_yolo_box_count)
        h_box.addStretch()
        
        self.btn_save_boxes = QPushButton("💾 Save Boxes (Ctrl+S)")
        self.btn_save_boxes.setObjectName("btnSaveBoxes")
        self.btn_save_boxes.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px;")
        self.btn_save_boxes.setToolTip("Save bounding boxes to .txt label file (Ctrl+S)")
        self.btn_save_boxes.clicked.connect(self._on_save_yolo_boxes)
        h_box.addWidget(self.btn_save_boxes)
        y_layout.addLayout(h_box)
        
        # 2. Table of Bounding Boxes
        self.table_boxes = QTableWidget()
        self.table_boxes.setColumnCount(4)
        self.table_boxes.setHorizontalHeaderLabels(["#", "Class", "Center (cx, cy)", "Size (w × h)"])
        self.table_boxes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_boxes.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_boxes.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_boxes.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_boxes.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_boxes.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table_boxes.setMinimumHeight(130)
        self.table_boxes.setMaximumHeight(200)
        self.table_boxes.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2d;
                color: #f1f5f9;
                gridline-color: #334155;
                border: 1px solid #334155;
                border-radius: 4px;
                font-size: 11px;
            }
            QTableWidget::item:selected {
                background-color: #3b82f6;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #0f172a;
                color: #94a3b8;
                font-weight: bold;
                padding: 4px;
                border: none;
            }
        """)
        self.table_boxes.itemSelectionChanged.connect(self._on_table_box_selected)
        y_layout.addWidget(self.table_boxes)
        
        # 3. Selected Box Detail Card (Coordinates & Class editor)
        sel_card = QFrame()
        sel_card.setStyleSheet("background-color: #1a1a24; border: 1px solid #374151; border-radius: 6px; padding: 6px;")
        card_layout = QVBoxLayout(sel_card)
        card_layout.setContentsMargins(6, 6, 6, 6)
        card_layout.setSpacing(6)
        
        self.lbl_selected_box = QLabel("No box selected (click on box or row)")
        self.lbl_selected_box.setStyleSheet("color: #94a3b8; font-size: 11px;")
        card_layout.addWidget(self.lbl_selected_box)
        
        # Class selector row
        row_class = QHBoxLayout()
        row_class.addWidget(QLabel("Class:"))
        self.combo_box_class = QComboBox()
        for cid, cname in sorted(self.session.class_names.items()):
            self.combo_box_class.addItem(f"{cid}: {cname}", userData=cid)
        row_class.addWidget(self.combo_box_class, 1)
        
        self.btn_apply_class = QPushButton("✏️ Set Class")
        self.btn_apply_class.setToolTip("Apply chosen class to selected box")
        self.btn_apply_class.clicked.connect(self._on_apply_class_clicked)
        row_class.addWidget(self.btn_apply_class)
        card_layout.addLayout(row_class)
        
        # Coordinates grid (cx, cy, w, h)
        grid_coords = QGridLayout()
        grid_coords.setSpacing(4)
        
        lbl_cx = QLabel("cx:")
        lbl_cx.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.spin_cx = QDoubleSpinBox()
        self.spin_cx.setRange(0.0, 1.0)
        self.spin_cx.setDecimals(4)
        self.spin_cx.setSingleStep(0.005)
        self.spin_cx.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(lbl_cx, 0, 0)
        grid_coords.addWidget(self.spin_cx, 0, 1)
        
        lbl_cy = QLabel("cy:")
        lbl_cy.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.spin_cy = QDoubleSpinBox()
        self.spin_cy.setRange(0.0, 1.0)
        self.spin_cy.setDecimals(4)
        self.spin_cy.setSingleStep(0.005)
        self.spin_cy.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(lbl_cy, 0, 2)
        grid_coords.addWidget(self.spin_cy, 0, 3)
        
        lbl_w = QLabel("w:")
        lbl_w.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.spin_w = QDoubleSpinBox()
        self.spin_w.setRange(0.0001, 1.0)
        self.spin_w.setDecimals(4)
        self.spin_w.setSingleStep(0.005)
        self.spin_w.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(lbl_w, 1, 0)
        grid_coords.addWidget(self.spin_w, 1, 1)
        
        lbl_h = QLabel("h:")
        lbl_h.setStyleSheet("color: #94a3b8; font-weight: bold;")
        self.spin_h = QDoubleSpinBox()
        self.spin_h.setRange(0.0001, 1.0)
        self.spin_h.setDecimals(4)
        self.spin_h.setSingleStep(0.005)
        self.spin_h.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(lbl_h, 1, 2)
        grid_coords.addWidget(self.spin_h, 1, 3)
        
        card_layout.addLayout(grid_coords)
        y_layout.addWidget(sel_card)
        
        # 4. Action buttons: Add & Delete
        ctrl_layout = QHBoxLayout()
        self.btn_add_box = QPushButton("➕ Add Box")
        self.btn_add_box.setToolTip("Add new bounding box with selected class")
        self.btn_add_box.clicked.connect(self._on_add_box_clicked)
        ctrl_layout.addWidget(self.btn_add_box, 1)
        
        self.btn_delete_box = QPushButton("🗑️ Delete Box (Del)")
        self.btn_delete_box.setStyleSheet("background-color: #ef4444; color: white; font-weight: bold;")
        self.btn_delete_box.setToolTip("Delete selected bounding box [Shortcut: Delete]")
        self.btn_delete_box.clicked.connect(self._on_delete_box_clicked)
        ctrl_layout.addWidget(self.btn_delete_box, 1)
        
        y_layout.addLayout(ctrl_layout)
        return gb

    def _on_viewer_box_clicked(self, idx: int):
        """Called when user clicks directly on a bounding box in the graphics view."""
        if idx == -1:
            self.selected_box_index = None
            if hasattr(self, 'table_boxes'):
                self.table_boxes.blockSignals(True)
                self.table_boxes.clearSelection()
                self.table_boxes.blockSignals(False)
            self._update_selected_box_card()
            return
            
        self.selected_box_index = idx
        if hasattr(self, 'table_boxes') and 0 <= idx < self.table_boxes.rowCount():
            self.table_boxes.blockSignals(True)
            self.table_boxes.selectRow(idx)
            self.table_boxes.blockSignals(False)
        self._update_selected_box_card()
        item = self.session.get_current_item()
        if item and 0 <= idx < len(item.boxes):
            cname = item.boxes[idx].get("cls_name", "box")
            self.lbl_status_msg.setText(f"Selected box #{idx + 1} ({cname}). Press Delete to remove.")

    def _on_table_box_selected(self):
        """Called when user selects a row in the bounding box table."""
        row = self.table_boxes.currentRow()
        if row >= 0:
            self.selected_box_index = row
            self.image_viewer.select_box(row)
            self._update_selected_box_card()
        else:
            self.selected_box_index = None
            self.image_viewer.select_box(None)
            self._update_selected_box_card()

    def _update_selected_box_card(self):
        """Synchronizes selected box details to the class combo and coordinate spinboxes."""
        item = self.session.get_current_item()
        has_sel = (
            item is not None and
            self.selected_box_index is not None and
            0 <= self.selected_box_index < len(item.boxes)
        )
        if not has_sel:
            self.lbl_selected_box.setText("No box selected (click box on image or select row)")
            self.lbl_selected_box.setStyleSheet("color: #94a3b8; font-size: 11px;")
            self.btn_delete_box.setEnabled(bool(item and item.boxes))
            return
            
        b = item.boxes[self.selected_box_index]
        cls_id = int(b.get("cls_id", 0))
        cls_name = b.get("cls_name", f"Class {cls_id}")
        cx, cy, w, h = b.get("bbox_norm", (0.5, 0.5, 0.2, 0.2))
        
        self.lbl_selected_box.setText(f"Selected: #{self.selected_box_index + 1} - {cls_name}")
        self.lbl_selected_box.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        
        self.combo_box_class.blockSignals(True)
        for i in range(self.combo_box_class.count()):
            if self.combo_box_class.itemData(i) == cls_id:
                self.combo_box_class.setCurrentIndex(i)
                break
        self.combo_box_class.blockSignals(False)
        
        self._syncing_coords = True
        self.spin_cx.setValue(cx)
        self.spin_cy.setValue(cy)
        self.spin_w.setValue(w)
        self.spin_h.setValue(h)
        self._syncing_coords = False
        self.btn_delete_box.setEnabled(True)

    def _on_spin_coords_changed(self):
        """Called when user edits coordinates in the spinboxes."""
        if getattr(self, '_syncing_coords', False):
            return
        item = self.session.get_current_item()
        if not item or self.selected_box_index is None or not (0 <= self.selected_box_index < len(item.boxes)):
            return
            
        cx = self.spin_cx.value()
        cy = self.spin_cy.value()
        w = self.spin_w.value()
        h = self.spin_h.value()
        
        item.boxes[self.selected_box_index]["bbox_norm"] = (cx, cy, w, h)
        
        row = self.selected_box_index
        if hasattr(self, 'table_boxes') and row < self.table_boxes.rowCount():
            self.table_boxes.blockSignals(True)
            self.table_boxes.setItem(row, 2, QTableWidgetItem(f"{cx:.3f}, {cy:.3f}"))
            self.table_boxes.setItem(row, 3, QTableWidgetItem(f"{w:.3f} × {h:.3f}"))
            self.table_boxes.blockSignals(False)
            
        self.image_viewer.set_bounding_boxes(item.boxes)
        self.image_viewer.select_box(self.selected_box_index)

    def _on_apply_class_clicked(self):
        """Changes the class of the currently selected box."""
        item = self.session.get_current_item()
        if not item or self.selected_box_index is None or not (0 <= self.selected_box_index < len(item.boxes)):
            QMessageBox.information(self, "No Box Selected", "Please select a bounding box first.")
            return
            
        new_cid = self.combo_box_class.currentData()
        if new_cid is None:
            return
        new_cname = self.session.class_names.get(new_cid, f"Class {new_cid}")
        
        item.boxes[self.selected_box_index]["cls_id"] = new_cid
        item.boxes[self.selected_box_index]["cls_name"] = new_cname
        item.save_yolo_boxes()
        
        self._refresh_yolo_table()
        self.image_viewer.set_bounding_boxes(item.boxes)
        self.image_viewer.select_box(self.selected_box_index)
        self._update_selected_box_card()
        self.lbl_status_msg.setText(f"Changed box #{self.selected_box_index + 1} class to {new_cname}")

    def _refresh_yolo_table(self):
        """Populates the YOLO bounding box inspector table for the current image."""
        item = self.session.get_current_item()
        if not item or not hasattr(self, 'table_boxes'):
            return
            
        boxes = item.boxes
        self.lbl_yolo_box_count.setText(f"Boxes: {len(boxes)}")
        
        self.table_boxes.blockSignals(True)
        self.table_boxes.setRowCount(len(boxes))
        
        from .image_viewer import DEFAULT_CLASS_COLORS
        
        for row, b in enumerate(boxes):
            cls_id = int(b.get("cls_id", 0))
            cls_name = b.get("cls_name", f"Class {cls_id}")
            cx, cy, w, h = b.get("bbox_norm", (0, 0, 0, 0))
            
            color = DEFAULT_CLASS_COLORS[cls_id % len(DEFAULT_CLASS_COLORS)]
            
            item_num = QTableWidgetItem(f"#{row + 1}")
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            
            item_cls = QTableWidgetItem(f"● {cls_name}")
            item_cls.setForeground(color)
            item_cls.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            
            item_center = QTableWidgetItem(f"{cx:.3f}, {cy:.3f}")
            item_size = QTableWidgetItem(f"{w:.3f} × {h:.3f}")
            
            self.table_boxes.setItem(row, 0, item_num)
            self.table_boxes.setItem(row, 1, item_cls)
            self.table_boxes.setItem(row, 2, item_center)
            self.table_boxes.setItem(row, 3, item_size)
            
        if self.selected_box_index is not None and 0 <= self.selected_box_index < len(boxes):
            self.table_boxes.selectRow(self.selected_box_index)
        self.table_boxes.blockSignals(False)
        self._update_selected_box_card()

    def _on_delete_box_clicked(self):
        """Deletes selected box from current image, saves, and updates view."""
        item = self.session.get_current_item()
        if not item or not item.boxes:
            QMessageBox.information(self, "No Boxes", "There are no bounding boxes on this image.")
            return
            
        target_idx = self.selected_box_index
        if target_idx is None or target_idx < 0 or target_idx >= len(item.boxes):
            target_idx = self.table_boxes.currentRow()
            
        if target_idx is None or target_idx < 0 or target_idx >= len(item.boxes):
            if len(item.boxes) == 1:
                target_idx = 0
            else:
                QMessageBox.warning(self, "Select Box", "Please click a box on the image or select a row in the table to delete.")
                return
                
        deleted_box = item.boxes.pop(target_idx)
        self.selected_box_index = None
        ok = item.save_yolo_boxes()
        
        self._refresh_yolo_table()
        self.image_viewer.set_bounding_boxes(item.boxes)
        self.image_viewer.select_box(None)
        self._update_selected_box_card()
        self._update_ui_state()
        
        fname = item.label_path.name if item.label_path else "label file"
        self.lbl_status_msg.setText(f"Deleted box #{target_idx + 1} ({deleted_box.get('cls_name', 'box')}) and saved {fname}.")

    def _on_add_box_clicked(self):
        """Adds a new default bounding box to current image and selects it."""
        item = self.session.get_current_item()
        if not item:
            return
        cls_id = self.combo_box_class.currentData()
        if cls_id is None:
            cls_id = 0
        cls_name = self.session.class_names.get(cls_id, f"Class {cls_id}")
        new_box = {
            "cls_id": cls_id,
            "cls_name": cls_name,
            "bbox_norm": (0.5000, 0.5000, 0.2000, 0.2000)
        }
        item.boxes.append(new_box)
        new_idx = len(item.boxes) - 1
        self.selected_box_index = new_idx
        
        item.save_yolo_boxes()
        
        self._refresh_yolo_table()
        self.image_viewer.set_bounding_boxes(item.boxes)
        self.image_viewer.select_box(new_idx)
        self._update_selected_box_card()
        self._update_ui_state()
        self.lbl_status_msg.setText(f"Added new {cls_name} box #{new_idx + 1}. Adjust coordinates or press Delete to remove.")

    def _on_save_yolo_boxes(self):
        """Saves current bounding boxes to YOLO label file."""
        item = self.session.get_current_item()
        if not item:
            return
        ok = item.save_yolo_boxes()
        if ok:
            fname = item.label_path.name if item.label_path else "label file"
            self.lbl_status_msg.setText(f"Saved {len(item.boxes)} bounding box(es) to {fname}")
        else:
            QMessageBox.critical(self, "Save Error", f"Failed to save bounding boxes to disk.")

    def _refresh_class_filter_combo(self):
        """Populates class filter dropdown with classes and counts."""
        if not hasattr(self, 'combo_class_filter'):
            return
        self.combo_class_filter.blockSignals(True)
        self.combo_class_filter.clear()
        
        if self.session.dataset_type == "yolo":
            stats = self.session.get_yolo_stats(self.session.active_subfolder)
            total = stats["total_images"]
            self.combo_class_filter.addItem(f"(All Classes) ({total})", userData=-1)
            
            for cid, cname in sorted(self.session.class_names.items()):
                cnt = stats["class_counts"].get(cid, 0)
                self.combo_class_filter.addItem(f"{cid}: {cname} ({cnt} boxes)", userData=cid)
                
            unannotated_cnt = total - stats["images_with_boxes"]
            if unannotated_cnt > 0:
                self.combo_class_filter.addItem(f"(No Annotations) ({unannotated_cnt})", userData=-2)
                
            if self.session.active_class_filter is not None:
                for idx in range(self.combo_class_filter.count()):
                    if self.combo_class_filter.itemData(idx) == self.session.active_class_filter:
                        self.combo_class_filter.setCurrentIndex(idx)
                        break
        self.combo_class_filter.blockSignals(False)

    def _on_class_filter_changed(self, index: int):
        """Filters images by selected YOLO class."""
        data = self.combo_class_filter.currentData()
        self.session.active_class_filter = None if (data is None or data == -1) else data
        
        filtered = self.session.get_filtered_indices(self.session.active_subfolder, self.session.active_class_filter, getattr(self.session, "active_status_filter", "all"))
        if filtered:
            if self.session.current_index not in filtered:
                self._show_item_at(filtered[0])
            else:
                self._update_ui_state()
        else:
            self._update_ui_state()

    def _refresh_status_filter_combo(self):
        """Populates status filter dropdown with live counts for current subfolder."""
        if not hasattr(self, 'combo_status_filter'):
            return
            
        active_sf = self.session.active_subfolder
        stats = self.session.get_stats(active_sf)
        
        all_count = stats["total"]
        pending_count = stats["pending"] + stats["skipped"]
        not_plate_count = stats["not_plate"]
        cannot_read_count = stats["cannot_read"]
        approved_count = stats["approved"]
        
        dup_count = sum(1 for it in self.session.items if (not active_sf or active_sf == "(All)" or it.subfolder == active_sf) and ("_dup" in it.filename.lower()))
        ai_diff_count = sum(1 for it in self.session.items if (not active_sf or active_sf == "(All)" or it.subfolder == active_sf) and it.get_ai_comparison()["has_ai"] and not it.get_ai_comparison()["all_match"])
        
        curr_data = self.combo_status_filter.currentData() or getattr(self.session, "active_status_filter", "all") or "all"
        
        self.combo_status_filter.blockSignals(True)
        self.combo_status_filter.clear()
        self.combo_status_filter.addItem(f"All Images ({all_count:,})", userData="all")
        self.combo_status_filter.addItem(f"⏳ Pending ({pending_count:,})", userData="pending")
        self.combo_status_filter.addItem(f"🚫 Not Plate ({not_plate_count:,})", userData="not_plate")
        self.combo_status_filter.addItem(f"❓ Cannot Read ({cannot_read_count:,})", userData="cannot_read")
        self.combo_status_filter.addItem(f"✅ Approved ({approved_count:,})", userData="approved")
        if ai_diff_count > 0:
            self.combo_status_filter.addItem(f"⚡ AI Diff ({ai_diff_count:,})", userData="ai_diff")
        if dup_count > 0:
            self.combo_status_filter.addItem(f"📑 Duplicates ({dup_count:,})", userData="dup")
            
        idx = self.combo_status_filter.findData(curr_data)
        if idx >= 0:
            self.combo_status_filter.setCurrentIndex(idx)
        else:
            self.combo_status_filter.setCurrentIndex(0)
            self.session.active_status_filter = "all"
            
        self.combo_status_filter.blockSignals(False)

    def _on_status_filter_changed(self, index: int):
        """Filters images to annotate by selected status (e.g. Not Plate, Cannot Read, Pending)."""
        status_code = self.combo_status_filter.currentData() or "all"
        self.session.active_status_filter = status_code
        
        filtered = self.session.get_filtered_indices(
            self.session.active_subfolder,
            self.session.active_class_filter,
            status_code
        )
        
        if filtered:
            if self.session.current_index not in filtered:
                self._show_item_at(filtered[0])
            else:
                self._update_ui_state()
            filter_text = self.combo_status_filter.currentText()
            self.lbl_status_msg.setText(f"Filter active: {filter_text} ({len(filtered)} images)")
        else:
            self._update_ui_state()
            self.lbl_status_msg.setText(f"No images found for filter: {self.combo_status_filter.currentText()}")

    def _set_status_filter(self, filter_code: str):
        """Programmatically sets the status filter dropdown and applies filter."""
        if not hasattr(self, 'combo_status_filter'):
            return
        idx = self.combo_status_filter.findData(filter_code)
        if idx >= 0:
            self.combo_status_filter.setCurrentIndex(idx)

    def _update_image_jump_combo(self, filtered: Optional[List[int]] = None, curr_idx: Optional[int] = None):
        """Populates image jump dropdown with list of images matching current filter."""
        if not hasattr(self, 'combo_image_jump'):
            return
            
        if filtered is None:
            active_sf = self.session.active_subfolder
            active_cf = self.session.active_class_filter
            active_st = getattr(self.session, "active_status_filter", "all")
            filtered = self.session.get_filtered_indices(active_sf, active_cf, active_st)
            
        if curr_idx is None:
            curr_idx = self.session.current_index

        self.combo_image_jump.blockSignals(True)
        self.combo_image_jump.clear()
        
        if not filtered:
            self.combo_image_jump.addItem("(No images in filter)", userData=-1)
            self.combo_image_jump.setEnabled(False)
            self.combo_image_jump.blockSignals(False)
            return
            
        self.combo_image_jump.setEnabled(True)
        current_combo_idx = 0
        
        for pos, i in enumerate(filtered, start=1):
            it = self.session.items[i]
            st_icon = "🚫" if it.status == "not_plate" else ("❓" if it.status == "cannot_read" else ("✅" if it.status == "approved" else "⏳"))
            label = f"{pos}. {it.filename} {st_icon}"
            self.combo_image_jump.addItem(label, userData=i)
            if i == curr_idx:
                current_combo_idx = pos - 1
                
        self.combo_image_jump.setCurrentIndex(current_combo_idx)
        self.combo_image_jump.blockSignals(False)

    def _on_jump_image_selected(self, index: int):
        """Jumps directly to selected image from the filtered image list."""
        target_idx = self.combo_image_jump.currentData()
        if target_idx is not None and target_idx >= 0 and target_idx != self.session.current_index:
            self._show_item_at(target_idx)

    def _on_toggle_view_mode(self):
        """Toggles between YOLO Object Detection Mode and Plate OCR Mode."""
        self.session.dataset_type = "ocr" if self.session.dataset_type == "yolo" else "yolo"
        self._apply_dataset_mode_ui()
        self._update_ui_state()
        if self.session.items:
            self._show_item_at(self.session.current_index)

    def _apply_dataset_mode_ui(self):
        """Updates UI layout visibility based on YOLO vs OCR mode."""
        is_yolo = (self.session.dataset_type == "yolo")
        self.badge_mode.setText("🎯 YOLO Detection" if is_yolo else "🔤 Plate OCR")
        self.badge_mode.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
            if is_yolo else
            "background-color: #059669; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        )
        self.combo_class_filter.setVisible(is_yolo)
        self.lbl_filter_class.setVisible(is_yolo)
        self.combo_status_filter.setVisible(not is_yolo)
        self.lbl_filter_status.setVisible(not is_yolo)
        self.gb_yolo_inspector.setVisible(is_yolo)
        
        self.gb_plate_info.setVisible(not is_yolo)
        self.gb_ai.setVisible(not is_yolo)
        self.btn_clean_dup.setVisible(not is_yolo)
        self.btn_batch_ai.setVisible(not is_yolo)
        self.btn_not_plate.setVisible(not is_yolo)
        self.btn_cannot_read.setVisible(not is_yolo)
        
        if is_yolo:
            self.btn_approve.setText("💾 Save Bounding Boxes (Enter)")
        else:
            self.btn_approve.setText("✅ Correct / Approve (Enter)")

    def _on_train_models_clicked(self):
        """Opens the Model Training Studio dialog."""
        dlg = TrainingDialog(self)
        dlg.exec()

    def _create_bottom_bar(self) -> QWidget:
        bar = QWidget(self)
        b_layout = QHBoxLayout(bar)
        b_layout.setContentsMargins(4, 0, 4, 0)
        b_layout.setSpacing(12)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("appProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        b_layout.addWidget(self.progress_bar, 1)
        
        self.lbl_status_msg = QLabel("Ready. Open a folder to begin.")
        self.lbl_status_msg.setStyleSheet("color: #9ca3af; font-size: 12px;")
        b_layout.addWidget(self.lbl_status_msg)
        
        return bar

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Left"), self, self._on_back_clicked)
        QShortcut(QKeySequence("Right"), self, self._on_next_clicked)
        QShortcut(QKeySequence("Ctrl+Left"), self, self._on_back_clicked)
        QShortcut(QKeySequence("Ctrl+Right"), self, self._on_next_clicked)
        QShortcut(QKeySequence("Ctrl+R"), self, self._on_replace_all_ai)
        QShortcut(QKeySequence("Ctrl+N"), self, self._on_not_plate_clicked)
        QShortcut(QKeySequence("Ctrl+D"), self, self._on_cannot_read_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_ctrl_s_pressed)
        QShortcut(QKeySequence("Delete"), self, self._on_delete_key_pressed)
        QShortcut(QKeySequence("Ctrl+Delete"), self, self._on_delete_clicked)
        QShortcut(QKeySequence("F2"), self, self._on_rename_clicked)
        QShortcut(QKeySequence("B"), self, self.image_viewer.btn_toggle_boxes.click)
        QShortcut(QKeySequence("L"), self, self.image_viewer.btn_toggle_labels.click)
        QShortcut(QKeySequence("Ctrl+T"), self, self._on_train_models_clicked)
        QShortcut(QKeySequence("Alt+N"), self, lambda: self._set_status_filter("not_plate"))
        QShortcut(QKeySequence("Alt+D"), self, lambda: self._set_status_filter("cannot_read"))
        QShortcut(QKeySequence("Alt+P"), self, lambda: self._set_status_filter("pending"))
        QShortcut(QKeySequence("Alt+A"), self, lambda: self._set_status_filter("all"))

    def _on_ctrl_s_pressed(self):
        """Ctrl+S saves bounding boxes in YOLO mode, or skips in OCR mode."""
        if self.session.dataset_type == "yolo":
            self._on_save_yolo_boxes()
        else:
            self._on_skip_clicked()

    def _on_delete_key_pressed(self):
        """Delete key removes selected bounding box in YOLO mode, or prompts image delete in OCR mode."""
        if self.session.dataset_type == "yolo":
            self._on_delete_box_clicked()
        else:
            self._on_delete_clicked()

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder:
            return
            
        success = self.session.load_folder(folder)
        if not success:
            QMessageBox.warning(self, "Error", "Failed to load selected folder.")
            return
            
        self._refresh_subfolder_combo()
        self._refresh_class_filter_combo()
        self._refresh_status_filter_combo()
        self._apply_dataset_mode_ui()
        
        if not self.session.items:
            self.image_viewer.clear()
            self._update_ui_state()
            QMessageBox.information(self, "No Images", "No supported images found in this folder or its subfolders.")
            return
            
        self._show_item_at(self.session.current_index)
        mode_str = "YOLO Object Detection Mode" if self.session.dataset_type == "yolo" else "OCR Plate Mode"
        self.lbl_status_msg.setText(f"Folder loaded with {len(self.session.items)} images [{mode_str}]. Resumed at #{self.session.current_index + 1}.")

    def _refresh_subfolder_combo(self):
        self.combo_subfolder.blockSignals(True)
        self.combo_subfolder.clear()
        
        total_count = len(self.session.items)
        self.combo_subfolder.addItem(f"All Subfolders ({total_count})", userData="(All)")
        
        subfolders = self.session.get_subfolders()
        root_count = len([it for it in self.session.items if not it.subfolder])
        if root_count > 0 and subfolders:
            self.combo_subfolder.addItem(f"(Root Folder) ({root_count})", userData="(Root)")
            
        for sf in subfolders:
            sf_count = len([it for it in self.session.items if it.subfolder == sf])
            self.combo_subfolder.addItem(f"{sf} ({sf_count})", userData=sf)
            
        self.combo_subfolder.blockSignals(False)

    def _on_subfolder_changed(self, index: int):
        data = self.combo_subfolder.currentData()
        self.session.active_subfolder = None if data == "(All)" else data
        
        self._refresh_status_filter_combo()
        self._refresh_class_filter_combo()
        
        # Jump to first item in this subfolder if current item doesn't belong to it
        curr = self.session.get_current_item()
        filtered = self.session.get_filtered_indices(self.session.active_subfolder, self.session.active_class_filter, getattr(self.session, "active_status_filter", "all"))
        if filtered:
            if not curr or self.session.current_index not in filtered:
                self._show_item_at(filtered[0])
            else:
                self._update_ui_state()
        else:
            self._update_ui_state()

    def _show_item_at(self, index: int):
        if not (0 <= index < len(self.session.items)):
            return
            
        self.session.current_index = index
        item = self.session.items[index]
        
        # Load image into viewer
        self.image_viewer.load_image(str(item.path))

        # Reset selection state
        self.selected_box_index = None

        # Render bounding boxes if in YOLO mode or item has boxes
        if self.session.dataset_type == "yolo" or item.boxes:
            self.image_viewer.set_bounding_boxes(item.boxes)
            self.image_viewer.select_box(None)
            self._refresh_yolo_table()
        else:
            self.image_viewer.clear_bounding_boxes()
        
        # Update Form Fields (block signals temporarily to prevent loop)
        self.txt_d3.blockSignals(True)
        self.txt_d4.blockSignals(True)
        self.combo_province.blockSignals(True)
        
        sf_display = item.subfolder if item.subfolder else "(Root Folder)"
        self.lbl_current_subfolder.setText(f"📂 Folder: {sf_display}")
        self.lbl_current_filename.setText(item.filename)
        self.txt_name.setText(item.name)
        self.combo_plate_type.set_plate_type_value(item.plate_type)
        self.txt_d3.setText(item.d3)
        self.txt_d4.setText(item.d4)
        self.combo_province.set_province_value(item.province)
        
        self.txt_d3.blockSignals(False)
        self.txt_d4.blockSignals(False)
        self.combo_province.blockSignals(False)
        
        # Update Status Badge
        self._update_status_display(item)
        
        # Update Stats & Progress
        self._update_ui_state()
        
        # Mode-specific focus & AI handling
        if self.session.dataset_type == "ocr":
            self._update_ai_prediction_for_item(item)
            self.txt_d3.setFocus()
            self.txt_d3.selectAll()

    # -------------------------------------------------------------------------
    # AI Reader (AI Reader) Execution & Live Comparison
    # -------------------------------------------------------------------------

    def _update_ai_prediction_for_item(self, item: ImageItem):
        """Checks for existing AI prediction or triggers AI Reader inference."""
        if item.ai_d3 or item.ai_d4 or item.ai_province:
            # Use cached prediction on item
            self._latest_ai_d3 = item.ai_d3
            self._latest_ai_d4 = item.ai_d4
            self._latest_ai_prov = item.ai_province
            self._latest_ai_conf = item.ai_confidence
            self._refresh_comparison_view()
        elif self.chk_auto_ai.isChecked():
            # Trigger background prediction
            self._run_single_ai_read(item)
        else:
            # Auto-read is disabled and no cached result
            self._latest_ai_d3 = ""
            self._latest_ai_d4 = ""
            self._latest_ai_prov = ""
            self._latest_ai_conf = 0.0
            self.lbl_ai_status.setText("AI Status: Idle. Click 'Read Plate' to infer.")
            self.lbl_ai_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
            self._refresh_comparison_view()

    def _run_single_ai_read(self, item: ImageItem):
        """Spawns an asynchronous worker to predict the current image."""
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.cancel()
            self._active_worker.wait(300)
            
        self.lbl_ai_status.setText("AI Status: Running AI Reader inference...")
        self.lbl_ai_status.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: bold;")
        self.lbl_comp_ai_d3.setText("...")
        self.lbl_comp_ai_d4.setText("...")
        self.lbl_comp_ai_prov.setText("...")
        
        self._active_worker = AISingleReaderWorker(str(item.path), self)
        
        def on_pred(path_str, res):
            curr_item = self.session.get_current_item()
            if curr_item and str(curr_item.path) == path_str:
                if res.get("error"):
                    self.lbl_ai_status.setText(f"AI Error: {res['error']}")
                    self.lbl_ai_status.setStyleSheet("color: #ef4444; font-size: 11px;")
                    self._latest_ai_d3 = ""
                    self._latest_ai_d4 = ""
                    self._latest_ai_prov = ""
                    self._latest_ai_conf = 0.0
                else:
                    curr_item.set_ai_prediction(
                        res["d3"], res["d4"], res["province"], res["confidence"]
                    )
                    self._latest_ai_d3 = res["d3"]
                    self._latest_ai_d4 = res["d4"]
                    self._latest_ai_prov = res["province"]
                    self._latest_ai_conf = res["confidence"]
                    
                self._refresh_comparison_view()
                
        self._active_worker.prediction_ready.connect(on_pred)
        self._active_worker.start()

    def _on_manual_ai_read_clicked(self):
        item = self.session.get_current_item()
        if not item:
            return
        self._run_single_ai_read(item)

    def _on_manual_field_changed(self):
        """Called whenever user manually edits d3, d4, or province in the form."""
        self._refresh_comparison_view()

    def _refresh_comparison_view(self):
        """Updates the Live Comparison labels, diff badges, and button states."""
        curr_d3 = self.txt_d3.text().strip()
        curr_d4 = self.txt_d4.text().strip()
        curr_prov = self.combo_province.get_province_value().strip()
        
        ai_d3 = self._latest_ai_d3.strip()
        ai_d4 = self._latest_ai_d4.strip()
        ai_prov = self._latest_ai_prov.strip()
        
        self.lbl_comp_curr_d3.setText(curr_d3 if curr_d3 else "(empty)")
        self.lbl_comp_curr_d4.setText(curr_d4 if curr_d4 else "(empty)")
        self.lbl_comp_curr_prov.setText(curr_prov if curr_prov else "(empty)")
        
        self.lbl_comp_ai_d3.setText(ai_d3 if ai_d3 else "-")
        self.lbl_comp_ai_d4.setText(ai_d4 if ai_d4 else "-")
        self.lbl_comp_ai_prov.setText(ai_prov if ai_prov else "-")
        
        has_ai = bool(ai_d3 or ai_d4 or ai_prov)
        
        if has_ai:
            # 1. d3 Match Check
            d3_match = (normalize_thai_script(curr_d3) == normalize_thai_script(ai_d3))
            if d3_match:
                self.lbl_comp_ai_d3.setStyleSheet("color: #4ade80; font-weight: bold;")
                self.btn_apply_d3.setEnabled(False)
            else:
                self.lbl_comp_ai_d3.setStyleSheet("color: #facc15; font-weight: bold;")
                self.btn_apply_d3.setEnabled(True)
                
            # 2. d4 Match Check
            d4_match = (curr_d4 == ai_d4)
            if d4_match:
                self.lbl_comp_ai_d4.setStyleSheet("color: #4ade80; font-weight: bold;")
                self.btn_apply_d4.setEnabled(False)
            else:
                self.lbl_comp_ai_d4.setStyleSheet("color: #facc15; font-weight: bold;")
                self.btn_apply_d4.setEnabled(True)
                
            # 3. Province Match Check
            prov_match = (normalize_thai_script(curr_prov) == normalize_thai_script(ai_prov))
            if prov_match:
                self.lbl_comp_ai_prov.setStyleSheet("color: #4ade80; font-weight: bold;")
                self.btn_apply_prov.setEnabled(False)
            else:
                self.lbl_comp_ai_prov.setStyleSheet("color: #facc15; font-weight: bold;")
                self.btn_apply_prov.setEnabled(True)
                
            # Overall Status & Action buttons
            all_match = (d3_match and d4_match and prov_match)
            if all_match:
                self.lbl_ai_status.setText(f"AI Status: All Values Matched ✅ (Confidence: {self._latest_ai_conf:.1%})")
                self.lbl_ai_status.setStyleSheet("color: #4ade80; font-weight: bold; font-size: 11px;")
                self.btn_replace_all.setEnabled(False)
            else:
                diffs = []
                if not d3_match: diffs.append("d3")
                if not d4_match: diffs.append("d4")
                if not prov_match: diffs.append("Province")
                self.lbl_ai_status.setText(f"AI Status: Differences in [{', '.join(diffs)}] ⚠️ (Confidence: {self._latest_ai_conf:.1%})")
                self.lbl_ai_status.setStyleSheet("color: #facc15; font-weight: bold; font-size: 11px;")
                self.btn_replace_all.setEnabled(True)
                
            self.btn_replace_selected.setEnabled(not all_match)
        else:
            self.lbl_comp_ai_d3.setStyleSheet("color: #94a3b8;")
            self.lbl_comp_ai_d4.setStyleSheet("color: #94a3b8;")
            self.lbl_comp_ai_prov.setStyleSheet("color: #94a3b8;")
            self.btn_apply_d3.setEnabled(False)
            self.btn_apply_d4.setEnabled(False)
            self.btn_apply_prov.setEnabled(False)
            self.btn_replace_selected.setEnabled(False)
            self.btn_replace_all.setEnabled(False)

    # -------------------------------------------------------------------------
    # Replacement Handlers (Individual and Bulk)
    # -------------------------------------------------------------------------

    def _on_apply_ai_d3(self):
        """Replaces only d3 with the predicted value."""
        if not self._latest_ai_d3:
            return
        self.txt_d3.setText(self._latest_ai_d3)
        self.lbl_status_msg.setText(f"Replaced d3 with AI: '{self._latest_ai_d3}'")
        self._refresh_comparison_view()

    def _on_apply_ai_d4(self):
        """Replaces only d4 with the predicted value."""
        if not self._latest_ai_d4:
            return
        self.txt_d4.setText(self._latest_ai_d4)
        self.lbl_status_msg.setText(f"Replaced d4 with AI: '{self._latest_ai_d4}'")
        self._refresh_comparison_view()

    def _on_apply_ai_prov(self):
        """Replaces only Province with the predicted value."""
        if not self._latest_ai_prov:
            return
        self.combo_province.set_province_value(self._latest_ai_prov)
        self.lbl_status_msg.setText(f"Replaced Province with AI: '{self._latest_ai_prov}'")
        self._refresh_comparison_view()

    def _on_replace_selected_ai(self):
        """Replaces whichever fields have their checkboxes checked."""
        replaced = []
        if self.chk_sel_d3.isChecked() and self._latest_ai_d3:
            self.txt_d3.setText(self._latest_ai_d3)
            replaced.append("d3")
        if self.chk_sel_d4.isChecked() and self._latest_ai_d4:
            self.txt_d4.setText(self._latest_ai_d4)
            replaced.append("d4")
        if self.chk_sel_prov.isChecked() and self._latest_ai_prov:
            self.combo_province.set_province_value(self._latest_ai_prov)
            replaced.append("Province")
            
        if replaced:
            self.lbl_status_msg.setText(f"Replaced selected fields ({', '.join(replaced)}) with AI prediction.")
        else:
            self.lbl_status_msg.setText("No checked fields or AI predictions available to replace.")
        self._refresh_comparison_view()

    def _on_replace_all_ai(self):
        """Replaces d3, d4, and Province with AI prediction (Ctrl+R)."""
        if not (self._latest_ai_d3 or self._latest_ai_d4 or self._latest_ai_prov):
            self.lbl_status_msg.setText("No AI prediction available to replace.")
            return
            
        if self._latest_ai_d3:
            self.txt_d3.setText(self._latest_ai_d3)
        if self._latest_ai_d4:
            self.txt_d4.setText(self._latest_ai_d4)
        if self._latest_ai_prov:
            self.combo_province.set_province_value(self._latest_ai_prov)
            
        full = f"{self._latest_ai_d3} {self._latest_ai_d4} {self._latest_ai_prov}".strip()
        self.lbl_status_msg.setText(f"Replaced ALL fields with AI prediction: {full}")
        self._refresh_comparison_view()

    def _on_batch_ai_clicked(self):
        """Runs AI Reader reader across all images in active subfolder or folder."""
        if not self.session.items:
            QMessageBox.information(self, "No Images", "Please open an images folder first.")
            return
            
        active_sf = self.session.active_subfolder
        indices = self.session.get_filtered_indices(active_sf)
        if not indices:
            QMessageBox.information(self, "No Images", "No images found in current filter.")
            return
            
        reply = QMessageBox.question(
            self,
            "Run Batch AI Reader",
            f"Run AI Reader AI Reader on {len(indices)} image(s) in "
            f"{'All Subfolders' if not active_sf or active_sf == '(All)' else active_sf}?\n\n"
            f"Predictions will be stored for instant review and comparison.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
            
        items_to_read = [(idx, str(self.session.items[idx].path)) for idx in indices]
        
        dlg = QProgressDialog("Initializing AI Reader (AI Reader)...", "Cancel", 0, len(items_to_read), self)
        dlg.setWindowTitle("Batch AI Reader (AI Reader)")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.resize(450, 130)
        dlg.show()
        
        self._batch_worker = AIBatchReaderWorker(items_to_read, self)
        
        def on_batch_progress(done, total, fname):
            dlg.setValue(done)
            dlg.setLabelText(f"Reading {done}/{total}: {fname}")
            
        def on_batch_item_predicted(idx, res):
            if not res.get("error"):
                self.session.items[idx].set_ai_prediction(
                    res["d3"], res["d4"], res["province"], res["confidence"]
                )
                if idx == self.session.current_index:
                    self._update_ai_prediction_for_item(self.session.items[idx])
                    
        def on_batch_finished(completed):
            dlg.close()
            self._update_ui_state()
            self._show_item_at(self.session.current_index)
            QMessageBox.information(
                self,
                "Batch AI Reader Complete",
                f"Successfully inferred AI Reader reader on {completed} image(s)!\n\n"
                f"Use 'Check Menu' or navigate to review and approve predictions."
            )
            
        dlg.canceled.connect(self._batch_worker.cancel)
        self._batch_worker.progress.connect(on_batch_progress)
        self._batch_worker.item_predicted.connect(on_batch_item_predicted)
        self._batch_worker.batch_finished.connect(on_batch_finished)
        self._batch_worker.start()

    # -------------------------------------------------------------------------
    # Core Lifecycle & Save Actions
    # -------------------------------------------------------------------------

    def _update_status_display(self, item: ImageItem):
        status_map = {
            "approved": ("Status: Approved ✅", "#22c55e"),
            "not_plate": ("Status: Not Plate (+)", "#3b82f6"),
            "cannot_read": ("Status: Cannot Read (-)", "#ef4444"),
            "skipped": ("Status: Skipped ⚠️", "#f59e0b"),
            "pending": ("Status: Pending ⏳", "#9ca3af"),
        }
        text, color = status_map.get(item.status, ("Status: Ready", "#9ca3af"))
        self.lbl_current_status.setText(text)
        self.lbl_current_status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 13px;")

    def _update_ui_state(self):
        active_sf = self.session.active_subfolder
        active_cf = self.session.active_class_filter
        active_st = getattr(self.session, "active_status_filter", "all")
        filtered = self.session.get_filtered_indices(active_sf, active_cf, active_st)
        curr_idx = self.session.current_index
        
        if self.session.dataset_type == "yolo":
            stats = self.session.get_yolo_stats(active_sf)
            self.badge_total.setText(f"Images: {stats['total_images']}")
            self.badge_left.setText(f"Filtered: {len(filtered)}")
            self.badge_approved.setText(f"Total Boxes: {stats['total_boxes']}")
            self.badge_not_plate.setText(f"With Boxes: {stats['images_with_boxes']}")
            self.badge_cannot_read.setText(f"Classes: {len(self.session.class_names)}")
        else:
            stats = self.session.get_stats(active_sf)
            total = stats["total"]
            self.badge_total.setText(f"Total: {total}")
            self.badge_left.setText(f"Left: {stats['left']}")
            self.badge_approved.setText(f"Approved: {stats['approved']}")
            self.badge_not_plate.setText(f"Not Plate: {stats['not_plate']}")
            self.badge_cannot_read.setText(f"Cannot Read: {stats['cannot_read']}")
        
        if filtered and curr_idx in filtered:
            pos = filtered.index(curr_idx) + 1
            sf_label = f" ({active_sf})" if active_sf and active_sf != "(All)" else ""
            cf_label = f" [{self.session.class_names.get(active_cf, 'Class')}]" if active_cf is not None and active_cf >= 0 else ""
            st_map = {
                "not_plate": " [Not Plate 🚫]",
                "cannot_read": " [Cannot Read ❓]",
                "pending": " [Pending ⏳]",
                "approved": " [Approved ✅]",
                "ai_diff": " [AI Diff ⚡]",
                "dup": " [Duplicates 📑]",
            }
            st_label = st_map.get(active_st, "")
            self.lbl_nav_counter.setText(f"{pos} / {len(filtered)}{sf_label}{cf_label}{st_label}")
            pct = int((pos / len(filtered)) * 100) if filtered else 0
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}% ({pos}/{len(filtered)})")
        elif filtered:
            st_map = {
                "not_plate": " [Not Plate 🚫]",
                "cannot_read": " [Cannot Read ❓]",
                "pending": " [Pending ⏳]",
                "approved": " [Approved ✅]",
                "ai_diff": " [AI Diff ⚡]",
                "dup": " [Duplicates 📑]",
            }
            st_label = st_map.get(active_st, "")
            self.lbl_nav_counter.setText(f"0 / {len(filtered)}{st_label}")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(f"0 / {len(filtered)}")
        else:
            self.lbl_nav_counter.setText("0 / 0")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0%")
            
        self.btn_back.setEnabled(len(filtered) > 1 and curr_idx in filtered and curr_idx != filtered[0])
        self.btn_next.setEnabled(len(filtered) > 1 and curr_idx in filtered and curr_idx != filtered[-1])
        
        self._update_image_jump_combo(filtered, curr_idx)

    def _on_approve_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        item = self.session.items[idx]
        
        if self.session.dataset_type == "yolo":
            item.save_yolo_boxes()
            if getattr(self, "chk_auto_staging", None) and self.chk_auto_staging.isChecked():
                self.session.add_to_staging(item)
            self.lbl_status_msg.setText(f"Saved bounding boxes for #{idx + 1}: {item.filename}")
        else:
            name = self.txt_name.text()
            plate_type = self.combo_plate_type.get_plate_type_value()
            d3 = self.txt_d3.text()
            d4 = self.txt_d4.text()
            province = self.combo_province.get_province_value()
            self.session.approve_and_save(idx, name, plate_type, d3, d4, province)
            if item.status == "approved":
                self.lbl_status_msg.setText(f"Approved #{idx + 1} -> Staged to: {plate_type}_{d3}_{d4}_{province}/")
            else:
                self.lbl_status_msg.setText(f"Saved #{idx + 1}: {item.status}")
        
        active_sf = self.session.active_subfolder
        active_cf = self.session.active_class_filter
        active_st = getattr(self.session, "active_status_filter", "all")
        next_idx = self.session.get_next_index(idx, active_sf, active_cf, active_st)
        self._refresh_status_filter_combo()
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)
            self.lbl_status_msg.setText("Reached last image in current filter!")

    def _on_not_plate_clicked(self):
        if not self.session.items:
            return
            
        self.combo_plate_type.set_plate_type_value("+")
        self.txt_d3.setText("+")
        self.txt_d4.setText("+")
        self.combo_province.set_province_value("+")
        self.lbl_status_msg.setText("Filled boxes with '+' (Not Plate). Click Approve (Enter) to save.")

    def _on_cannot_read_clicked(self):
        if not self.session.items:
            return
            
        self.combo_plate_type.set_plate_type_value("-")
        self.txt_d3.setText("-")
        self.txt_d4.setText("-")
        self.combo_province.set_province_value("-")
        self.lbl_status_msg.setText("Filled boxes with '-' (Cannot Read). Click Approve (Enter) to save.")

    def _on_skip_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        self.session.mark_skipped(idx)
        self.lbl_status_msg.setText(f"Skipped image #{idx + 1}")
        
        active_sf = self.session.active_subfolder
        active_cf = self.session.active_class_filter
        active_st = getattr(self.session, "active_status_filter", "all")
        next_idx = self.session.get_next_index(idx, active_sf, active_cf, active_st)
        self._refresh_status_filter_combo()
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)

    def _on_back_clicked(self):
        active_sf = self.session.active_subfolder
        active_cf = self.session.active_class_filter
        active_st = getattr(self.session, "active_status_filter", "all")
        prev_idx = self.session.get_prev_index(self.session.current_index, active_sf, active_cf, active_st)
        if prev_idx is not None and prev_idx != self.session.current_index:
            self._show_item_at(prev_idx)

    def _on_next_clicked(self):
        active_sf = self.session.active_subfolder
        active_cf = self.session.active_class_filter
        active_st = getattr(self.session, "active_status_filter", "all")
        next_idx = self.session.get_next_index(self.session.current_index, active_sf, active_cf, active_st)
        if next_idx is not None and next_idx != self.session.current_index:
            self._show_item_at(next_idx)

    def _on_rename_clicked(self):
        """Custom manual renaming."""
        item = self.session.get_current_item()
        if not item:
            return
            
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Image File",
            f"Enter new filename for {item.filename}:",
            text=item.filename
        )
        if ok and new_name.strip():
            if self.session.custom_rename(self.session.current_index, new_name.strip()):
                self._show_item_at(self.session.current_index)
                self.lbl_status_msg.setText(f"Renamed file to {item.filename}")
            else:
                QMessageBox.warning(self, "Rename Failed", "Failed to rename file.")

    def _on_delete_clicked(self):
        """Deletes current image with confirmation."""
        item = self.session.get_current_item()
        if not item:
            return
            
        reply = QMessageBox.question(
            self,
            "Confirm Delete Image",
            f"Are you sure you want to permanently delete:\n{item.filename}\n\nfrom disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            idx = self.session.current_index
            if self.session.delete_item(idx):
                self._refresh_subfolder_combo()
                self._refresh_status_filter_combo()
                if self.session.items:
                    filtered = self.session.get_filtered_indices(
                        self.session.active_subfolder,
                        self.session.active_class_filter,
                        getattr(self.session, "active_status_filter", "all")
                    )
                    next_target = idx if idx in filtered else (filtered[0] if filtered else 0)
                    self._show_item_at(min(next_target, len(self.session.items) - 1))
                else:
                    self.image_viewer.clear()
                    self._update_ui_state()
                self.lbl_status_msg.setText("Image deleted from disk.")

    def _on_remove_dup_clicked(self):
        """Batch remove '_dup' images with confirmation."""
        active_sf = self.session.active_subfolder
        dup_indices = self.session.find_dup_items(active_sf)
        
        scope_str = f" in '{active_sf}'" if active_sf and active_sf != "(All)" else ""
        if not dup_indices:
            QMessageBox.information(self, "No Duplicates Found", f"No images containing '_dup'{scope_str} were found.")
            return
            
        reply = QMessageBox.question(
            self,
            "Remove Duplicate Images",
            f"Found {len(dup_indices)} duplicate image(s) matching '_dup'{scope_str}.\n\n"
            f"Are you sure you want to permanently DELETE these {len(dup_indices)} image(s) from disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            removed_count = self.session.remove_dup_images(active_sf)
            self._refresh_subfolder_combo()
            if self.session.items:
                self._show_item_at(min(self.session.current_index, len(self.session.items) - 1))
            else:
                self.image_viewer.clear()
                self._update_ui_state()
            QMessageBox.information(self, "Duplicates Removed", f"Successfully deleted {removed_count} duplicate image(s).")

    def _on_open_check_menu(self):
        if not self.session.items:
            QMessageBox.information(self, "No Images", "Please open an images folder first.")
            return
            
        dialog = CheckViewDialog(self.session, self.session.current_index, self)
        dialog.image_selected.connect(self._show_item_at)
        dialog.image_deleted.connect(lambda idx: self._show_item_at(min(self.session.current_index, len(self.session.items) - 1)) if self.session.items else self.image_viewer.clear())
        dialog.exec()
        self._refresh_subfolder_combo()
        self._refresh_status_filter_combo()
        self._update_ui_state()

    def _on_export_staging_clicked(self):
        """Exports all approved images to dataset staging directory."""
        if not self.session.items:
            QMessageBox.information(self, "No Images", "Please open an images folder first.")
            return

        active_sf = self.session.active_subfolder
        count, staging_dir = self.session.export_all_approved_to_staging(active_sf)
        sf_str = f" in folder '{active_sf}'" if active_sf and active_sf != "(All)" else ""

        self.lbl_status_msg.setText(f"Staged {count} approved plate(s) into {staging_dir.name}/")
        
        box = QMessageBox(self)
        box.setWindowTitle("Dataset Staging Complete")
        box.setText(
            f"Successfully staged {count} approved plate(s){sf_str} into:\n"
            f"{staging_dir}\n\n"
            f"License plates for reading task are stored in folders named:\n"
            f"  {{type}}_{{d3}}_{{d4}}_{{province}}\n\n"
            f"These images are organized and waiting for building dataset."
        )
        btn_open = box.addButton("Open Staging Folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() == btn_open:
            self._on_open_staging_folder()

    def _on_open_staging_folder(self):
        """Opens staging folder in file explorer."""
        staging_dir = self.session.staging_dir
        staging_dir.mkdir(parents=True, exist_ok=True)
        import os
        try:
            os.startfile(str(staging_dir))
        except Exception as e:
            QMessageBox.warning(self, "Open Folder Error", f"Could not open {staging_dir}: {e}")

    def _on_open_owod_studio(self):
        try:
            from owod.ui.main_window import OWODMainWindow
            self.owod_window = OWODMainWindow()
            self.owod_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error Opening OWOD Studio", str(e))

    def _stop_workers(self):
        """Stops and waits for background workers safely."""
        if self._active_worker and self._active_worker.isRunning():
            self._active_worker.cancel()
            self._active_worker.wait(2000)
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.cancel()
            self._batch_worker.wait(2000)

    def closeEvent(self, event):
        self._stop_workers()
        self.session.save_session()
        event.accept()

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121217;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            #headerCard, #editPanelCard {
                background-color: #1a1a23;
                border: 1px solid #2a2a38;
                border-radius: 8px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            #lblFileName {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
            #lblFolderPath {
                font-size: 12px;
                color: #a0a0b8;
            }
            #groupBoxAI {
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #3b3b55;
                background-color: #1f1f2c;
                border-radius: 6px;
                margin-top: 4px;
                padding-top: 10px;
            }
            #groupBoxAI::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #a78bfa;
            }
            #groupBoxEdit {
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #2e2e40;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            #groupBoxEdit::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox {
                background-color: #262634;
                color: #ffffff;
                border: 1px solid #3b3b4f;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #6366f1;
            }
            #txtPlateField {
                font-size: 15px;
                font-weight: bold;
                color: #38bdf8;
            }
            QPushButton {
                background-color: #2e2e40;
                color: #ffffff;
                border: 1px solid #43435c;
                border-radius: 6px;
                padding: 7px 14px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3e3e56;
                border-color: #6366f1;
            }
            QPushButton:pressed {
                background-color: #272737;
            }
            #btnOpenFolder {
                background-color: #3730a3;
                border-color: #4f46e5;
            }
            #btnOpenFolder:hover {
                background-color: #4338ca;
            }
            #btnCleanDup {
                background-color: #7c2d12;
                border-color: #ea580c;
            }
            #btnCleanDup:hover {
                background-color: #9a3412;
            }
            #btnAiReadAll {
                background-color: #4338ca;
                border-color: #6366f1;
            }
            #btnAiReadAll:hover {
                background-color: #4f46e5;
            }
            #btnAiRead {
                background-color: #4f46e5;
                border-color: #6366f1;
                font-size: 11px;
                padding: 4px 8px;
            }
            #btnAiRead:hover {
                background-color: #4338ca;
            }
            #btnApplySingle {
                background-color: #1e293b;
                border: 1px solid #475569;
                font-size: 11px;
                font-weight: bold;
                padding: 3px 6px;
                color: #38bdf8;
            }
            #btnApplySingle:hover {
                background-color: #334155;
                border-color: #38bdf8;
            }
            #btnApplySingle:disabled {
                background-color: #1a1a24;
                border-color: #2a2a36;
                color: #555566;
            }
            #btnReplaceSelected {
                background-color: #0284c7;
                border-color: #0ea5e9;
                font-size: 12px;
                padding: 6px 10px;
            }
            #btnReplaceSelected:hover {
                background-color: #0369a1;
            }
            #btnReplaceAll {
                background-color: #059669;
                border-color: #10b981;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
            }
            #btnReplaceAll:hover {
                background-color: #047857;
            }
            #btnCheckMenu {
                background-color: #0d9488;
                border-color: #14b8a6;
            }
            #btnCheckMenu:hover {
                background-color: #0f766e;
            }
            #btnRenameFile {
                background-color: #334155;
                border-color: #475569;
                font-size: 12px;
                padding: 6px;
            }
            #btnRenameFile:hover {
                background-color: #475569;
            }
            #btnDeleteFile {
                background-color: #991b1b;
                border-color: #dc2626;
                font-size: 12px;
                padding: 6px;
            }
            #btnDeleteFile:hover {
                background-color: #b91c1c;
            }
            #btnApprove {
                background-color: #16a34a;
                border-color: #22c55e;
                font-size: 14px;
                padding: 9px;
            }
            #btnApprove:hover {
                background-color: #15803d;
            }
            #btnNotPlate {
                background-color: #2563eb;
                border-color: #3b82f6;
                padding: 7px;
            }
            #btnNotPlate:hover {
                background-color: #1d4ed8;
            }
            #btnCannotRead {
                background-color: #dc2626;
                border-color: #ef4444;
                padding: 7px;
            }
            #btnCannotRead:hover {
                background-color: #b91c1c;
            }
            #btnSkip {
                background-color: #d97706;
                border-color: #f59e0b;
                padding: 7px;
            }
            #btnSkip:hover {
                background-color: #b45309;
            }
            #btnNav {
                padding: 6px 12px;
            }
            #badgeNeutral {
                background-color: #282836;
                color: #e0e0e0;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeWarning {
                background-color: #78350f;
                color: #fde047;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeSuccess {
                background-color: #14532d;
                color: #4ade80;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeBlue {
                background-color: #1e3a8a;
                color: #93c5fd;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeDanger {
                background-color: #7f1d1d;
                color: #fca5a5;
                padding: 5px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeNeutral:hover, #badgeWarning:hover, #badgeSuccess:hover, #badgeBlue:hover, #badgeDanger:hover {
                border: 1px solid #ffffff;
            }
            #appProgressBar {
                background-color: #262634;
                border: 1px solid #3b3b4f;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                min-height: 18px;
            }
            #appProgressBar::chunk {
                background-color: #10b981;
                border-radius: 3px;
            }
            QCheckBox {
                color: #e2e8f0;
                font-size: 12px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #4b5563;
                background-color: #1e293b;
            }
            QCheckBox::indicator:checked {
                background-color: #3b82f6;
                border-color: #60a5fa;
            }
        """)
