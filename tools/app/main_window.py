"""
Main Application Window for Thai License Plate (LPR) Annotation & Review.
Updated to support subfolders, grouping by subfolder, custom rename,
single image delete, and batch removing '_dup' duplicate images.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QFileDialog, QMessageBox,
    QProgressBar, QFrame, QGroupBox, QGridLayout, QComboBox,
    QInputDialog
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QKeySequence, QIcon, QShortcut

from .session_manager import SessionManager, ImageItem
from .image_viewer import ImageViewerWidget
from .searchable_combo import SearchableProvinceComboBox
from .plate_type_combo import PlateTypeComboBox
from .check_view import CheckViewDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thai LPR Dataset Annotator & Reviewer")
        self.resize(1380, 890)
        self.setMinimumSize(1000, 650)
        
        self.session = SessionManager()
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
        
        # 2. Central Splitter (Left: Viewer, Right: Edit Panel)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        
        self.image_viewer = ImageViewerWidget(self)
        splitter.addWidget(self.image_viewer)
        
        self.edit_panel = self._create_edit_panel()
        splitter.addWidget(self.edit_panel)
        
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
        lbl_sf = QLabel("Group Subfolder:")
        lbl_sf.setStyleSheet("font-weight: bold; color: #a78bfa; margin-left: 4px;")
        h_layout.addWidget(lbl_sf)
        
        self.combo_subfolder = QComboBox()
        self.combo_subfolder.setObjectName("comboSubfolder")
        self.combo_subfolder.addItem("All Subfolders", userData="(All)")
        self.combo_subfolder.currentIndexChanged.connect(self._on_subfolder_changed)
        h_layout.addWidget(self.combo_subfolder)
        
        # Remove _dup Images Button
        self.btn_clean_dup = QPushButton("🧹 Remove _dup")
        self.btn_clean_dup.setObjectName("btnCleanDup")
        self.btn_clean_dup.setToolTip("Find and permanently remove images with '_dup' in their filename")
        self.btn_clean_dup.clicked.connect(self._on_remove_dup_clicked)
        h_layout.addWidget(self.btn_clean_dup)
        
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
        
        # Check Menu Button
        self.btn_check_menu = QPushButton("📋 Check Menu")
        self.btn_check_menu.setObjectName("btnCheckMenu")
        self.btn_check_menu.setToolTip("Open Review List to inspect skipped/approved images")
        self.btn_check_menu.clicked.connect(self._on_open_check_menu)
        h_layout.addWidget(self.btn_check_menu)
        
        return header

    def _create_edit_panel(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("editPanelCard")
        panel.setMinimumWidth(350)
        panel.setMaximumWidth(460)
        
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
        
        # Group Box: Plate Structure Editor
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
        self.txt_d3.returnPressed.connect(self._on_approve_clicked)
        grid.addWidget(lbl_d3, 2, 0)
        grid.addWidget(self.txt_d3, 2, 1)
        
        # 4. d4 field (numbers)
        lbl_d4 = QLabel("d4 (Number):")
        lbl_d4.setStyleSheet("font-weight: bold; color: #60a5fa;")
        self.txt_d4 = QLineEdit()
        self.txt_d4.setPlaceholderText("e.g. 1234, 9999, +")
        self.txt_d4.setObjectName("txtPlateField")
        self.txt_d4.returnPressed.connect(self._on_approve_clicked)
        grid.addWidget(lbl_d4, 3, 0)
        grid.addWidget(self.txt_d4, 3, 1)
        
        # 5. province field
        lbl_province = QLabel("Province:")
        lbl_province.setStyleSheet("font-weight: bold; color: #60a5fa;")
        self.combo_province = SearchableProvinceComboBox()
        self.combo_province.setObjectName("comboProvince")
        grid.addWidget(lbl_province, 4, 0)
        grid.addWidget(self.combo_province, 4, 1)
        
        p_layout.addWidget(gb_edit)
        
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
        
        # Correct / Approve Button
        self.btn_approve = QPushButton("✅ Correct / Approve (Enter)")
        self.btn_approve.setObjectName("btnApprove")
        self.btn_approve.clicked.connect(self._on_approve_clicked)
        p_layout.addWidget(self.btn_approve)
        
        # Not Plate Button
        self.btn_not_plate = QPushButton("🚫 Not Plate [_+_+_+_+] (Ctrl+N)")
        self.btn_not_plate.setObjectName("btnNotPlate")
        self.btn_not_plate.clicked.connect(self._on_not_plate_clicked)
        p_layout.addWidget(self.btn_not_plate)
        
        # Cannot Read Button
        self.btn_cannot_read = QPushButton("❓ Cannot Read [-_-_-_-_] (Ctrl+D)")
        self.btn_cannot_read.setObjectName("btnCannotRead")
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
        
        return panel

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
        QShortcut(QKeySequence("Ctrl+N"), self, self._on_not_plate_clicked)
        QShortcut(QKeySequence("Ctrl+D"), self, self._on_cannot_read_clicked)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_skip_clicked)
        QShortcut(QKeySequence("Delete"), self, self._on_delete_clicked)
        QShortcut(QKeySequence("Ctrl+Delete"), self, self._on_delete_clicked)
        QShortcut(QKeySequence("F2"), self, self._on_rename_clicked)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder:
            return
            
        success = self.session.load_folder(folder)
        if not success:
            QMessageBox.warning(self, "Error", "Failed to load selected folder.")
            return
            
        self._refresh_subfolder_combo()
        
        if not self.session.items:
            self.image_viewer.clear()
            self._update_ui_state()
            QMessageBox.information(self, "No Images", "No supported images found in this folder or its subfolders.")
            return
            
        self._show_item_at(self.session.current_index)
        self.lbl_status_msg.setText(f"Folder loaded with {len(self.session.items)} images. Resumed at #{self.session.current_index + 1}.")

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
        
        # Jump to first item in this subfolder if current item doesn't belong to it
        curr = self.session.get_current_item()
        filtered = self.session.get_filtered_indices(self.session.active_subfolder)
        if filtered:
            if not curr or self.session.current_index not in filtered:
                self._show_item_at(filtered[0])
            else:
                self._update_ui_state()

    def _show_item_at(self, index: int):
        if not (0 <= index < len(self.session.items)):
            return
            
        self.session.current_index = index
        item = self.session.items[index]
        
        # Load image into viewer
        self.image_viewer.load_image(str(item.path))
        
        # Update Form Fields
        sf_display = item.subfolder if item.subfolder else "(Root Folder)"
        self.lbl_current_subfolder.setText(f"📂 Folder: {sf_display}")
        self.lbl_current_filename.setText(item.filename)
        self.txt_name.setText(item.name)
        self.combo_plate_type.set_plate_type_value(item.plate_type)
        self.txt_d3.setText(item.d3)
        self.txt_d4.setText(item.d4)
        self.combo_province.set_province_value(item.province)
        
        # Update Status Badge
        self._update_status_display(item)
        
        # Update Stats & Progress
        self._update_ui_state()
        
        # Focus d3 for quick editing
        self.txt_d3.setFocus()
        self.txt_d3.selectAll()

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
        stats = self.session.get_stats(active_sf)
        total = stats["total"]
        
        self.badge_total.setText(f"Total: {total}")
        self.badge_left.setText(f"Left: {stats['left']}")
        self.badge_approved.setText(f"Approved: {stats['approved']}")
        self.badge_not_plate.setText(f"Not Plate: {stats['not_plate']}")
        self.badge_cannot_read.setText(f"Cannot Read: {stats['cannot_read']}")
        
        filtered = self.session.get_filtered_indices(active_sf)
        curr_idx = self.session.current_index
        
        if filtered and curr_idx in filtered:
            pos = filtered.index(curr_idx) + 1
            sf_label = f" ({active_sf})" if active_sf and active_sf != "(All)" else ""
            self.lbl_nav_counter.setText(f"{pos} / {len(filtered)}{sf_label}")
            pct = int((stats["processed"] / total) * 100) if total > 0 else 0
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}% ({stats['processed']}/{total} done, {stats['left']} left)")
        else:
            self.lbl_nav_counter.setText("0 / 0")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0%")
            
        self.btn_back.setEnabled(len(filtered) > 1 and curr_idx != filtered[0])
        self.btn_next.setEnabled(len(filtered) > 1 and curr_idx != filtered[-1])

    def _on_approve_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        name = self.txt_name.text()
        plate_type = self.combo_plate_type.get_plate_type_value()
        d3 = self.txt_d3.text()
        d4 = self.txt_d4.text()
        province = self.combo_province.get_province_value()
        
        self.session.approve_and_save(idx, name, plate_type, d3, d4, province)
        self.lbl_status_msg.setText(f"Saved & Approved image #{idx + 1}")
        
        next_idx = self.session.get_next_index(idx, self.session.active_subfolder)
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)
            self.lbl_status_msg.setText("Reached last image in folder!")

    def _on_not_plate_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        self.session.mark_not_plate(idx)
        self.lbl_status_msg.setText(f"Marked image #{idx + 1} as Not Plate (_+_+_+_+)")
        
        next_idx = self.session.get_next_index(idx, self.session.active_subfolder)
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)

    def _on_cannot_read_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        self.session.mark_cannot_read(idx)
        self.lbl_status_msg.setText(f"Marked image #{idx + 1} as Cannot Read (_-_-_-_-_)")
        
        next_idx = self.session.get_next_index(idx, self.session.active_subfolder)
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)

    def _on_skip_clicked(self):
        if not self.session.items:
            return
            
        idx = self.session.current_index
        self.session.mark_skipped(idx)
        self.lbl_status_msg.setText(f"Skipped image #{idx + 1}")
        
        next_idx = self.session.get_next_index(idx, self.session.active_subfolder)
        if next_idx is not None and next_idx != idx:
            self._show_item_at(next_idx)
        else:
            self._show_item_at(idx)

    def _on_back_clicked(self):
        prev_idx = self.session.get_prev_index(self.session.current_index, self.session.active_subfolder)
        if prev_idx is not None and prev_idx != self.session.current_index:
            self._show_item_at(prev_idx)

    def _on_next_clicked(self):
        next_idx = self.session.get_next_index(self.session.current_index, self.session.active_subfolder)
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
                if self.session.items:
                    filtered = self.session.get_filtered_indices(self.session.active_subfolder)
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
        self._update_ui_state()

    def closeEvent(self, event):
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
                background-color: #282838;
                color: #e5e7eb;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeWarning {
                background-color: #78350f;
                color: #fef3c7;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeSuccess {
                background-color: #14532d;
                color: #dcfce7;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeBlue {
                background-color: #1e3a8a;
                color: #dbeafe;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            #badgeDanger {
                background-color: #7f1d1d;
                color: #fee2e2;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            #appProgressBar {
                background-color: #22222d;
                border: 1px solid #363647;
                border-radius: 6px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 18px;
            }
            #appProgressBar::chunk {
                background-color: #4f46e5;
                border-radius: 5px;
            }
        """)
