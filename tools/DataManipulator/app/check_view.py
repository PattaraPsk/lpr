"""
Check / Review Menu Dialog for inspecting and jumping to images (e.g. skipped, pending, approved).
Supports 9 columns including Subfolder and Plate Type, with Subfolder filtering, AI Difference filter,
and quick AI prediction replacement.
"""

import re
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QButtonGroup, QRadioButton, QAbstractItemView, QFrame,
    QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from typing import List, Optional
from .session_manager import ImageItem, SessionManager

STATUS_COLORS = {
    "approved": ("#22c55e", "#dcfce7"),     # Green
    "not_plate": ("#3b82f6", "#dbeafe"),    # Blue
    "cannot_read": ("#ef4444", "#fee2e2"),  # Red
    "skipped": ("#f59e0b", "#fef3c7"),      # Amber/Yellow
    "pending": ("#6b7280", "#f3f4f6"),      # Gray
}

STATUS_LABELS = {
    "approved": "Approved",
    "not_plate": "Not Plate (+)",
    "cannot_read": "Cannot Read (-)",
    "skipped": "Skipped",
    "pending": "Pending",
}

class CheckViewDialog(QDialog):
    image_selected = pyqtSignal(int)
    image_deleted = pyqtSignal(int)
    
    def __init__(self, session: SessionManager, current_idx: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Check & Review Image Status [a1.3.4.1 AI Enabled]")
        self.resize(1200, 680)
        self.session = session
        self.items = session.items
        self.selected_index = current_idx
        
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e24;
                color: #e0e0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QLabel {
                color: #e0e0e0;
            }
            QLineEdit, QComboBox {
                background-color: #2b2b36;
                color: #ffffff;
                border: 1px solid #4a4a5a;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #6366f1;
            }
            QTableWidget {
                background-color: #252530;
                color: #f0f0f0;
                gridline-color: #383848;
                border: 1px solid #383848;
                border-radius: 6px;
                selection-background-color: #4f46e5;
                selection-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2b2b36;
                color: #a0a0b0;
                padding: 8px 6px;
                font-weight: bold;
                border: 1px solid #383848;
            }
            QPushButton {
                background-color: #3b3b4a;
                color: #ffffff;
                border: 1px solid #4a4a5a;
                border-radius: 6px;
                padding: 6px 14px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #505068;
                border-color: #6366f1;
            }
            QPushButton:pressed {
                background-color: #38384a;
            }
            QRadioButton {
                color: #d0d0e0;
                font-size: 13px;
                spacing: 6px;
                padding: 4px 6px;
            }
            QRadioButton:hover {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header title & counts
        title_box = QHBoxLayout()
        title_lbl = QLabel("📋 Image Review & Status Overview")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_box.addWidget(title_lbl)
        
        title_box.addStretch()
        
        self.count_badge = QLabel(f"Total: {len(self.items)} images")
        self.count_badge.setStyleSheet("color: #9ca3af; font-weight: bold; font-size: 13px;")
        title_box.addWidget(self.count_badge)
        layout.addLayout(title_box)
        
        # Filter & Search Bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(10)
        
        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search filename, plate type, d3, d4, province...")
        self.search_input.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self.search_input, 2)
        
        # Subfolder filter combobox
        lbl_sub = QLabel("Subfolder:")
        lbl_sub.setStyleSheet("font-weight: bold; color: #a78bfa;")
        top_bar.addWidget(lbl_sub)
        
        self.combo_subfolder_filter = QComboBox()
        self.combo_subfolder_filter.addItem("All Subfolders", userData="(All)")
        self.combo_subfolder_filter.addItem("(Root Folder)", userData="(Root)")
        for sf in self.session.get_subfolders():
            self.combo_subfolder_filter.addItem(sf, userData=sf)
            
        if self.session.active_subfolder:
            idx = self.combo_subfolder_filter.findData(self.session.active_subfolder)
            if idx >= 0:
                self.combo_subfolder_filter.setCurrentIndex(idx)
                
        self.combo_subfolder_filter.currentIndexChanged.connect(self._apply_filter)
        top_bar.addWidget(self.combo_subfolder_filter, 1)
        
        layout.addLayout(top_bar)
        
        # Status Filter Buttons
        self.filter_group = QButtonGroup(self)
        self.filters = [
            ("all", "All"),
            ("skipped", "Skipped ⚠️"),
            ("pending", "Pending ⏳"),
            ("approved", "Approved ✅"),
            ("not_plate", "Not Plate 🚫"),
            ("cannot_read", "Cannot Read ❓"),
            ("dup", "Duplicates (_dup) 🧹"),
            ("ai_diff", "AI Diff ⚠️"),
        ]
        
        filter_widget = QFrame()
        f_layout = QHBoxLayout(filter_widget)
        f_layout.setContentsMargins(0, 0, 0, 0)
        f_layout.setSpacing(8)
        
        self.current_filter = "all"
        for code, label in self.filters:
            btn = QRadioButton(label)
            if code == "all":
                btn.setChecked(True)
            btn.toggled.connect(lambda checked, c=code: self._on_filter_changed(checked, c))
            self.filter_group.addButton(btn)
            f_layout.addWidget(btn)
            
        layout.addWidget(filter_widget)
        
        # Table with 9 columns (including Subfolder)
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "#", "Status", "Subfolder", "Filename", "Name", "Plate Type", "d3", "d4", "Province"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch) # Filename stretch
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)
        
        # Bottom controls
        bottom_bar = QHBoxLayout()
        hint_lbl = QLabel("💡 Tip: Double-click or select any row to edit, or use AI replace.")
        hint_lbl.setStyleSheet("color: #9ca3af; font-size: 12px;")
        bottom_bar.addWidget(hint_lbl)
        
        bottom_bar.addStretch()
        
        self.btn_replace_ai = QPushButton("⚡ Replace with AI (a1.3.4.1)")
        self.btn_replace_ai.setStyleSheet("background-color: #059669; color: white; font-weight: bold;")
        self.btn_replace_ai.setToolTip("Replace d3, d4, and Province of selected image with AI prediction")
        self.btn_replace_ai.clicked.connect(self._replace_selected_ai)
        bottom_bar.addWidget(self.btn_replace_ai)
        
        self.btn_delete = QPushButton("🗑️ Delete Image")
        self.btn_delete.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        self.btn_delete.clicked.connect(self._delete_selected)
        bottom_bar.addWidget(self.btn_delete)
        
        self.btn_jump = QPushButton("🚀 Jump to Selected Image")
        self.btn_jump.setStyleSheet("background-color: #4f46e5; color: white; font-weight: bold;")
        self.btn_jump.clicked.connect(self._jump_to_selected)
        bottom_bar.addWidget(self.btn_jump)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        bottom_bar.addWidget(self.btn_close)
        
        layout.addLayout(bottom_bar)
        
        # Populate table data
        self._populate_table()

    def _on_filter_changed(self, checked: bool, filter_code: str):
        if checked:
            self.current_filter = filter_code
            self._apply_filter()

    def _populate_table(self):
        self._filtered_indices = []
        search_text = self.search_input.text().lower().strip()
        selected_sf = self.combo_subfolder_filter.currentData()
        
        for idx, item in enumerate(self.items):
            # 1. Subfolder filter
            if selected_sf and selected_sf != "(All)":
                if selected_sf == "(Root)" and item.subfolder:
                    continue
                elif selected_sf != "(Root)" and item.subfolder != selected_sf:
                    continue
                    
            # 2. Status / Dup / AI Diff filter
            if self.current_filter == "dup":
                if not (re.search(r'_dup(\d+)?(_|$|\.)', item.filename, re.IGNORECASE) or "_dup" in item.filename.lower()):
                    continue
            elif self.current_filter == "ai_diff":
                comp = item.get_ai_comparison()
                if not comp["has_ai"] or comp["all_match"]:
                    continue
            elif self.current_filter != "all" and item.status != self.current_filter:
                continue
                
            # 3. Search query
            if search_text:
                searchable = f"{item.filename} {item.subfolder} {item.name} {item.plate_type} {item.d3} {item.d4} {item.province} {item.ai_d3} {item.ai_d4} {item.ai_province}".lower()
                if search_text not in searchable:
                    continue
                    
            self._filtered_indices.append(idx)
            
        self.table.setRowCount(len(self._filtered_indices))
        
        for row, idx in enumerate(self._filtered_indices):
            item = self.items[idx]
            comp = item.get_ai_comparison()
            
            # 0: # (Index 1-based)
            item_num = QTableWidgetItem(str(idx + 1))
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_num.setData(Qt.ItemDataRole.UserRole, idx)
            self.table.setItem(row, 0, item_num)
            
            # 1: Status Badge + AI match status
            status_text = STATUS_LABELS.get(item.status, item.status.capitalize())
            if comp["has_ai"]:
                if comp["all_match"]:
                    status_text += " [AI ✅]"
                else:
                    status_text += " [AI ⚠️]"
            item_status = QTableWidgetItem(status_text)
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            fg_color, _ = STATUS_COLORS.get(item.status, ("#9ca3af", "#1f2937"))
            item_status.setForeground(QColor(fg_color))
            self.table.setItem(row, 1, item_status)
            
            # 2: Subfolder
            sf_display = item.subfolder if item.subfolder else "(Root)"
            item_sf = QTableWidgetItem(sf_display)
            item_sf.setForeground(QColor("#a78bfa"))
            self.table.setItem(row, 2, item_sf)
            
            # 3: Filename
            self.table.setItem(row, 3, QTableWidgetItem(item.filename))
            
            # 4: Name
            self.table.setItem(row, 4, QTableWidgetItem(item.name))
            
            # 5: Plate Type
            item_pt = QTableWidgetItem(item.plate_type)
            item_pt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, item_pt)
            
            # 6: d3
            item_d3 = QTableWidgetItem(item.d3)
            item_d3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if comp["has_ai"]:
                item_d3.setToolTip(f"AI Prediction: '{item.ai_d3}' (Match: {comp['d3_match']})")
                if not comp["d3_match"]:
                    item_d3.setForeground(QColor("#facc15"))
            self.table.setItem(row, 6, item_d3)
            
            # 7: d4
            item_d4 = QTableWidgetItem(item.d4)
            item_d4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if comp["has_ai"]:
                item_d4.setToolTip(f"AI Prediction: '{item.ai_d4}' (Match: {comp['d4_match']})")
                if not comp["d4_match"]:
                    item_d4.setForeground(QColor("#facc15"))
            self.table.setItem(row, 7, item_d4)
            
            # 8: Province
            item_prov = QTableWidgetItem(item.province)
            if comp["has_ai"]:
                item_prov.setToolTip(f"AI Prediction: '{item.ai_province}' (Match: {comp['prov_match']})")
                if not comp["prov_match"]:
                    item_prov.setForeground(QColor("#facc15"))
            self.table.setItem(row, 8, item_prov)
            
        self.count_badge.setText(f"Showing {len(self._filtered_indices)} of {len(self.items)} images")
        
        for row in range(self.table.rowCount()):
            idx_item = self.table.item(row, 0)
            if idx_item and idx_item.data(Qt.ItemDataRole.UserRole) == self.selected_index:
                self.table.selectRow(row)
                break

    def _apply_filter(self):
        self._populate_table()

    def _on_row_double_clicked(self, item):
        self._jump_to_selected()

    def _jump_to_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        idx_item = self.table.item(row, 0)
        if idx_item:
            target_idx = idx_item.data(Qt.ItemDataRole.UserRole)
            self.selected_index = target_idx
            self.image_selected.emit(target_idx)
            self.accept()

    def _replace_selected_ai(self):
        """Replaces d3, d4, and Province of selected image with AI prediction."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        idx_item = self.table.item(row, 0)
        if not idx_item:
            return
            
        target_idx = idx_item.data(Qt.ItemDataRole.UserRole)
        item = self.items[target_idx]
        if not (item.ai_d3 or item.ai_d4 or item.ai_province):
            QMessageBox.information(self, "No AI Prediction", "No AI prediction available for this image yet. Run AI Reader first.")
            return
            
        self.session.apply_ai_to_item(target_idx, replace_d3=True, replace_d4=True, replace_prov=True)
        self._populate_table()
        QMessageBox.information(
            self,
            "Replaced with AI",
            f"Successfully updated image #{target_idx + 1} with AI values:\n"
            f"d3: {item.d3}\nd4: {item.d4}\nProvince: {item.province}"
        )

    def _delete_selected(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        idx_item = self.table.item(row, 0)
        if not idx_item:
            return
            
        target_idx = idx_item.data(Qt.ItemDataRole.UserRole)
        item = self.items[target_idx]
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete Image",
            f"Are you sure you want to permanently delete:\n{item.filename}\n\nfrom disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            if self.session.delete_item(target_idx):
                self.image_deleted.emit(target_idx)
                self._populate_table()
