"""
Check / Review Menu Dialog for inspecting and jumping to images (e.g. skipped, pending, approved).
Supports 8 columns including Plate Type.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QButtonGroup, QRadioButton, QAbstractItemView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from typing import List
from .session_manager import ImageItem

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
    
    def __init__(self, items: List[ImageItem], current_idx: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Check & Review Image Status")
        self.resize(1100, 640)
        self.items = items
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
            QLineEdit {
                background-color: #2b2b36;
                color: #ffffff;
                border: 1px solid #4a4a5a;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
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
                padding: 4px 8px;
            }
            QRadioButton:hover {
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Header title
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
        top_bar.setSpacing(12)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search filename, plate type, d3, d4, province...")
        self.search_input.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self.search_input, 2)
        
        # Status Filter Buttons
        self.filter_group = QButtonGroup(self)
        self.filters = [
            ("all", "All"),
            ("skipped", "Skipped ⚠️"),
            ("pending", "Pending ⏳"),
            ("approved", "Approved ✅"),
            ("not_plate", "Not Plate 🚫"),
            ("cannot_read", "Cannot Read ❓"),
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
            
        top_bar.addWidget(filter_widget, 3)
        layout.addLayout(top_bar)
        
        # Table with 8 columns
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "#", "Status", "Filename", "Name", "Plate Type", "d3", "d4", "Province"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table)
        
        # Bottom controls
        bottom_bar = QHBoxLayout()
        hint_lbl = QLabel("💡 Tip: Double-click or select any row and click 'Jump to Image' to edit.")
        hint_lbl.setStyleSheet("color: #9ca3af; font-size: 12px;")
        bottom_bar.addWidget(hint_lbl)
        
        bottom_bar.addStretch()
        
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
        
        for idx, item in enumerate(self.items):
            if self.current_filter != "all" and item.status != self.current_filter:
                continue
                
            if search_text:
                searchable = f"{item.filename} {item.name} {item.plate_type} {item.d3} {item.d4} {item.province}".lower()
                if search_text not in searchable:
                    continue
                    
            self._filtered_indices.append(idx)
            
        self.table.setRowCount(len(self._filtered_indices))
        
        for row, idx in enumerate(self._filtered_indices):
            item = self.items[idx]
            
            # # (Index 1-based)
            item_num = QTableWidgetItem(str(idx + 1))
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_num.setData(Qt.ItemDataRole.UserRole, idx)
            self.table.setItem(row, 0, item_num)
            
            # Status Badge
            status_text = STATUS_LABELS.get(item.status, item.status.capitalize())
            item_status = QTableWidgetItem(status_text)
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            fg_color, _ = STATUS_COLORS.get(item.status, ("#9ca3af", "#1f2937"))
            item_status.setForeground(QColor(fg_color))
            self.table.setItem(row, 1, item_status)
            
            # Filename
            self.table.setItem(row, 2, QTableWidgetItem(item.filename))
            
            # Name
            self.table.setItem(row, 3, QTableWidgetItem(item.name))
            
            # Plate Type
            item_pt = QTableWidgetItem(item.plate_type)
            item_pt.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 4, item_pt)
            
            # d3
            item_d3 = QTableWidgetItem(item.d3)
            item_d3.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 5, item_d3)
            
            # d4
            item_d4 = QTableWidgetItem(item.d4)
            item_d4.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, item_d4)
            
            # Province
            self.table.setItem(row, 7, QTableWidgetItem(item.province))
            
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
