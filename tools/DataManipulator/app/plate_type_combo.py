"""
ComboBox with color preview and descriptions for Thai License Plate Types.
"""

from PyQt6.QtWidgets import QComboBox, QCompleter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap, QIcon
from .plate_types import get_plate_type_entries, PLATE_TYPES_DATA, SPECIAL_PLATE_TYPES, match_plate_type

class PlateTypeComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self._populate_items()
        
        # Setup completer
        self.completer = QCompleter(self)
        self.completer.setModel(self.model())
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setCompleter(self.completer)
        
        self.completer.activated.connect(self._on_completer_activated)
        
        line_edit = self.lineEdit()
        if line_edit:
            line_edit.setPlaceholderText("Select or search plate type...")

    def _create_color_icon(self, bg_hex: str, text_hex: str, border_hex: str = "#000000", badge_text: str = "••") -> QIcon:
        """Create a small 28x16 icon simulating plate colors and type"""
        pixmap = QPixmap(28, 16)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill background
        painter.setBrush(QColor(bg_hex))
        painter.setPen(QColor(border_hex))
        painter.drawRoundedRect(0, 0, 27, 15, 3, 3)
        
        # Draw small inner letter indicator / badge
        painter.setPen(QColor(text_hex))
        font = painter.font()
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, badge_text)
        painter.end()
        return QIcon(pixmap)

    def _populate_items(self):
        # Add special entries
        for val, disp in SPECIAL_PLATE_TYPES:
            self.addItem(disp, userData=val)
            
        # Add plate types with visual color icons and badges
        for pt in PLATE_TYPES_DATA:
            icon = self._create_color_icon(
                pt["bg_color"], pt["text_color"], pt["border_color"], pt.get("badge_text", "••")
            )
            self.addItem(icon, pt["display"], userData=pt["id"])


    def _on_completer_activated(self, text):
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def set_plate_type_value(self, val: str):
        if not val:
            self.setCurrentIndex(-1)
            self.setEditText("")
            return
            
        canonical = match_plate_type(val)
        
        # Check userData match
        for i in range(self.count()):
            if self.itemData(i) == canonical or self.itemData(i) == val:
                self.setCurrentIndex(i)
                return
                
        # Check text match
        for i in range(self.count()):
            if canonical.lower() in self.itemText(i).lower():
                self.setCurrentIndex(i)
                return
                
        # Fallback to direct text
        self.setEditText(str(val))

    def get_plate_type_value(self) -> str:
        idx = self.currentIndex()
        if idx >= 0:
            data = self.itemData(idx)
            if data:
                return str(data)
                
        text = self.currentText().strip()
        # Try matching text
        for pt in PLATE_TYPES_DATA:
            if text == pt["display"] or text == pt["id"]:
                return pt["id"]
        for val, disp in SPECIAL_PLATE_TYPES:
            if text == disp or text == val:
                return val
        return text
