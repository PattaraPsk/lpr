"""
Searchable ComboBox with autocomplete and filtering for Thailand Provinces and special options.
"""

from PyQt6.QtWidgets import QComboBox, QCompleter
from PyQt6.QtCore import Qt, QSortFilterProxyModel, QStringListModel
from .provinces import get_province_entries

class SearchableProvinceComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Populate items
        self._entries = get_province_entries() # list of (val, disp)
        for val, disp in self._entries:
            self.addItem(disp, userData=val)
            
        # Setup completer with substring matching (MatchContains)
        self.completer = QCompleter(self)
        self.completer.setModel(self.model())
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.setCompleter(self.completer)
        
        # When completer activated, update line edit
        self.completer.activated.connect(self._on_completer_activated)
        
        # Style line edit
        line_edit = self.lineEdit()
        if line_edit:
            line_edit.setPlaceholderText("Search province or type + / - ...")
            line_edit.returnPressed.connect(self._on_return_pressed)
            
    def _on_completer_activated(self, text):
        idx = self.findText(text)
        if idx >= 0:
            self.setCurrentIndex(idx)
            
    def _on_return_pressed(self):
        text = self.currentText().strip()
        idx = self.findText(text, Qt.MatchFlag.MatchContains)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def set_province_value(self, val: str):
        """Set current item by value (val is either '+', '-', or Thai province name)"""
        if not val:
            self.setCurrentIndex(-1)
            self.setEditText("")
            return
            
        val_str = str(val).strip()
        # First check userData match
        for i in range(self.count()):
            if self.itemData(i) == val_str:
                self.setCurrentIndex(i)
                return
                
        # Second check display text match or partial match
        for i in range(self.count()):
            text = self.itemText(i)
            if val_str.lower() in text.lower():
                self.setCurrentIndex(i)
                return
                
        # If not found, set text directly
        self.setEditText(val_str)

    def get_province_value(self) -> str:
        """Returns the canonical province value (e.g., '+', '-', or Thai name)"""
        idx = self.currentIndex()
        if idx >= 0:
            data = self.itemData(idx)
            if data:
                return str(data)
                
        # Fallback to current text
        text = self.currentText().strip()
        # Check if text matches any display text
        for val, disp in self._entries:
            if text == disp or text == val:
                return val
        return text
