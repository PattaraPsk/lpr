"""
Dark Theme Modern Stylesheet for OWOD License Plate & Pruning Studio.
"""

DARK_THEME_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #0f172a;
    color: #f1f5f9;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0f172a;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 10px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: #6366f1;
    color: #ffffff;
}

QTabBar::tab:hover:!selected {
    background-color: #334155;
    color: #e2e8f0;
}

QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 14px;
    font-weight: bold;
    color: #38bdf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 0 6px;
    background-color: #1e293b;
    border-radius: 4px;
}

QPushButton {
    background-color: #3b82f6;
    color: #ffffff;
    border: none;
    padding: 7px 14px;
    border-radius: 6px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #2563eb;
}

QPushButton:pressed {
    background-color: #1d4ed8;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}

QPushButton#btnSuccess {
    background-color: #10b981;
}
QPushButton#btnSuccess:hover {
    background-color: #059669;
}

QPushButton#btnDanger {
    background-color: #ef4444;
}
QPushButton#btnDanger:hover {
    background-color: #dc2626;
}

QPushButton#btnSecondary {
    background-color: #334155;
    color: #e2e8f0;
    border: 1px solid #475569;
}
QPushButton#btnSecondary:hover {
    background-color: #475569;
}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0b1120;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
    selection-background-color: #6366f1;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #6366f1;
}

QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #0b1120;
    border: 1px solid #334155;
    selection-background-color: #6366f1;
    color: #f8fafc;
}

QTreeWidget, QTableWidget, QPlainTextEdit, QTextEdit {
    background-color: #0b1120;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #f1f5f9;
    gridline-color: #1e293b;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    font-weight: 600;
}

QProgressBar {
    background-color: #0b1120;
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #6366f1;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QScrollBar:vertical {
    background: #0f172a;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
