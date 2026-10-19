"""
High-performance Zoomable & Pannable Image Viewer using QGraphicsView.
"""

from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent, QColor, QFont

class ZoomPanGraphicsView(QGraphicsView):
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        
        # Rendering settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor("#1e1e24"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        
        self._current_scale = 1.0
        self._min_scale = 0.05
        self._max_scale = 50.0
        self._has_image = False
        
    def load_image(self, file_path: str) -> bool:
        """Loads an image from file path and resets view to fit."""
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            return False
            
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = True
        self.fit_to_window()
        return True
        
    def clear_image(self):
        self.pixmap_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF())
        self._has_image = False
        self._current_scale = 1.0
        self.resetTransform()
        self.zoom_changed.emit(self._current_scale)

    def fit_to_window(self):
        if not self._has_image:
            return
        self.resetTransform()
        rect = self.scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
            # Calculate actual scale factor
            self._current_scale = self.transform().m11()
            self.zoom_changed.emit(self._current_scale)

    def reset_zoom(self):
        if not self._has_image:
            return
        self.resetTransform()
        self._current_scale = 1.0
        self.centerOn(self.pixmap_item)
        self.zoom_changed.emit(self._current_scale)

    def zoom_in(self, factor=1.25):
        self.zoom(factor)

    def zoom_out(self, factor=0.8):
        self.zoom(factor)

    def zoom(self, factor):
        if not self._has_image:
            return
        new_scale = self._current_scale * factor
        if self._min_scale <= new_scale <= self._max_scale:
            self.scale(factor, factor)
            self._current_scale = new_scale
            self.zoom_changed.emit(self._current_scale)

    def wheelEvent(self, event: QWheelEvent):
        if not self._has_image:
            return
        # Zoom on wheel scroll
        angle = event.angleDelta().y()
        if angle > 0:
            factor = 1.15
        elif angle < 0:
            factor = 1.0 / 1.15
        else:
            return
            
        self.zoom(factor)
        event.accept()

class ImageViewerWidget(QWidget):
    """
    Compound widget containing the zoom/pan view and overlay/toolbar controls.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Graphics view
        self.view = ZoomPanGraphicsView(self)
        layout.addWidget(self.view)
        
        # Bottom control bar
        self.control_bar = QWidget(self)
        self.control_bar.setStyleSheet("""
            QWidget {
                background-color: #2b2b36;
                color: #e0e0e0;
                font-size: 12px;
                border-top: 1px solid #3c3c4a;
            }
            QToolButton {
                background-color: #3b3b4a;
                color: #ffffff;
                border: 1px solid #4a4a5a;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
            }
            QToolButton:hover {
                background-color: #505068;
                border-color: #6366f1;
            }
            QToolButton:pressed {
                background-color: #38384a;
            }
        """)
        c_layout = QHBoxLayout(self.control_bar)
        c_layout.setContentsMargins(8, 6, 8, 6)
        c_layout.setSpacing(8)
        
        # Info labels
        self.lbl_dims = QLabel("No image loaded")
        c_layout.addWidget(self.lbl_dims)
        
        c_layout.addStretch()
        
        # Zoom Controls
        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("－")
        self.btn_zoom_out.setToolTip("Zoom Out (Wheel Down)")
        self.btn_zoom_out.clicked.connect(self.view.zoom_out)
        c_layout.addWidget(self.btn_zoom_out)
        
        self.lbl_zoom = QLabel("100%")
        self.lbl_zoom.setMinimumWidth(50)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_layout.addWidget(self.lbl_zoom)
        
        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("＋")
        self.btn_zoom_in.setToolTip("Zoom In (Wheel Up)")
        self.btn_zoom_in.clicked.connect(self.view.zoom_in)
        c_layout.addWidget(self.btn_zoom_in)
        
        self.btn_fit = QToolButton()
        self.btn_fit.setText("Fit View")
        self.btn_fit.setToolTip("Fit Image to Window")
        self.btn_fit.clicked.connect(self.view.fit_to_window)
        c_layout.addWidget(self.btn_fit)
        
        self.btn_actual = QToolButton()
        self.btn_actual.setText("1:1")
        self.btn_actual.setToolTip("Actual Size (100%)")
        self.btn_actual.clicked.connect(self.view.reset_zoom)
        c_layout.addWidget(self.btn_actual)
        
        layout.addWidget(self.control_bar)
        
        self.view.zoom_changed.connect(self._on_zoom_changed)
        
    def _on_zoom_changed(self, scale):
        pct = int(round(scale * 100))
        self.lbl_zoom.setText(f"{pct}%")
        
    def load_image(self, file_path: str) -> bool:
        success = self.view.load_image(file_path)
        if success:
            pixmap = self.view.pixmap_item.pixmap()
            self.lbl_dims.setText(f"{pixmap.width()} × {pixmap.height()} px | Drag to pan • Scroll to zoom")
        else:
            self.lbl_dims.setText("Failed to load image")
        return success
        
    def clear(self):
        self.view.clear_image()
        self.lbl_dims.setText("No image loaded")
        self.lbl_zoom.setText("100%")
