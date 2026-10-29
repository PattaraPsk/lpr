"""
High-performance Zoomable & Pannable Image Viewer using QGraphicsView.
Supports rendering YOLO bounding boxes with color-coded classes, translucent fills,
and toggleable labels.
"""

from typing import List, Dict, Any, Optional
from PyQt6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsSimpleTextItem, QWidget, QVBoxLayout, QHBoxLayout, QToolButton, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QWheelEvent, QMouseEvent, QColor, QFont, QPen, QBrush

# High-contrast color palette for YOLO object detection classes
DEFAULT_CLASS_COLORS = [
    QColor("#06b6d4"),  # 0: Front (Cyan)
    QColor("#a855f7"),  # 1: Rear (Purple)
    QColor("#eab308"),  # 2: LicensePlate (Gold/Yellow)
    QColor("#22c55e"),  # 3: Car (Green)
    QColor("#f97316"),  # 4: Motorcycle (Orange)
    QColor("#3b82f6"),  # 5: Truck (Blue)
    QColor("#ec4899"),  # 6: Bus (Pink)
    QColor("#14b8a6"),  # 7: Teal
    QColor("#8b5cf6"),  # 8: Indigo
    QColor("#ef4444"),  # 9: Red
]


class ZoomPanGraphicsView(QGraphicsView):
    zoom_changed = pyqtSignal(float)
    box_clicked = pyqtSignal(int)  # Emits index of clicked box
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene.addItem(self.pixmap_item)
        
        # Bounding box graphics items tracking
        self._box_items: List[QGraphicsRectItem] = []
        self._label_items: List[Any] = []
        self._raw_boxes: List[Dict[str, Any]] = []
        self.selected_box_index: Optional[int] = None
        self.show_boxes: bool = True
        self.show_labels: bool = True
        
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
            
        self.clear_bounding_boxes()
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self._has_image = True
        self.fit_to_window()
        return True
        
    def clear_image(self):
        self.clear_bounding_boxes()
        self.pixmap_item.setPixmap(QPixmap())
        self.scene.setSceneRect(QRectF())
        self._has_image = False
        self._current_scale = 1.0
        self.resetTransform()
        self.zoom_changed.emit(self._current_scale)

    def set_bounding_boxes(self, boxes: List[Dict[str, Any]]):
        """
        Renders bounding boxes onto the image scene.
        Each box dict format:
        {
            'cls_id': int,
            'cls_name': str,
            'bbox_norm': (cx, cy, w, h)   # or 'box_xyxy': (x1, y1, x2, y2)
        }
        """
        self.clear_bounding_boxes()
        self._raw_boxes = [dict(b) for b in boxes] if boxes else []
        
        if not self._has_image or not boxes:
            return
            
        img_w = self.pixmap_item.pixmap().width()
        img_h = self.pixmap_item.pixmap().height()
        
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        
        for idx, b in enumerate(boxes):
            cls_id = b.get("cls_id", 0)
            cls_name = b.get("cls_name", f"Class {cls_id}")
            
            if "bbox_norm" in b:
                cx, cy, bw, bh = b["bbox_norm"]
                x1 = (cx - bw / 2.0) * img_w
                y1 = (cy - bh / 2.0) * img_h
                w = bw * img_w
                h = bh * img_h
            elif "box_xyxy" in b:
                x1, y1, x2, y2 = b["box_xyxy"]
                w = x2 - x1
                h = y2 - y1
            else:
                continue
                
            color = DEFAULT_CLASS_COLORS[cls_id % len(DEFAULT_CLASS_COLORS)]
            
            # 1. Main Bounding Box
            rect_item = QGraphicsRectItem(QRectF(x1, y1, w, h))
            pen = QPen(color, 2.5)
            pen.setCosmetic(True)  # Keeps line width consistent during zoom
            rect_item.setPen(pen)
            rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
            rect_item.setToolTip(f"#{idx+1}: {cls_name} (ID: {cls_id})\nBox: {w:.1f} × {h:.1f} px")
            rect_item.setVisible(self.show_boxes)
            self.scene.addItem(rect_item)
            self._box_items.append(rect_item)
            
            # 2. Label Banner Badge
            label_text = f" {cls_name} "
            text_item = QGraphicsSimpleTextItem(label_text)
            text_item.setFont(font)
            text_item.setBrush(QBrush(QColor("#ffffff")))
            
            # Compute label background rectangle
            text_rect = text_item.boundingRect()
            badge_h = text_rect.height() + 4
            badge_w = text_rect.width() + 4
            badge_y = max(0.0, y1 - badge_h)
            
            badge_bg = QGraphicsRectItem(QRectF(x1, badge_y, badge_w, badge_h))
            badge_bg.setPen(QPen(Qt.PenStyle.NoPen))
            badge_bg.setBrush(QBrush(color))
            
            text_item.setPos(x1 + 2, badge_y + 2)
            
            badge_bg.setVisible(self.show_boxes and self.show_labels)
            text_item.setVisible(self.show_boxes and self.show_labels)
            
            self.scene.addItem(badge_bg)
            self.scene.addItem(text_item)
            self._label_items.extend([badge_bg, text_item])

    def clear_bounding_boxes(self):
        """Removes all bounding box and label graphics items from the scene."""
        for item in self._box_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        for item in self._label_items:
            if item.scene() == self.scene:
                self.scene.removeItem(item)
        self._box_items.clear()
        self._label_items.clear()
        self._raw_boxes = []
        self.selected_box_index = None

    def _find_box_at(self, scene_pt: QPointF) -> int:
        """Finds bounding box index at scene coordinates. Prioritizes smaller inner boxes."""
        if not self.show_boxes or not self._box_items:
            return -1
        matches = []
        for idx, rect_item in enumerate(self._box_items):
            rect = rect_item.rect()
            if rect.contains(scene_pt):
                area = rect.width() * rect.height()
                matches.append((area, idx))
        if matches:
            matches.sort(key=lambda x: x[0])  # Smaller box (e.g. plate inside car) first
            return matches[0][1]
        return -1

    def select_box(self, index: Optional[int]):
        """Sets the selected bounding box and highlights it visually."""
        self.selected_box_index = index
        self._update_box_appearances()

    def _update_box_appearances(self):
        """Applies normal or highlighted selection style to all bounding boxes."""
        for idx, rect_item in enumerate(self._box_items):
            if idx >= len(self._raw_boxes):
                continue
            b = self._raw_boxes[idx]
            cls_id = b.get("cls_id", 0)
            color = DEFAULT_CLASS_COLORS[cls_id % len(DEFAULT_CLASS_COLORS)]
            
            if idx == self.selected_box_index:
                # Highlighted selected box: bright white border and rich translucent fill
                pen = QPen(QColor("#ffffff"), 3.5)
                pen.setCosmetic(True)
                rect_item.setPen(pen)
                rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 110)))
                rect_item.setZValue(10)
            else:
                pen = QPen(color, 2.5)
                pen.setCosmetic(True)
                rect_item.setPen(pen)
                rect_item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 35)))
                rect_item.setZValue(1)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and hasattr(self, '_press_pos'):
            dist = (event.pos() - self._press_pos).manhattanLength()
            if dist <= 5:  # Click without dragging
                scene_pt = self.mapToScene(event.pos())
                clicked_idx = self._find_box_at(scene_pt)
                if clicked_idx != -1:
                    self.select_box(clicked_idx)
                    self.box_clicked.emit(clicked_idx)
                else:
                    self.select_box(None)
                    self.box_clicked.emit(-1)

    def toggle_boxes(self, visible: Optional[bool] = None):
        if visible is None:
            self.show_boxes = not self.show_boxes
        else:
            self.show_boxes = visible
            
        for item in self._box_items:
            item.setVisible(self.show_boxes)
        for item in self._label_items:
            item.setVisible(self.show_boxes and self.show_labels)

    def toggle_labels(self, visible: Optional[bool] = None):
        if visible is None:
            self.show_labels = not self.show_labels
        else:
            self.show_labels = visible
            
        for item in self._label_items:
            item.setVisible(self.show_boxes and self.show_labels)

    def fit_to_window(self):
        if not self._has_image:
            return
        self.resetTransform()
        rect = self.scene.sceneRect()
        if rect.width() > 0 and rect.height() > 0:
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
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
    box_clicked = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Graphics view
        self.view = ZoomPanGraphicsView(self)
        self.view.box_clicked.connect(self.box_clicked.emit)
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
            QToolButton:checked {
                background-color: #4f46e5;
                border-color: #818cf8;
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
        
        # Toggle Bounding Boxes & Labels (for YOLO mode)
        self.btn_toggle_boxes = QToolButton()
        self.btn_toggle_boxes.setCheckable(True)
        self.btn_toggle_boxes.setChecked(True)
        self.btn_toggle_boxes.setText("🔲 Boxes (B)")
        self.btn_toggle_boxes.setToolTip("Toggle Bounding Box visibility [Shortcut: B]")
        self.btn_toggle_boxes.toggled.connect(self.view.toggle_boxes)
        c_layout.addWidget(self.btn_toggle_boxes)
        
        self.btn_toggle_labels = QToolButton()
        self.btn_toggle_labels.setCheckable(True)
        self.btn_toggle_labels.setChecked(True)
        self.btn_toggle_labels.setText("🏷️ Labels (L)")
        self.btn_toggle_labels.setToolTip("Toggle Class Label text visibility [Shortcut: L]")
        self.btn_toggle_labels.toggled.connect(self.view.toggle_labels)
        c_layout.addWidget(self.btn_toggle_labels)
        
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

    def set_bounding_boxes(self, boxes: List[Dict[str, Any]]):
        self.view.set_bounding_boxes(boxes)

    def select_box(self, index: Optional[int]):
        self.view.select_box(index)

    def clear_bounding_boxes(self):
        self.view.clear_bounding_boxes()
        
    def clear(self):
        self.view.clear_image()
        self.lbl_dims.setText("No image loaded")
        self.lbl_zoom.setText("100%")
