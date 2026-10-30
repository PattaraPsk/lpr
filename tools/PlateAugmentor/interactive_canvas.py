#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Plate Canvas & Character Manipulator
================================================
Allows loading any reference license plate image and freely:
- Moving characters (drag & drop, arrow keys)
- Resizing characters (corner handle, scale sliders, font size)
- Rotating characters (top rotation handle, angle slider -180° to +180°)
- Changing character colors, opacity, and text
- Exporting composite plate image + YOLO bounding box annotations
- Saving / Loading canvas layout projects (.json)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

# Thai license plate font candidate paths
FONT_PATHS = [
    Path(__file__).resolve().parent.parent.parent / "dataset" / "fonts" / "rbz-licenseplate.ttf",
    Path(__file__).resolve().parent.parent.parent / "models" / "fonts" / "rbz-licenseplate.ttf",
    Path(__file__).resolve().parent.parent.parent / "model" / "fonts" / "rbz-licenseplate.ttf",
]

_THAI_FONT_FAMILY: Optional[str] = None
THAI_FONT_FAMILY = "rbz-licenseplate"

def get_thai_font_family() -> str:
    """Lazily load the Thai plate font once a QApplication instance exists."""
    global _THAI_FONT_FAMILY
    if _THAI_FONT_FAMILY is not None:
        return _THAI_FONT_FAMILY

    _THAI_FONT_FAMILY = "rbz-licenseplate"
    for fp in FONT_PATHS:
        if fp.exists():
            try:
                fid = QFontDatabase.addApplicationFont(str(fp))
                if fid >= 0:
                    fams = QFontDatabase.applicationFontFamilies(fid)
                    if fams:
                        _THAI_FONT_FAMILY = fams[0]
                        break
            except Exception:
                pass
    return _THAI_FONT_FAMILY


class DraggableCharacterItem(QGraphicsItem):
    """
    An interactive, movable, resizable, and rotatable Thai character or text item.
    Origin (0, 0) is at the center of the bounding box so rotation and scaling
    pivot naturally around the character center.
    """

    HANDLE_NONE = 0
    HANDLE_ROTATE = 1
    HANDLE_RESIZE_BR = 2
    HANDLE_RESIZE_TR = 3

    def __init__(
        self,
        text: str = "ก",
        font_size: int = 56,
        color: Optional[QColor] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.text = text
        self.font_size = font_size
        self.color = color if color is not None else QColor(0, 0, 0)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.opacity_val = 1.0
        self.char_type = "ThaiChar" if not text.isdigit() else "Digit"

        # Interaction flags
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        # Mouse tracking state
        self._active_handle = self.HANDLE_NONE
        self._drag_start_pos = QPointF()
        self._initial_scale_x = 1.0
        self._initial_scale_y = 1.0
        self._initial_rotation = 0.0

        # Calculate initial metrics
        self._recalc_bounds()

    def _recalc_bounds(self):
        font = QFont(get_thai_font_family(), self.font_size, QFont.Weight.Bold)
        fm = QFontMetrics(font)
        rect = fm.boundingRect(self.text)
        # Pad bounds slightly
        pad_x = max(6, int(rect.width() * 0.15))
        pad_y = max(6, int(rect.height() * 0.15))
        self._base_w = max(24, rect.width() + pad_x * 2)
        self._base_h = max(32, rect.height() + pad_y * 2)
        self.prepareGeometryChange()

    def boundingRect(self) -> QRectF:
        w = self._base_w * self.scale_x
        h = self._base_h * self.scale_y
        # Expand bounds to include selection handles and rotation handle
        margin = 35
        return QRectF(-w / 2 - margin, -h / 2 - margin, w + margin * 2, h + margin * 2)

    def text_rect(self) -> QRectF:
        """The actual bounds of the character itself."""
        w = self._base_w * self.scale_x
        h = self._base_h * self.scale_y
        return QRectF(-w / 2, -h / 2, w, h)

    def rotation_handle_center(self) -> QPointF:
        h = self._base_h * self.scale_y
        return QPointF(0, -h / 2 - 20)

    def resize_handle_rect(self) -> QRectF:
        w = self._base_w * self.scale_x
        h = self._base_h * self.scale_y
        return QRectF(w / 2 - 4, h / 2 - 4, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w = self._base_w * self.scale_x
        h = self._base_h * self.scale_y
        t_rect = QRectF(-w / 2, -h / 2, w, h)

        # Draw character text
        painter.save()
        painter.setOpacity(self.opacity_val)
        painter.setPen(self.color)
        font = QFont(get_thai_font_family(), int(self.font_size * min(self.scale_x, self.scale_y)), QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(t_rect, Qt.AlignmentFlag.AlignCenter, self.text)
        painter.restore()

        # Selection Gizmo (Handles & Border)
        if self.isSelected():
            painter.save()

            # Dashed bounding box
            pen_border = QPen(QColor("#00d2ff"), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen_border)
            painter.setBrush(QColor(0, 210, 255, 20))
            painter.drawRect(t_rect)

            # Rotation connector line and top handle
            rot_center = self.rotation_handle_center()
            painter.setPen(QPen(QColor("#00d2ff"), 1.5))
            painter.drawLine(QPointF(0, -h / 2), rot_center)

            # Rotation circle handle
            painter.setBrush(QColor("#fbbf24"))  # Gold
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawEllipse(rot_center, 6, 6)

            # Resize bottom-right handle
            painter.setBrush(QColor("#3b82f6"))  # Blue
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.drawRect(self.resize_handle_rect())

            # Center pivot marker
            painter.setPen(QPen(QColor("#ef4444"), 2))
            painter.drawPoint(0, 0)

            painter.restore()

    def mousePressEvent(self, event):
        pos = event.pos()
        rot_c = self.rotation_handle_center()
        dist_rot = math.hypot(pos.x() - rot_c.x(), pos.y() - rot_c.y())

        # Check rotation handle click
        if dist_rot <= 12:
            self._active_handle = self.HANDLE_ROTATE
            self._drag_start_pos = event.scenePos()
            self._initial_rotation = self.rotation()
            event.accept()
            return

        # Check resize handle click
        if self.resize_handle_rect().adjusted(-4, -4, 4, 4).contains(pos):
            self._active_handle = self.HANDLE_RESIZE_BR
            self._drag_start_pos = event.scenePos()
            self._initial_scale_x = self.scale_x
            self._initial_scale_y = self.scale_y
            event.accept()
            return

        # Otherwise standard move
        self._active_handle = self.HANDLE_NONE
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._active_handle == self.HANDLE_ROTATE:
            # Compute angle from item center to mouse in scene coordinates
            center_scene = self.mapToScene(QPointF(0, 0))
            cur_scene = event.scenePos()
            dx = cur_scene.x() - center_scene.x()
            dy = cur_scene.y() - center_scene.y()
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad) + 90.0  # +90 because handle is at top
            self.setRotation(angle_deg)
            self.update()
            if self.scene():
                self.scene().item_transformed.emit(self)
            event.accept()
            return

        elif self._active_handle == self.HANDLE_RESIZE_BR:
            # Resize based on distance from center
            pos = event.pos()
            orig_w = self._base_w / 2.0
            orig_h = self._base_h / 2.0
            if orig_w > 0 and orig_h > 0:
                new_sx = max(0.2, min(5.0, abs(pos.x()) / orig_w))
                new_sy = max(0.2, min(5.0, abs(pos.y()) / orig_h))
                # Maintain aspect ratio if shift is held
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    avg = (new_sx + new_sy) / 2.0
                    new_sx = avg
                    new_sy = avg
                self.scale_x = round(new_sx, 2)
                self.scale_y = round(new_sy, 2)
                self.prepareGeometryChange()
                self.update()
                if self.scene():
                    self.scene().item_transformed.emit(self)
            event.accept()
            return

        super().mouseMoveEvent(event)
        if self.scene():
            self.scene().item_transformed.emit(self)

    def mouseReleaseEvent(self, event):
        self._active_handle = self.HANDLE_NONE
        super().mouseReleaseEvent(event)
        if self.scene():
            self.scene().item_transformed.emit(self)

    def set_text(self, text: str):
        self.text = text
        self.char_type = "ThaiChar" if not text.isdigit() else "Digit"
        self._recalc_bounds()
        self.update()

    def set_font_size(self, size: int):
        self.font_size = max(10, min(200, size))
        self._recalc_bounds()
        self.update()

    def set_scale_factors(self, sx: float, sy: float):
        self.scale_x = max(0.1, min(10.0, sx))
        self.scale_y = max(0.1, min(10.0, sy))
        self.prepareGeometryChange()
        self.update()

    def set_char_color(self, color: QColor):
        self.color = color
        self.update()

    def set_opacity_val(self, op: float):
        self.opacity_val = max(0.05, min(1.0, op))
        self.update()

    def get_scene_bbox_rect(self) -> QRectF:
        """Returns the bounding rectangle of the character in scene coordinates."""
        w = self._base_w * self.scale_x
        h = self._base_h * self.scale_y
        local_rect = QRectF(-w / 2, -h / 2, w, h)
        return self.mapToScene(local_rect).boundingRect()

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "x": self.x(),
            "y": self.y(),
            "rotation": self.rotation(),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "font_size": self.font_size,
            "color": self.color.name(),
            "opacity": self.opacity_val,
            "char_type": self.char_type,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DraggableCharacterItem":
        item = cls(
            text=d.get("text", "ก"),
            font_size=d.get("font_size", 56),
            color=QColor(d.get("color", "#000000")),
        )
        item.setPos(d.get("x", 0), d.get("y", 0))
        item.setRotation(d.get("rotation", 0.0))
        item.scale_x = d.get("scale_x", 1.0)
        item.scale_y = d.get("scale_y", 1.0)
        item.opacity_val = d.get("opacity", 1.0)
        item.char_type = d.get("char_type", "ThaiChar")
        item.prepareGeometryChange()
        return item


class PlateGraphicsScene(QGraphicsScene):
    item_transformed = pyqtSignal(object)
    selection_changed_custom = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_item = QGraphicsPixmapItem()
        self.bg_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.addItem(self.bg_item)
        self.selectionChanged.connect(self._on_internal_selection_changed)

    def _on_internal_selection_changed(self):
        selected = self.selectedItems()
        char_items = [it for it in selected if isinstance(it, DraggableCharacterItem)]
        self.selection_changed_custom.emit(char_items[0] if char_items else None)


class PlateCanvasView(QGraphicsView):
    """Zoomable, pannable view for interactive plate editing."""

    def __init__(self, scene: PlateGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#18181b")))

    def wheelEvent(self, event):
        """Ctrl + Wheel for Zoom."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            factor = 1.15 if zoom_in else 1.0 / 1.15
            self.scale(factor, factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        """Arrow keys to nudge selected character."""
        selected = [it for it in self.scene().selectedItems() if isinstance(it, DraggableCharacterItem)]
        if selected:
            step = 10 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1
            if event.key() == Qt.Key.Key_Left:
                for it in selected:
                    it.setPos(it.x() - step, it.y())
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Right:
                for it in selected:
                    it.setPos(it.x() + step, it.y())
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Up:
                for it in selected:
                    it.setPos(it.x(), it.y() - step)
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Down:
                for it in selected:
                    it.setPos(it.x(), it.y() + step)
                event.accept()
                return
            elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                for it in selected:
                    self.scene().removeItem(it)
                event.accept()
                return
        super().keyPressEvent(event)
