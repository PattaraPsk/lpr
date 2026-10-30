#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Plate Editor Tab Widget
====================================
Combines the interactive canvas view with toolbars, character inspector,
quick Thai consonant/number palettes, and export actions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from .interactive_canvas import (
        DraggableCharacterItem,
        PlateCanvasView,
        PlateGraphicsScene,
        THAI_FONT_FAMILY,
    )
    from .plate_renderer import (
        PLATE_CONFIGS,
        PROVINCES,
        VALID_CONSONANTS,
        _draw_rounded_rect,
    )
except ImportError:
    from interactive_canvas import (
        DraggableCharacterItem,
        PlateCanvasView,
        PlateGraphicsScene,
        THAI_FONT_FAMILY,
    )
    from plate_renderer import (
        PLATE_CONFIGS,
        PROVINCES,
        VALID_CONSONANTS,
        _draw_rounded_rect,
    )


def create_blank_plate_pixmap(plate_type: str = "WhiteBlackCar2Line") -> QPixmap:
    """Generates a blank reference plate template with border and divider."""
    cfg = PLATE_CONFIGS.get(plate_type, PLATE_CONFIGS["WhiteBlackCar2Line"])
    w, h = cfg["width"], cfg["height"]
    img = Image.new("RGB", (w, h), cfg["bg_color"])
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    _draw_rounded_rect(draw, (0, 0, w - 1, h - 1), fill=cfg["bg_color"], outline=cfg["border_color"], width=3, radius=6)

    # Optional divider for two-line plates
    if cfg["layout"] == "two_line":
        div_y = int(h * 0.40)
        draw.line([(15, div_y), (w - 15, div_y)], fill=cfg["border_color"], width=1)
    elif cfg["layout"] == "motorcycle":
        div_y = int(h * 0.38)
        draw.line([(12, div_y), (w - 12, div_y)], fill=cfg["border_color"], width=1)

    # Convert PIL to QPixmap
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class InteractiveEditorTab(QWidget):
    # Signals
    composite_rendered = pyqtSignal(object, list, dict)  # (PIL.Image, bboxes, meta)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = PlateGraphicsScene(self)
        self.view = PlateCanvasView(self.scene, self)
        self.reference_pixmap: Optional[QPixmap] = None
        self.selected_item: Optional[DraggableCharacterItem] = None

        self._init_ui()
        self.scene.item_transformed.connect(self._on_item_transformed)
        self.scene.selection_changed_custom.connect(self._on_item_selected)

        # Default template
        self.load_blank_template("WhiteBlackCar2Line")
        # Populate default characters
        self.populate_plate_text("1กข", "1234", "กรุงเทพมหานคร")

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ----------------- Left Panel: Controls & Inspector -----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # 1. Reference Plate Background Group
        grp_ref = QGroupBox("1. Reference Plate Image")
        ref_grid = QGridLayout(grp_ref)

        btn_load_ref = QPushButton("📁 Load Reference Image...")
        btn_load_ref.clicked.connect(self._on_load_reference_image)
        ref_grid.addWidget(btn_load_ref, 0, 0, 1, 2)

        ref_grid.addWidget(QLabel("Or Template:"), 1, 0)
        self.combo_templates = QComboBox()
        self.combo_templates.addItem("Blank White Car 2-Line (340x170)", "WhiteBlackCar2Line")
        self.combo_templates.addItem("Blank White Car 1-Line (520x110)", "WhiteBlackCar")
        self.combo_templates.addItem("Blank Motorcycle (220x170)", "WhiteBlackMotor")
        self.combo_templates.addItem("Blank Red Dealer (340x170)", "RedBlack")
        self.combo_templates.addItem("Blank Yellow Taxi (340x170)", "YellowBlack")
        self.combo_templates.currentIndexChanged.connect(
            lambda: self.load_blank_template(self.combo_templates.currentData())
        )
        ref_grid.addWidget(self.combo_templates, 1, 1)

        self.lbl_canvas_info = QLabel("Size: -")
        self.lbl_canvas_info.setStyleSheet("color: #60a5fa; font-size: 11px;")
        ref_grid.addWidget(self.lbl_canvas_info, 2, 0, 1, 2)

        left_layout.addWidget(grp_ref)

        # 2. Add / Populate Characters Group
        grp_add = QGroupBox("2. Character Placement")
        add_layout = QVBoxLayout(grp_add)

        pop_grid = QGridLayout()
        pop_grid.addWidget(QLabel("D3 Consonants:"), 0, 0)
        self.txt_pop_d3 = QLineEdit("1กข")
        pop_grid.addWidget(self.txt_pop_d3, 0, 1)

        pop_grid.addWidget(QLabel("D4 Digits:"), 1, 0)
        self.txt_pop_d4 = QLineEdit("1234")
        pop_grid.addWidget(self.txt_pop_d4, 1, 1)

        pop_grid.addWidget(QLabel("Province:"), 2, 0)
        self.combo_pop_prov = QComboBox()
        for p in PROVINCES:
            self.combo_pop_prov.addItem(p)
        pop_grid.addWidget(self.combo_pop_prov, 2, 1)

        btn_auto_place = QPushButton("⚡ Auto-Place All on Plate")
        btn_auto_place.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold;")
        btn_auto_place.clicked.connect(self._on_auto_place_clicked)
        pop_grid.addWidget(btn_auto_place, 3, 0, 1, 2)
        add_layout.addLayout(pop_grid)

        # Quick Add Single Character
        h_single = QHBoxLayout()
        self.txt_single_char = QLineEdit("ก")
        self.txt_single_char.setMaximumWidth(50)
        btn_add_char = QPushButton("➕ Add Character")
        btn_add_char.clicked.connect(self._on_add_single_char)
        h_single.addWidget(self.txt_single_char)
        h_single.addWidget(btn_add_char)
        add_layout.addLayout(h_single)

        left_layout.addWidget(grp_add)

        # 3. Selected Character Inspector Group
        self.grp_inspector = QGroupBox("3. Selected Character Inspector")
        ins_grid = QGridLayout(self.grp_inspector)

        ins_grid.addWidget(QLabel("Character / Text:"), 0, 0)
        self.txt_ins_text = QLineEdit()
        self.txt_ins_text.textChanged.connect(self._on_ins_text_changed)
        ins_grid.addWidget(self.txt_ins_text, 0, 1)

        # Position X, Y
        ins_grid.addWidget(QLabel("Position (X, Y):"), 1, 0)
        h_pos = QHBoxLayout()
        self.spn_pos_x = QSpinBox()
        self.spn_pos_x.setRange(-2000, 4000)
        self.spn_pos_x.valueChanged.connect(self._on_ins_pos_changed)
        self.spn_pos_y = QSpinBox()
        self.spn_pos_y.setRange(-2000, 4000)
        self.spn_pos_y.valueChanged.connect(self._on_ins_pos_changed)
        h_pos.addWidget(self.spn_pos_x)
        h_pos.addWidget(self.spn_pos_y)
        ins_grid.addLayout(h_pos, 1, 1)

        # Rotation
        ins_grid.addWidget(QLabel("Rotation (°):"), 2, 0)
        h_rot = QHBoxLayout()
        self.slider_rot = QSlider(Qt.Orientation.Horizontal)
        self.slider_rot.setRange(-180, 180)
        self.slider_rot.setValue(0)
        self.spn_rot = QSpinBox()
        self.spn_rot.setRange(-180, 180)
        self.slider_rot.valueChanged.connect(self.spn_rot.setValue)
        self.spn_rot.valueChanged.connect(self.slider_rot.setValue)
        self.spn_rot.valueChanged.connect(self._on_ins_rot_changed)
        h_rot.addWidget(self.slider_rot)
        h_rot.addWidget(self.spn_rot)
        ins_grid.addLayout(h_rot, 2, 1)

        # Scale / Font size
        ins_grid.addWidget(QLabel("Font Size (pt):"), 3, 0)
        self.spn_font_size = QSpinBox()
        self.spn_font_size.setRange(12, 180)
        self.spn_font_size.setValue(56)
        self.spn_font_size.valueChanged.connect(self._on_ins_font_size_changed)
        ins_grid.addWidget(self.spn_font_size, 3, 1)

        ins_grid.addWidget(QLabel("Scale (W, H):"), 4, 0)
        h_scale = QHBoxLayout()
        self.spn_scale_x = QDoubleSpinBox()
        self.spn_scale_x.setRange(0.2, 5.0)
        self.spn_scale_x.setSingleStep(0.05)
        self.spn_scale_x.setValue(1.0)
        self.spn_scale_x.valueChanged.connect(self._on_ins_scale_changed)
        self.spn_scale_y = QDoubleSpinBox()
        self.spn_scale_y.setRange(0.2, 5.0)
        self.spn_scale_y.setSingleStep(0.05)
        self.spn_scale_y.setValue(1.0)
        self.spn_scale_y.valueChanged.connect(self._on_ins_scale_changed)
        h_scale.addWidget(self.spn_scale_x)
        h_scale.addWidget(self.spn_scale_y)
        ins_grid.addLayout(h_scale, 4, 1)

        # Color & Opacity
        ins_grid.addWidget(QLabel("Color & Alpha:"), 5, 0)
        h_col = QHBoxLayout()
        self.btn_color_pick = QPushButton("Color")
        self.btn_color_pick.clicked.connect(self._on_pick_color)
        self.spn_opacity = QSpinBox()
        self.spn_opacity.setRange(10, 100)
        self.spn_opacity.setValue(100)
        self.spn_opacity.setSuffix("%")
        self.spn_opacity.valueChanged.connect(self._on_ins_opacity_changed)
        h_col.addWidget(self.btn_color_pick)
        h_col.addWidget(self.spn_opacity)
        ins_grid.addLayout(h_col, 5, 1)

        # Actions on character
        h_act = QHBoxLayout()
        btn_dup = QPushButton("📋 Duplicate")
        btn_dup.clicked.connect(self._on_duplicate_selected)
        btn_del = QPushButton("🗑️ Delete")
        btn_del.setStyleSheet("background-color: #ef4444; color: white;")
        btn_del.clicked.connect(self._on_delete_selected)
        h_act.addWidget(btn_dup)
        h_act.addWidget(btn_del)
        ins_grid.addLayout(h_act, 6, 0, 1, 2)

        left_layout.addWidget(self.grp_inspector)

        # 4. Canvas Export & Augment Integration
        grp_exp = QGroupBox("4. Export & Pipeline Actions")
        exp_layout = QVBoxLayout(grp_exp)

        btn_render_and_aug = QPushButton("🚀 Send to Augmentation Pipeline")
        btn_render_and_aug.setStyleSheet("background-color: #10b981; color: white; font-weight: bold; padding: 8px;")
        btn_render_and_aug.clicked.connect(self._on_send_to_augment_pipeline)
        exp_layout.addWidget(btn_render_and_aug)

        h_exp_row = QHBoxLayout()
        btn_export_yolo = QPushButton("💾 Export Image + YOLO")
        btn_export_yolo.clicked.connect(self._on_export_image_and_yolo)
        btn_save_proj = QPushButton("📁 Save Project (.json)")
        btn_save_proj.clicked.connect(self._on_save_project)
        btn_load_proj = QPushButton("📂 Load Project")
        btn_load_proj.clicked.connect(self._on_load_project)
        h_exp_row.addWidget(btn_export_yolo)
        h_exp_row.addWidget(btn_save_proj)
        h_exp_row.addWidget(btn_load_proj)
        exp_layout.addLayout(h_exp_row)

        left_layout.addWidget(grp_exp)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        splitter.setStretchFactor(0, 3)

        # ----------------- Right Panel: Graphics Canvas -----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar above canvas
        canvas_bar = QHBoxLayout()
        lbl_hint = QLabel("💡 Drag to Move | Top circle to Rotate | Bottom-right corner to Resize | Shift for proportional")
        lbl_hint.setStyleSheet("color: #a1a1aa; font-size: 11px;")
        canvas_bar.addWidget(lbl_hint)
        canvas_bar.addStretch()

        btn_zoom_fit = QPushButton("Fit to Window")
        btn_zoom_fit.clicked.connect(self._on_zoom_fit)
        btn_zoom_100 = QPushButton("100% Zoom")
        btn_zoom_100.clicked.connect(lambda: self.view.resetTransform())
        canvas_bar.addWidget(btn_zoom_fit)
        canvas_bar.addWidget(btn_zoom_100)
        right_layout.addLayout(canvas_bar)

        right_layout.addWidget(self.view)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 7)

    # -----------------------------------------------------------------------
    # Canvas & Template Operations
    # -----------------------------------------------------------------------
    def load_blank_template(self, plate_type: str):
        pixmap = create_blank_plate_pixmap(plate_type)
        self.set_reference_pixmap(pixmap)

    def set_reference_pixmap(self, pixmap: QPixmap):
        self.reference_pixmap = pixmap
        self.scene.bg_item.setPixmap(pixmap)
        rect = QRectF(pixmap.rect())
        self.scene.setSceneRect(rect)
        self.lbl_canvas_info.setText(f"Resolution: {pixmap.width()} x {pixmap.height()} px")
        self._on_zoom_fit()

    def _on_load_reference_image(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Plate Image", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        if f:
            pix = QPixmap(f)
            if not pix.isNull():
                self.set_reference_pixmap(pix)
            else:
                QMessageBox.warning(self, "Failed", "Could not load selected image.")

    def _on_zoom_fit(self):
        if self.reference_pixmap and not self.reference_pixmap.isNull():
            self.view.fitInView(self.scene.bg_item, Qt.AspectRatioMode.KeepAspectRatio)

    # -----------------------------------------------------------------------
    # Character Placement
    # -----------------------------------------------------------------------
    def clear_characters(self):
        for it in list(self.scene.items()):
            if isinstance(it, DraggableCharacterItem):
                self.scene.removeItem(it)

    def populate_plate_text(self, d3: str, d4: str, province: str):
        self.clear_characters()
        if not self.reference_pixmap:
            return

        w = self.reference_pixmap.width()
        h = self.reference_pixmap.height()

        # Two-line or single-line layout detection
        is_single_line = (w / max(h, 1)) > 3.0

        if is_single_line:
            # Single-line: D3 + D4 on one line, province below
            combined = f"{d3} {d4}"
            spacing = w * 0.75 / max(len(combined), 1)
            start_x = w * 0.15
            y_main = h * 0.38
            font_size = int(h * 0.45)

            cur_x = start_x
            for ch in combined:
                if ch != " ":
                    item = DraggableCharacterItem(ch, font_size=font_size)
                    item.setPos(cur_x, y_main)
                    self.scene.addItem(item)
                cur_x += spacing

            # Province text
            prov_item = DraggableCharacterItem(province, font_size=int(h * 0.20), color=QColor(80, 80, 80))
            prov_item.setPos(w * 0.5, h * 0.82)
            self.scene.addItem(prov_item)

        else:
            # Two-line: D3 on line 1, D4 on line 2, province at bottom
            # Line 1: D3
            font_size_d3 = int(h * 0.28)
            y_d3 = h * 0.22
            count_d3 = len(d3)
            span_d3 = w * 0.5
            spacing_d3 = span_d3 / max(count_d3, 1)
            start_x_d3 = (w - (count_d3 - 1) * spacing_d3) / 2.0

            for i, ch in enumerate(d3):
                item = DraggableCharacterItem(ch, font_size=font_size_d3)
                item.setPos(start_x_d3 + i * spacing_d3, y_d3)
                self.scene.addItem(item)

            # Line 2: D4
            font_size_d4 = int(h * 0.28)
            y_d4 = h * 0.58
            count_d4 = len(d4)
            span_d4 = w * 0.65
            spacing_d4 = span_d4 / max(count_d4, 1)
            start_x_d4 = (w - (count_d4 - 1) * spacing_d4) / 2.0

            for i, ch in enumerate(d4):
                item = DraggableCharacterItem(ch, font_size=font_size_d4)
                item.setPos(start_x_d4 + i * spacing_d4, y_d4)
                self.scene.addItem(item)

            # Province
            font_size_p = int(h * 0.12)
            y_p = h * 0.86
            prov_item = DraggableCharacterItem(province, font_size=font_size_p, color=QColor(80, 80, 80))
            prov_item.setPos(w * 0.5, y_p)
            self.scene.addItem(prov_item)

    def _on_auto_place_clicked(self):
        d3 = self.txt_pop_d3.text().strip() or "กข"
        d4 = self.txt_pop_d4.text().strip() or "1234"
        prov = self.combo_pop_prov.currentText()
        self.populate_plate_text(d3, d4, prov)

    def _on_add_single_char(self):
        text = self.txt_single_char.text().strip()
        if not text:
            return
        cx = self.scene.sceneRect().center().x() if self.scene.sceneRect().width() > 0 else 100
        cy = self.scene.sceneRect().center().y() if self.scene.sceneRect().height() > 0 else 100
        item = DraggableCharacterItem(text, font_size=56)
        item.setPos(cx, cy)
        self.scene.addItem(item)
        self.scene.clearSelection()
        item.setSelected(True)

    # -----------------------------------------------------------------------
    # Inspector Event Handling
    # -----------------------------------------------------------------------
    def _on_item_selected(self, item: Optional[DraggableCharacterItem]):
        self.selected_item = item
        if item is None:
            self.grp_inspector.setTitle("3. Character Inspector (No selection)")
            return

        self.grp_inspector.setTitle(f"3. Character Inspector — Selected: '{item.text}'")
        self.txt_ins_text.blockSignals(True)
        self.spn_pos_x.blockSignals(True)
        self.spn_pos_y.blockSignals(True)
        self.spn_rot.blockSignals(True)
        self.slider_rot.blockSignals(True)
        self.spn_font_size.blockSignals(True)
        self.spn_scale_x.blockSignals(True)
        self.spn_scale_y.blockSignals(True)
        self.spn_opacity.blockSignals(True)

        self.txt_ins_text.setText(item.text)
        self.spn_pos_x.setValue(int(item.x()))
        self.spn_pos_y.setValue(int(item.y()))
        rot = int(round(item.rotation()))
        self.spn_rot.setValue(rot)
        self.slider_rot.setValue(rot)
        self.spn_font_size.setValue(item.font_size)
        self.spn_scale_x.setValue(item.scale_x)
        self.spn_scale_y.setValue(item.scale_y)
        self.spn_opacity.setValue(int(item.opacity_val * 100))
        self.btn_color_pick.setStyleSheet(f"background-color: {item.color.name()}; color: white;")

        self.txt_ins_text.blockSignals(False)
        self.spn_pos_x.blockSignals(False)
        self.spn_pos_y.blockSignals(False)
        self.spn_rot.blockSignals(False)
        self.slider_rot.blockSignals(False)
        self.spn_font_size.blockSignals(False)
        self.spn_scale_x.blockSignals(False)
        self.spn_scale_y.blockSignals(False)
        self.spn_opacity.blockSignals(False)

    def _on_item_transformed(self, item: DraggableCharacterItem):
        if self.selected_item == item:
            self._on_item_selected(item)

    def _on_ins_text_changed(self, text: str):
        if self.selected_item and text:
            self.selected_item.set_text(text)

    def _on_ins_pos_changed(self):
        if self.selected_item:
            self.selected_item.setPos(self.spn_pos_x.value(), self.spn_pos_y.value())

    def _on_ins_rot_changed(self, val: int):
        if self.selected_item:
            self.selected_item.setRotation(val)

    def _on_ins_font_size_changed(self, val: int):
        if self.selected_item:
            self.selected_item.set_font_size(val)

    def _on_ins_scale_changed(self):
        if self.selected_item:
            self.selected_item.set_scale_factors(self.spn_scale_x.value(), self.spn_scale_y.value())

    def _on_ins_opacity_changed(self, val: int):
        if self.selected_item:
            self.selected_item.set_opacity_val(val / 100.0)

    def _on_pick_color(self):
        if self.selected_item:
            col = QColorDialog.getColor(self.selected_item.color, self, "Pick Character Color")
            if col.isValid():
                self.selected_item.set_char_color(col)
                self.btn_color_pick.setStyleSheet(f"background-color: {col.name()}; color: white;")

    def _on_duplicate_selected(self):
        if self.selected_item:
            d = self.selected_item.to_dict()
            d["x"] += 20
            d["y"] += 10
            new_item = DraggableCharacterItem.from_dict(d)
            self.scene.addItem(new_item)
            self.scene.clearSelection()
            new_item.setSelected(True)

    def _on_delete_selected(self):
        if self.selected_item:
            self.scene.removeItem(self.selected_item)
            self.selected_item = None
            self._on_item_selected(None)

    # -----------------------------------------------------------------------
    # Rendering & Export
    # -----------------------------------------------------------------------
    def render_composite(self) -> Tuple[Image.Image, List[Dict], Dict]:
        """
        Renders the reference plate + all arranged character items into a high-res PIL Image.
        Computes accurate bounding boxes in image pixel coordinates.
        """
        if not self.reference_pixmap:
            raise ValueError("No reference plate loaded.")

        w = self.reference_pixmap.width()
        h = self.reference_pixmap.height()

        # Render QGraphicsScene to QImage
        qimg = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        qimg.fill(Qt.GlobalColor.transparent)

        painter = QPainter(qimg)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Clear selection temporarily so handles aren't painted
        self.scene.clearSelection()
        self.scene.render(painter, QRectF(0, 0, w, h), QRectF(0, 0, w, h))
        painter.end()

        # Convert QImage to PIL Image
        ptr = qimg.bits()
        ptr.setsize(h * w * 4)
        import numpy as np

        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
        pil_img = Image.fromarray(arr).convert("RGB")

        # Collect character bounding boxes
        char_bboxes = []
        for it in self.scene.items():
            if isinstance(it, DraggableCharacterItem):
                rect = it.get_scene_bbox_rect()
                x1 = max(0, int(rect.left()))
                y1 = max(0, int(rect.top()))
                x2 = min(w, int(rect.right()))
                y2 = min(h, int(rect.bottom()))
                if x2 > x1 and y2 > y1:
                    char_bboxes.append({
                        "char": it.text,
                        "bbox": (x1, y1, x2, y2),
                        "type": it.char_type,
                    })

        metadata = {
            "width": w,
            "height": h,
            "d3": self.txt_pop_d3.text().strip(),
            "d4": self.txt_pop_d4.text().strip(),
            "province": self.combo_pop_prov.currentText(),
            "full_text": f"{self.txt_pop_d3.text().strip()} {self.txt_pop_d4.text().strip()} {self.combo_pop_prov.currentText()}",
        }

        return pil_img, char_bboxes, metadata

    def _on_send_to_augment_pipeline(self):
        try:
            img, bboxes, meta = self.render_composite()
            self.composite_rendered.emit(img, bboxes, meta)
            QMessageBox.information(
                self,
                "Sent to Pipeline",
                "Composite plate successfully rendered and loaded into the Augmentation Pipeline tab!",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_export_image_and_yolo(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Save Composite Plate Image", "custom_plate.jpg", "JPEG Image (*.jpg);;PNG Image (*.png)"
        )
        if not f:
            return

        try:
            img, bboxes, meta = self.render_composite()
            dest_img = Path(f)
            img.save(dest_img, quality=95)

            # Save YOLO txt
            dest_lbl = dest_img.with_suffix(".txt")
            w, h = img.size
            lines = [f"0 0.500000 0.500000 1.000000 1.000000"]  # Plate
            for b in bboxes:
                x1, y1, x2, y2 = b["bbox"]
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                cls_id = 2 if b["type"] == "Digit" else 1
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            with open(dest_lbl, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines) + "\n")

            QMessageBox.information(
                self,
                "Exported Successfully",
                f"Saved plate image:\n{dest_img}\n\nSaved YOLO annotations:\n{dest_lbl}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def _on_save_project(self):
        f, _ = QFileDialog.getSaveFileName(
            self, "Save Plate Layout Project", "plate_project.json", "JSON (*.json)"
        )
        if not f:
            return

        items_data = []
        for it in self.scene.items():
            if isinstance(it, DraggableCharacterItem):
                items_data.append(it.to_dict())

        project_data = {
            "template": self.combo_templates.currentData(),
            "width": self.reference_pixmap.width() if self.reference_pixmap else 340,
            "height": self.reference_pixmap.height() if self.reference_pixmap else 170,
            "d3": self.txt_pop_d3.text(),
            "d4": self.txt_pop_d4.text(),
            "province": self.combo_pop_prov.currentText(),
            "characters": items_data,
        }

        with open(f, "w", encoding="utf-8") as jf:
            json.dump(project_data, jf, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Project Saved", f"Project saved to:\n{f}")

    def _on_load_project(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Open Plate Layout Project", "", "JSON (*.json)"
        )
        if not f:
            return

        try:
            with open(f, "r", encoding="utf-8") as jf:
                project_data = json.load(jf)

            self.clear_characters()
            for cd in project_data.get("characters", []):
                item = DraggableCharacterItem.from_dict(cd)
                self.scene.addItem(item)

            self.txt_pop_d3.setText(project_data.get("d3", ""))
            self.txt_pop_d4.setText(project_data.get("d4", ""))
            prov = project_data.get("province", "")
            idx = self.combo_pop_prov.findText(prov)
            if idx >= 0:
                self.combo_pop_prov.setCurrentIndex(idx)

            QMessageBox.information(self, "Loaded", f"Loaded {len(project_data.get('characters', []))} characters.")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))
