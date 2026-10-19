"""
Model Surgery & Pruning Tab.
Allows selecting base YOLO models, inspecting layers, freezing weights, slicing networks,
removing specific nodes/layers, and applying structured/unstructured pruning.
"""

from pathlib import Path
from typing import List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QLineEdit, QFileDialog, QComboBox, QSlider,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QSpinBox, QFormLayout, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.model_surgery import ModelSurgery, LayerInfo, SurgeryStats

class ModelSurgeryTab(QWidget):
    """Tab 3: YOLO Model Architecture Surgery, Freezing, Slicing & Pruning."""
    
    model_configured = pyqtSignal(str, list)  # emits (model_path, list_of_frozen_indices)
    
    def __init__(self):
        super().__init__()
        self.surgery = ModelSurgery("yolov8s.pt")
        self._table_updating = False
        self._init_ui()
        self._populate_table()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # 1. Model Selector Header Card
        top_group = QGroupBox("🧠 Base YOLO Model Selection")
        top_layout = QHBoxLayout(top_group)
        top_layout.setContentsMargins(12, 14, 12, 12)
        top_layout.setSpacing(8)
        
        self.combo_base_model = QComboBox()
        self.combo_base_model.addItems(ModelSurgery.YOLO_PRESETS)
        self.combo_base_model.setEditable(True)
        top_layout.addWidget(QLabel("Base Model:"), 0)
        top_layout.addWidget(self.combo_base_model, 1)
        
        self.btn_browse_model = QPushButton("📁 Custom .pt...")
        self.btn_browse_model.setObjectName("btnSecondary")
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        top_layout.addWidget(self.btn_browse_model)
        
        self.btn_load_model = QPushButton("🔄 Load Architecture")
        self.btn_load_model.clicked.connect(self._on_load_model)
        top_layout.addWidget(self.btn_load_model)
        
        layout.addWidget(top_group)
        
        # 2. Main Central Splitter (Left: Layer Table, Right: Surgery Operations)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        
        # Left: Layer Architecture Table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        
        lbl_tbl = QLabel("📋 Model Layers & Network Hierarchy:")
        lbl_tbl.setStyleSheet("font-weight: bold; color: #38bdf8;")
        left_layout.addWidget(lbl_tbl)
        
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Select", "Idx", "Layer Name", "Type", "Total Params",
            "Trainable", "Status", "Sparsity %", "Details"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.table, 1)
        
        # Quick table selection toolbar
        tb_layout = QHBoxLayout()
        self.btn_tbl_select_all = QPushButton("☑️ Check All")
        self.btn_tbl_select_all.setObjectName("btnSecondary")
        self.btn_tbl_select_all.clicked.connect(lambda: self._bulk_table_check(True))
        tb_layout.addWidget(self.btn_tbl_select_all)
        
        self.btn_tbl_deselect_all = QPushButton("⬜ Uncheck All")
        self.btn_tbl_deselect_all.setObjectName("btnSecondary")
        self.btn_tbl_deselect_all.clicked.connect(lambda: self._bulk_table_check(False))
        tb_layout.addWidget(self.btn_tbl_deselect_all)
        
        tb_layout.addStretch()
        left_layout.addLayout(tb_layout)
        
        splitter.addWidget(left_widget)
        
        # Right: Surgery Operations Panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Group 1: Layer Freezing
        grp_freeze = QGroupBox("❄️ Layer Freezing (Transfer Learning)")
        v_frz = QVBoxLayout(grp_freeze)
        v_frz.setContentsMargins(12, 14, 12, 12)
        v_frz.setSpacing(8)
        
        h_frz_btn = QHBoxLayout()
        self.btn_freeze_selected = QPushButton("❄️ Freeze Selected")
        self.btn_freeze_selected.clicked.connect(self._freeze_selected)
        h_frz_btn.addWidget(self.btn_freeze_selected)
        
        self.btn_unfreeze_selected = QPushButton("🔥 Unfreeze Selected")
        self.btn_unfreeze_selected.setObjectName("btnSecondary")
        self.btn_unfreeze_selected.clicked.connect(self._unfreeze_selected)
        h_frz_btn.addWidget(self.btn_unfreeze_selected)
        v_frz.addLayout(h_frz_btn)
        
        # Presets
        h_preset = QHBoxLayout()
        h_preset.addWidget(QLabel("Quick Preset:"))
        self.combo_freeze_preset = QComboBox()
        self.combo_freeze_preset.addItems([
            "Freeze Backbone (Layers 0-9)",
            "Freeze All Except Detect Head",
            "Unfreeze All (Train Full Network)"
        ])
        h_preset.addWidget(self.combo_freeze_preset, 1)
        
        self.btn_apply_preset = QPushButton("Apply")
        self.btn_apply_preset.setObjectName("btnSecondary")
        self.btn_apply_preset.clicked.connect(self._apply_freeze_preset)
        h_preset.addWidget(self.btn_apply_preset)
        v_frz.addLayout(h_preset)
        
        right_layout.addWidget(grp_freeze)
        
        # Group 2: Slicing & Node Removal
        grp_slice = QGroupBox("✂️ Slicing & Node Removal")
        v_slc = QVBoxLayout(grp_slice)
        v_slc.setContentsMargins(12, 14, 12, 12)
        v_slc.setSpacing(8)
        
        h_slc = QHBoxLayout()
        h_slc.addWidget(QLabel("Slice Up To Layer Index:"))
        self.spin_slice_idx = QSpinBox()
        self.spin_slice_idx.setRange(0, 50)
        self.spin_slice_idx.setValue(15)
        h_slc.addWidget(self.spin_slice_idx)
        
        self.btn_slice = QPushButton("✂️ Slice Model")
        self.btn_slice.setObjectName("btnSecondary")
        self.btn_slice.clicked.connect(self._on_slice_model)
        h_slc.addWidget(self.btn_slice)
        v_slc.addLayout(h_slc)
        
        self.btn_remove_node = QPushButton("🗑️ Remove Selected Node/Layer")
        self.btn_remove_node.setObjectName("btnDanger")
        self.btn_remove_node.clicked.connect(self._on_remove_selected_node)
        v_slc.addWidget(self.btn_remove_node)
        
        right_layout.addWidget(grp_slice)
        
        # Group 3: Pruning Controls
        grp_prune = QGroupBox("🎯 Weight & Channel Pruning")
        v_prn = QVBoxLayout(grp_prune)
        v_prn.setContentsMargins(12, 14, 12, 12)
        v_prn.setSpacing(8)
        
        h_prn_ratio = QHBoxLayout()
        self.lbl_prn_ratio = QLabel("Prune Amount: 30%")
        self.slider_prune = QSlider(Qt.Orientation.Horizontal)
        self.slider_prune.setRange(5, 80)
        self.slider_prune.setValue(30)
        self.slider_prune.valueChanged.connect(lambda v: self.lbl_prn_ratio.setText(f"Prune Amount: {v}%"))
        h_prn_ratio.addWidget(self.lbl_prn_ratio)
        h_prn_ratio.addWidget(self.slider_prune)
        v_prn.addLayout(h_prn_ratio)
        
        h_prn_method = QHBoxLayout()
        h_prn_method.addWidget(QLabel("Method:"))
        self.combo_prune_method = QComboBox()
        self.combo_prune_method.addItems([
            "L1 Unstructured Pruning (Fine-Grained)",
            "Structured Channel Pruning (Speedup)"
        ])
        h_prn_method.addWidget(self.combo_prune_method, 1)
        v_prn.addLayout(h_prn_method)
        
        self.btn_apply_prune = QPushButton("🎯 Prune Selected Layers")
        self.btn_apply_prune.setObjectName("btnSuccess")
        self.btn_apply_prune.clicked.connect(self._apply_pruning)
        v_prn.addWidget(self.btn_apply_prune)
        
        right_layout.addWidget(grp_prune)
        
        # Group 4: Export Checkpoint
        grp_export = QGroupBox("💾 Save & Export Modified Architecture")
        v_exp = QVBoxLayout(grp_export)
        v_exp.setContentsMargins(12, 14, 12, 12)
        v_exp.setSpacing(8)
        
        self.btn_export = QPushButton("💾 Export Modified Model (.pt)")
        self.btn_export.setObjectName("btnSuccess")
        self.btn_export.setMinimumHeight(36)
        self.btn_export.clicked.connect(self._on_export_model)
        v_exp.addWidget(self.btn_export)
        
        right_layout.addWidget(grp_export)
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)
        
        # 3. Bottom Summary Cards
        self.card_summary = QLabel()
        self.card_summary.setStyleSheet("background-color: #1e293b; padding: 10px 14px; border-radius: 6px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(self.card_summary)
        self._update_stats_display()

    def _on_browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model Checkpoint", "", "PyTorch Models (*.pt)")
        if file_path:
            self.combo_base_model.insertItem(0, file_path)
            self.combo_base_model.setCurrentIndex(0)
            self._on_load_model()

    def _on_load_model(self):
        model_name = self.combo_base_model.currentText().strip()
        self.surgery.load_model(model_name)
        self._populate_table()
        self._update_stats_display()

    def _populate_table(self):
        self._table_updating = True
        self.table.setRowCount(len(self.surgery.layers))
        
        for row, layer in enumerate(self.surgery.layers):
            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, chk_item)
            
            # Idx
            idx_item = QTableWidgetItem(str(layer.index))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, idx_item)
            
            # Name
            self.table.setItem(row, 2, QTableWidgetItem(layer.name))
            
            # Type
            type_item = QTableWidgetItem(layer.layer_type)
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 3, type_item)
            
            # Total Params
            p_str = f"{layer.params:,}"
            p_item = QTableWidgetItem(p_str)
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 4, p_item)
            
            # Trainable Params
            tr_str = f"{layer.trainable_params:,}"
            tr_item = QTableWidgetItem(tr_str)
            tr_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(row, 5, tr_item)
            
            # Status
            if layer.is_frozen:
                st_str = "Frozen ❄️"
            elif layer.is_pruned:
                st_str = "Pruned ✂️"
            else:
                st_str = "Active ✅"
            st_item = QTableWidgetItem(st_str)
            st_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 6, st_item)
            
            # Sparsity %
            sp_str = f"{layer.sparsity_pct:.1f}%"
            sp_item = QTableWidgetItem(sp_str)
            sp_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 7, sp_item)
            
            # Details
            self.table.setItem(row, 8, QTableWidgetItem(layer.details))
            
        self.spin_slice_idx.setMaximum(max(0, len(self.surgery.layers) - 1))
        self._table_updating = False

    def _get_checked_indices(self) -> List[int]:
        indices = []
        for row in range(self.table.rowCount()):
            chk = self.table.item(row, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                idx_item = self.table.item(row, 1)
                if idx_item:
                    indices.append(int(idx_item.text()))
        return indices

    def _bulk_table_check(self, state: bool):
        for row in range(self.table.rowCount()):
            chk = self.table.item(row, 0)
            if chk:
                chk.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def _freeze_selected(self):
        checked = self._get_checked_indices()
        if not checked:
            QMessageBox.information(self, "No Layers Selected", "Please check the box next to layers you wish to freeze.")
            return
        self.surgery.freeze_layers(checked)
        self._populate_table()
        self._update_stats_display()

    def _unfreeze_selected(self):
        checked = self._get_checked_indices()
        currently_frozen = [l.index for l in self.surgery.layers if l.is_frozen]
        new_frozen = [i for i in currently_frozen if i not in checked]
        self.surgery.freeze_layers(new_frozen)
        self._populate_table()
        self._update_stats_display()

    def _apply_freeze_preset(self):
        preset_text = self.combo_freeze_preset.currentText()
        if "Backbone" in preset_text:
            backbone_indices = [i for i in range(min(10, len(self.surgery.layers)))]
            self.surgery.freeze_layers(backbone_indices)
        elif "Detect Head" in preset_text:
            all_except_head = [l.index for l in self.surgery.layers if l.layer_type != "Detect"]
            self.surgery.freeze_layers(all_except_head)
        elif "Unfreeze" in preset_text:
            self.surgery.freeze_layers([])
            
        self._populate_table()
        self._update_stats_display()

    def _on_slice_model(self):
        k = self.spin_slice_idx.value()
        confirm = QMessageBox.question(
            self, "Confirm Model Slice",
            f"Are you sure you want to slice the model up to layer {k} and remove all subsequent layers?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.surgery.slice_model(k)
            self._populate_table()
            self._update_stats_display()

    def _on_remove_selected_node(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.information(self, "No Layer Selected", "Please select a row in the layer table to remove.")
            return
            
        row = selected_rows[0].row()
        idx = int(self.table.item(row, 1).text())
        confirm = QMessageBox.question(
            self, "Remove Node/Layer",
            f"Remove layer index {idx} ({self.surgery.layers[row].name}) from network?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.surgery.remove_node_layer(idx)
            self._populate_table()
            self._update_stats_display()

    def _apply_pruning(self):
        checked = self._get_checked_indices()
        if not checked:
            QMessageBox.information(self, "No Layers Selected", "Please check the box next to layers you wish to prune.")
            return
            
        ratio = self.slider_prune.value() / 100.0
        method_str = "l1_unstructured" if "L1" in self.combo_prune_method.currentText() else "ln_structured"
        self.surgery.prune_layers(checked, amount=ratio, method=method_str)
        self._populate_table()
        self._update_stats_display()
        QMessageBox.information(self, "Pruning Complete", f"Successfully pruned {len(checked)} layers by {ratio*100:.0f}%.")

    def _on_export_model(self):
        default_name = f"pruned_{Path(self.surgery.model_path).stem}.pt"
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Modified Model Checkpoint", default_name, "PyTorch Model (*.pt)")
        if save_path:
            out = self.surgery.save_checkpoint(save_path)
            frozen_indices = [l.index for l in self.surgery.layers if l.is_frozen]
            self.model_configured.emit(out, frozen_indices)
            QMessageBox.information(self, "Model Exported", f"Successfully exported modified model to:\n{out}")

    def _update_stats_display(self):
        stats = self.surgery.get_stats()
        self.card_summary.setText(
            f"📊 Architecture Stats:  Layers: {stats.total_layers}  |  "
            f"Total Params: {stats.total_params:,}  |  "
            f"Trainable Params: {stats.trainable_params:,}  |  "
            f"Frozen Layers: {stats.frozen_layers_count} ❄️  |  "
            f"Pruned Layers: {stats.pruned_layers_count} ✂️  |  "
            f"Overall Sparsity: {stats.overall_sparsity}%"
        )
        frozen_indices = [l.index for l in self.surgery.layers if l.is_frozen]
        self.model_configured.emit(self.surgery.model_path, frozen_indices)

    def get_configured_model_info(self) -> dict:
        frozen_indices = [l.index for l in self.surgery.layers if l.is_frozen]
        return {
            'model_path': self.surgery.model_path,
            'frozen_layers': frozen_indices
        }
