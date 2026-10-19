"""
YOLO Fine-Tuning Tab.
Configures and launches model training with custom frozen layers, loss streaming, and mAP monitoring.
"""

from pathlib import Path
from typing import List, Optional
import os
import subprocess
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox,
    QProgressBar, QGroupBox, QFormLayout, QTextEdit, QMessageBox,
    QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.trainer_engine import TrainerEngine

class TrainerTab(QWidget):
    """Tab 4: YOLO Fine-Tuning Runner & Monitoring Dashboard."""
    
    def __init__(self):
        super().__init__()
        self.trainer = TrainerEngine()
        self.trainer.log_signal.connect(self._on_log)
        self.trainer.epoch_signal.connect(self._on_epoch)
        self.trainer.metrics_signal.connect(self._on_metrics)
        self.trainer.finished_signal.connect(self._on_finished)
        self.trainer.error_signal.connect(self._on_error)
        self.last_best_pt: Optional[str] = None
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # 1. Training Setup Group
        grp_setup = QGroupBox("⚙️ Fine-Tuning Setup & Checkpoints")
        form_s = QFormLayout(grp_setup)
        form_s.setContentsMargins(12, 14, 12, 12)
        form_s.setSpacing(8)
        
        # Dataset YAML
        h_data = QHBoxLayout()
        self.edit_data_yaml = QLineEdit(str(Path("curated_lpr_dataset/yolo_license_plates/data.yaml").resolve()))
        h_data.addWidget(self.edit_data_yaml, 1)
        self.btn_browse_yaml = QPushButton("📁 Browse...")
        self.btn_browse_yaml.setObjectName("btnSecondary")
        self.btn_browse_yaml.clicked.connect(self._on_browse_yaml)
        h_data.addWidget(self.btn_browse_yaml)
        form_s.addRow("Dataset YAML (data.yaml):", h_data)
        
        # Model Checkpoint
        h_model = QHBoxLayout()
        self.edit_model_pt = QLineEdit("yolov8s.pt")
        h_model.addWidget(self.edit_model_pt, 1)
        self.btn_browse_pt = QPushButton("📁 Browse...")
        self.btn_browse_pt.setObjectName("btnSecondary")
        self.btn_browse_pt.clicked.connect(self._on_browse_pt)
        h_model.addWidget(self.btn_browse_pt)
        form_s.addRow("Model Checkpoint (.pt):", h_model)
        
        # Frozen layers input
        self.edit_frozen_layers = QLineEdit("0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
        self.edit_frozen_layers.setToolTip("Comma-separated layer indices to freeze during training (e.g. 0,1,2,3...9)")
        form_s.addRow("Frozen Layer Indices:", self.edit_frozen_layers)
        
        layout.addWidget(grp_setup)
        
        # 2. Hyperparameters Grid
        grp_hp = QGroupBox("🎛️ Training Hyperparameters")
        grid_hp = QGridLayout(grp_hp)
        grid_hp.setContentsMargins(12, 14, 12, 12)
        grid_hp.setSpacing(10)
        
        # Epochs
        grid_hp.addWidget(QLabel("Epochs:"), 0, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(50)
        grid_hp.addWidget(self.spin_epochs, 0, 1)
        
        # Batch Size
        grid_hp.addWidget(QLabel("Batch Size:"), 0, 2)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        grid_hp.addWidget(self.spin_batch, 0, 3)
        
        # Image Size
        grid_hp.addWidget(QLabel("Img Size:"), 0, 4)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(128, 2048)
        self.spin_imgsz.setSingleStep(32)
        self.spin_imgsz.setValue(640)
        grid_hp.addWidget(self.spin_imgsz, 0, 5)
        
        # Learning Rate
        grid_hp.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 1.0)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setValue(0.01)
        grid_hp.addWidget(self.spin_lr, 1, 1)
        
        # Optimizer
        grid_hp.addWidget(QLabel("Optimizer:"), 1, 2)
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(["auto", "AdamW", "SGD"])
        grid_hp.addWidget(self.combo_opt, 1, 3)
        
        # Device
        grid_hp.addWidget(QLabel("Device:"), 1, 4)
        self.combo_dev = QComboBox()
        self.combo_dev.addItems(["cuda:0", "cpu"])
        grid_hp.addWidget(self.combo_dev, 1, 5)
        
        layout.addWidget(grp_hp)
        
        # 3. Live Metrics Cards Dashboard
        grp_metrics = QGroupBox("📊 Live Training Metrics")
        h_met = QHBoxLayout(grp_metrics)
        h_met.setContentsMargins(12, 14, 12, 12)
        h_met.setSpacing(12)
        
        self.card_epoch = self._create_metric_card("Epoch", "0 / 50", "#38bdf8")
        h_met.addWidget(self.card_epoch)
        
        self.card_box_loss = self._create_metric_card("Box Loss", "0.0000", "#f87171")
        h_met.addWidget(self.card_box_loss)
        
        self.card_cls_loss = self._create_metric_card("Cls Loss", "0.0000", "#fbbf24")
        h_met.addWidget(self.card_cls_loss)
        
        self.card_map50 = self._create_metric_card("mAP@50", "0.000", "#34d399")
        h_met.addWidget(self.card_map50)
        
        self.card_map50_95 = self._create_metric_card("mAP@50-95", "0.000", "#a78bfa")
        h_met.addWidget(self.card_map50_95)
        
        layout.addWidget(grp_metrics)
        
        # 4. Console Logs Output
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setStyleSheet("background-color: #0b1120; font-family: 'Consolas', monospace; font-size: 11px;")
        layout.addWidget(self.txt_console, 1)
        
        # 5. Progress Bar & Action Buttons
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(10)
        
        self.btn_train = QPushButton("🚀 Start Fine-Tuning")
        self.btn_train.setObjectName("btnSuccess")
        self.btn_train.setMinimumHeight(38)
        self.btn_train.clicked.connect(self._on_start_train)
        bot_bar.addWidget(self.btn_train, 1)
        
        self.btn_stop = QPushButton("🛑 Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_train)
        bot_bar.addWidget(self.btn_stop)
        
        self.btn_open_runs = QPushButton("📁 Open Best Weights")
        self.btn_open_runs.setObjectName("btnSecondary")
        self.btn_open_runs.setMinimumHeight(38)
        self.btn_open_runs.clicked.connect(self._on_open_runs)
        bot_bar.addWidget(self.btn_open_runs)
        
        layout.addLayout(bot_bar)

    def _create_metric_card(self, title: str, default_val: str, color_hex: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("background-color: #0b1120; border-radius: 6px; padding: 6px;")
        v = QVBoxLayout(card)
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(2)
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lbl_t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl_t)
        lbl_v = QLabel(default_val)
        lbl_v.setStyleSheet(f"color: {color_hex}; font-size: 16px; font-weight: bold;")
        lbl_v.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card.value_label = lbl_v
        v.addWidget(lbl_v)
        return card

    def set_dataset_yaml(self, yaml_path: str):
        self.edit_data_yaml.setText(yaml_path)

    def set_model_and_frozen_layers(self, model_path: str, frozen_layers: List[int]):
        self.edit_model_pt.setText(model_path)
        if frozen_layers:
            self.edit_frozen_layers.setText(", ".join(str(i) for i in frozen_layers))
        else:
            self.edit_frozen_layers.setText("")

    def _on_browse_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset YAML", "", "YAML files (*.yaml *.yml)")
        if file_path:
            self.edit_data_yaml.setText(file_path)

    def _on_browse_pt(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "PyTorch Models (*.pt)")
        if file_path:
            self.edit_model_pt.setText(file_path)

    def _parse_frozen_layers(self) -> List[int]:
        txt = self.edit_frozen_layers.text().strip()
        if not txt:
            return []
        indices = []
        for part in txt.split(","):
            part = part.strip()
            if part.isdigit():
                indices.append(int(part))
        return indices

    def _on_start_train(self):
        yaml_path = self.edit_data_yaml.text().strip()
        model_path = self.edit_model_pt.text().strip()
        
        if not Path(yaml_path).exists():
            QMessageBox.warning(self, "Dataset Missing", f"Could not find dataset YAML at:\n{yaml_path}\nPlease generate dataset in Tab 2 first.")
            return
            
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.txt_console.clear()
        self.progress_bar.setValue(0)
        
        frozen = self._parse_frozen_layers()
        dev = self.combo_dev.currentText()
        
        self.trainer.start_training(
            model_path=model_path,
            data_yaml=yaml_path,
            epochs=self.spin_epochs.value(),
            batch_size=self.spin_batch.value(),
            imgsz=self.spin_imgsz.value(),
            frozen_layers=frozen,
            lr0=self.spin_lr.value(),
            optimizer=self.combo_opt.currentText(),
            device=dev,
            project_dir="runs/detect",
            run_name="lpr_finetuned"
        )

    def _on_stop_train(self):
        self.trainer.stop_training()
        self.btn_stop.setEnabled(False)

    def _on_log(self, msg: str):
        self.txt_console.append(msg)

    def _on_epoch(self, current: int, total: int):
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.card_epoch.value_label.setText(f"{current} / {total}")

    def _on_metrics(self, m: dict):
        if 'box_loss' in m:
            self.card_box_loss.value_label.setText(f"{m['box_loss']:.4f}")
        if 'cls_loss' in m:
            self.card_cls_loss.value_label.setText(f"{m['cls_loss']:.4f}")
        if 'mAP50' in m:
            self.card_map50.value_label.setText(f"{m['mAP50']:.3f}")
        if 'mAP50_95' in m:
            self.card_map50_95.value_label.setText(f"{m['mAP50_95']:.3f}")

    def _on_finished(self, best_pt: str):
        self.last_best_pt = best_pt
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Training Complete", f"Fine-tuning complete!\nBest model saved to:\n{best_pt}")

    def _on_error(self, err: str):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "Training Error", err)

    def _on_open_runs(self):
        if self.last_best_pt and Path(self.last_best_pt).exists():
            target_dir = Path(self.last_best_pt).parent
        else:
            target_dir = Path("runs/detect").resolve()
            target_dir.mkdir(parents=True, exist_ok=True)
            
        os.startfile(str(target_dir))
