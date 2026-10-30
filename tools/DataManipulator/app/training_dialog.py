"""
Interactive Model Training Hub Dialog for DataManipulator.
Supports fine-tuning Plate Detector & Vehicle Classifier (YOLO) and Thai Plate Reader (OCR).
Runs in background QThread with real-time log streaming and progress tracking.
"""

from pathlib import Path
from typing import Optional
import os
import sys

from PyQt6.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QProgressBar, QTextEdit, QGroupBox, QFormLayout,
    QFileDialog, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent


class TrainingWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # current, total
    finished_signal = pyqtSignal(str)      # result path
    error_signal = pyqtSignal(str)

    def __init__(self, task: str, params: dict):
        super().__init__()
        self.task = task
        self.params = params
        self._is_stopped = False

    def run(self):
        try:
            from tools.train_models import train_yolo_detector, train_plate_reader

            def stream_log(msg: str):
                self.log_signal.emit(msg)

            if self.task == "detector":
                result = train_yolo_detector(
                    data_yaml=self.params["data"],
                    base_model=self.params.get("weights"),
                    epochs=self.params.get("epochs", 30),
                    batch_size=self.params.get("batch", 16),
                    imgsz=self.params.get("imgsz", 640),
                    device=self.params.get("device", "cuda:0"),
                    project_dir=self.params.get("output", "runs/train"),
                    run_name=self.params.get("run_name", "detector_run"),
                    log_callback=stream_log
                )
            elif self.task == "reader":
                result = train_plate_reader(
                    data_dir=self.params["data"],
                    base_weights=self.params.get("weights"),
                    epochs=self.params.get("epochs", 30),
                    batch_size=self.params.get("batch", 32),
                    lr=self.params.get("lr", 0.0005),
                    device=self.params.get("device", "cuda:0"),
                    output_dir=self.params.get("output", "runs/train/reader"),
                    log_callback=stream_log
                )
            else:
                self.error_signal.emit(f"Unknown task: {self.task}")
                return

            if result:
                self.finished_signal.emit(str(result))
            else:
                self.error_signal.emit("Training completed with no checkpoint returned.")

        except Exception as e:
            import traceback
            err_text = traceback.format_exc()
            self.error_signal.emit(f"Training exception: {e}\n{err_text}")


class TrainingDialog(QDialog):
    """Integrated Model Training Studio Dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🚀 Model Training Studio (GPU Accelerated)")
        self.resize(880, 680)
        self.setMinimumSize(750, 520)

        self.worker_thread: Optional[TrainingWorkerThread] = None

        self._init_ui()
        self._apply_styles()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header Banner
        header = QFrame()
        header.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 6px 12px;")
        h_layout = QHBoxLayout(header)
        h_title = QLabel("⚡ Thai LPR Model Training & Fine-Tuning Studio")
        h_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        h_layout.addWidget(h_title)
        h_layout.addStretch()

        import torch
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode"
        lbl_dev = QLabel(f"GPU: {dev_name}")
        lbl_dev.setStyleSheet("color: #34d399; font-weight: bold;")
        h_layout.addWidget(lbl_dev)
        layout.addWidget(header)

        # Tabs for Detector vs Reader
        self.tabs = QTabWidget()

        # Tab 1: Detector
        self.tab_detector = self._create_detector_tab()
        self.tabs.addTab(self.tab_detector, "🎯 1. Plate Detector & Vehicle Classifier (YOLO)")

        # Tab 2: Reader
        self.tab_reader = self._create_reader_tab()
        self.tabs.addTab(self.tab_reader, "🔤 2. Thai Plate Reader (OCR Edge Reader)")

        layout.addWidget(self.tabs)

        # Execution Controls
        grp_ctrl = QGroupBox("Execution Controls")
        c_layout = QHBoxLayout(grp_ctrl)

        self.btn_start = QPushButton("▶ Start Training")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setMinimumHeight(36)
        self.btn_start.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 6px 16px;")
        self.btn_start.clicked.connect(self._on_start_training)
        c_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("⏹ Stop Training")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(36)
        self.btn_stop.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; border-radius: 4px; padding: 6px 16px;")
        self.btn_stop.clicked.connect(self._on_stop_training)
        c_layout.addWidget(self.btn_stop)

        c_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumWidth(260)
        c_layout.addWidget(self.progress_bar)

        layout.addWidget(grp_ctrl)

        # Console Log Window
        grp_log = QGroupBox("Training Console Logs")
        v_log = QVBoxLayout(grp_log)
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setFont(QFont("Consolas", 9))
        self.txt_logs.setStyleSheet("background-color: #0f172a; color: #f1f5f9; border: 1px solid #334155; border-radius: 4px;")
        v_log.addWidget(self.txt_logs)
        layout.addWidget(grp_log, 1)

    def _create_detector_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        # Dataset YAML
        h_data = QHBoxLayout()
        def_yaml = ROOT_DIR / "dataset" / "labeled" / "dataset-vehicle-orientation" / "data.yaml"
        if not def_yaml.exists():
            def_yaml = ROOT_DIR / "dataset" / "dataset-vehicle-orientation" / "data.yaml"
        self.edit_det_yaml = QLineEdit(str(def_yaml))
        h_data.addWidget(self.edit_det_yaml, 1)
        btn_b_yaml = QPushButton("Browse...")
        btn_b_yaml.clicked.connect(lambda: self._browse_file(self.edit_det_yaml, "YAML Files (*.yaml *.yml)"))
        h_data.addWidget(btn_b_yaml)
        form.addRow("Dataset (data.yaml):", h_data)

        # Base Model
        h_model = QHBoxLayout()
        default_pt = ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt"
        if not default_pt.exists():
            default_pt = ROOT_DIR / "models" / "detector" / "yolov8s.pt"
        self.edit_det_pt = QLineEdit(str(default_pt))
        h_model.addWidget(self.edit_det_pt, 1)
        btn_b_pt = QPushButton("Browse...")
        btn_b_pt.clicked.connect(lambda: self._browse_file(self.edit_det_pt, "YOLO Weights (*.pt)"))
        h_model.addWidget(btn_b_pt)
        form.addRow("Base Weights (.pt):", h_model)

        # Hyperparameters
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Epochs:"))
        self.spin_det_epochs = QSpinBox()
        self.spin_det_epochs.setRange(1, 1000)
        self.spin_det_epochs.setValue(30)
        h_params.addWidget(self.spin_det_epochs)

        h_params.addWidget(QLabel("Batch Size:"))
        self.spin_det_batch = QSpinBox()
        self.spin_det_batch.setRange(1, 256)
        self.spin_det_batch.setValue(16)
        h_params.addWidget(self.spin_det_batch)

        h_params.addWidget(QLabel("Img Size:"))
        self.spin_det_imgsz = QSpinBox()
        self.spin_det_imgsz.setRange(320, 1280)
        self.spin_det_imgsz.setSingleStep(32)
        self.spin_det_imgsz.setValue(640)
        h_params.addWidget(self.spin_det_imgsz)

        h_params.addWidget(QLabel("Device:"))
        self.combo_det_dev = QComboBox()
        self.combo_det_dev.addItems(["cuda:0", "cpu"])
        h_params.addWidget(self.combo_det_dev)
        h_params.addStretch()

        form.addRow("Parameters:", h_params)
        return widget

    def _create_reader_tab(self) -> QWidget:
        widget = QWidget()
        form = QFormLayout(widget)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        # Dataset Folder
        h_data = QHBoxLayout()
        def_ocr = ROOT_DIR / "dataset" / "labeled" / "dataset-auto-label-a1341"
        if not def_ocr.exists():
            def_ocr = ROOT_DIR / "dataset" / "dataset-auto-label-a1341"
        self.edit_ocr_data = QLineEdit(str(def_ocr))
        h_data.addWidget(self.edit_ocr_data, 1)
        btn_b_data = QPushButton("Browse...")
        btn_b_data.clicked.connect(lambda: self._browse_dir(self.edit_ocr_data))
        h_data.addWidget(btn_b_data)
        form.addRow("Dataset Folder:", h_data)

        # Base Model
        h_model = QHBoxLayout()
        default_pth = ROOT_DIR / "models" / "reader" / "thai_plate_reader" / "best_norm_ED.pth"
        self.edit_ocr_pth = QLineEdit(str(default_pth))
        h_model.addWidget(self.edit_ocr_pth, 1)
        btn_b_pth = QPushButton("Browse...")
        btn_b_pth.clicked.connect(lambda: self._browse_file(self.edit_ocr_pth, "PyTorch Weights (*.pth *.pt)"))
        h_model.addWidget(btn_b_pth)
        form.addRow("Base Weights (.pth):", h_model)

        # Hyperparameters
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Epochs:"))
        self.spin_ocr_epochs = QSpinBox()
        self.spin_ocr_epochs.setRange(1, 200)
        self.spin_ocr_epochs.setValue(30)
        h_params.addWidget(self.spin_ocr_epochs)

        h_params.addWidget(QLabel("Batch Size:"))
        self.spin_ocr_batch = QSpinBox()
        self.spin_ocr_batch.setRange(1, 256)
        self.spin_ocr_batch.setValue(32)
        h_params.addWidget(self.spin_ocr_batch)

        h_params.addWidget(QLabel("Learning Rate:"))
        self.spin_ocr_lr = QDoubleSpinBox()
        self.spin_ocr_lr.setRange(0.00001, 0.05)
        self.spin_ocr_lr.setDecimals(5)
        self.spin_ocr_lr.setSingleStep(0.0001)
        self.spin_ocr_lr.setValue(0.0005)
        h_params.addWidget(self.spin_ocr_lr)

        h_params.addWidget(QLabel("Device:"))
        self.combo_ocr_dev = QComboBox()
        self.combo_ocr_dev.addItems(["cuda:0", "cpu"])
        h_params.addWidget(self.combo_ocr_dev)
        h_params.addStretch()

        form.addRow("Parameters:", h_params)
        return widget

    def _browse_file(self, target_line_edit: QLineEdit, file_filter: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", str(ROOT_DIR), file_filter)
        if path:
            target_line_edit.setText(path)

    def _browse_dir(self, target_line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", str(ROOT_DIR))
        if path:
            target_line_edit.setText(path)

    def _on_start_training(self):
        current_tab = self.tabs.currentIndex()
        if current_tab == 0:
            task = "detector"
            params = {
                "data": self.edit_det_yaml.text().strip(),
                "weights": self.edit_det_pt.text().strip() or None,
                "epochs": self.spin_det_epochs.value(),
                "batch": self.spin_det_batch.value(),
                "imgsz": self.spin_det_imgsz.value(),
                "device": self.combo_det_dev.currentText(),
                "output": "runs/train",
                "run_name": "vehicle_plate_detector"
            }
        else:
            task = "reader"
            params = {
                "data": self.edit_ocr_data.text().strip(),
                "weights": self.edit_ocr_pth.text().strip() or None,
                "epochs": self.spin_ocr_epochs.value(),
                "batch": self.spin_ocr_batch.value(),
                "lr": self.spin_ocr_lr.value(),
                "device": self.combo_ocr_dev.currentText(),
                "output": "runs/train/reader"
            }

        self.txt_logs.clear()
        self.txt_logs.append(f"[INIT] Starting training task: {task.upper()}...")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(10)

        self.worker_thread = TrainingWorkerThread(task, params)
        self.worker_thread.log_signal.connect(self._append_log)
        self.worker_thread.finished_signal.connect(self._on_training_finished)
        self.worker_thread.error_signal.connect(self._on_training_error)
        self.worker_thread.start()

    def _on_stop_training(self):
        if self.worker_thread and self.worker_thread.isRunning():
            self.txt_logs.append("[USER] Stop requested. Terminating worker thread...")
            self.worker_thread.terminate()
            self.worker_thread.wait(2000)
            self._append_log("[STOPPED] Training process aborted by user.")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_bar.setValue(0)

    def _append_log(self, text: str):
        self.txt_logs.append(text)
        # Scroll to bottom
        sb = self.txt_logs.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_training_finished(self, result_path: str):
        self.progress_bar.setValue(100)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"\n🎉 [SUCCESS] Training completed! Model saved to: {result_path}")
        QMessageBox.information(self, "Training Complete", f"Model training completed successfully!\n\nSaved to:\n{result_path}")

    def _on_training_error(self, err_text: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(0)
        self._append_log(f"\n❌ [ERROR] {err_text}")
        QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n\n{err_text[:300]}")

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 14px;
                font-weight: bold;
                color: #94a3b8;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #1e293b;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #0f172a;
                color: #94a3b8;
                padding: 8px 14px;
                border: 1px solid #334155;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #1e293b;
                color: #38bdf8;
            }
        """)
