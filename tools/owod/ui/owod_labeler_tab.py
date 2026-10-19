"""
OWOD Auto-Labeling Tab.
Runs Open-World Object Detection (YOLO-World, SAM, Custom, or Fallback) on selected media files.
Streams live preview bounding boxes and exports YOLO dataset with data.yaml.
"""

from pathlib import Path
from typing import List, Optional
import time
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QLineEdit, QFileDialog, QComboBox, QSlider,
    QProgressBar, QGroupBox, QCheckBox, QFormLayout, QTextEdit,
    QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from ..core.media_scanner import MediaScanner, MediaItem
from ..core.video_processor import VideoProcessor
from ..core.owod_engine import OWODDetector, Detection
from ..core.dataset_generator import DatasetGenerator

class AutoLabelWorker(QThread):
    """Background worker that executes OWOD inference on images and video frames."""
    
    frame_processed = pyqtSignal(np.ndarray, list, str)  # annotated_rgb, detections, status_text
    progress_updated = pyqtSignal(int, int)  # current, total
    stats_updated = pyqtSignal(int, int)  # processed_count, plates_found
    log_message = pyqtSignal(str)
    dataset_ready = pyqtSignal(str)  # data.yaml path
    finished = pyqtSignal()
    
    def __init__(
        self,
        items: List[MediaItem],
        sampling_cfg: dict,
        model_name: str,
        prompt: str,
        conf_thresh: float,
        iou_thresh: float,
        device: str,
        output_dir: str,
        train_ratio: float,
        save_crops: bool,
        save_vis: bool
    ):
        super().__init__()
        self.items = items
        self.sampling_cfg = sampling_cfg
        self.model_name = model_name
        self.prompt = prompt
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.save_crops = save_crops
        self.save_vis = save_vis
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        self.log_message.emit("🚀 Starting OWOD Auto-Labeling...")
        self.log_message.emit(f"• Model: {self.model_name}")
        self.log_message.emit(f"• Prompt: '{self.prompt}'")
        self.log_message.emit(f"• Device: {self.device} | Conf: {self.conf_thresh} | IoU: {self.iou_thresh}")
        self.log_message.emit(f"• Target Output: {self.output_dir}")
        
        # Initialize detector
        detector = OWODDetector(
            model_name_or_path=self.model_name,
            prompt=self.prompt,
            conf_threshold=self.conf_thresh,
            iou_threshold=self.iou_thresh,
            device=self.device
        )
        
        # Initialize dataset generator
        generator = DatasetGenerator(
            output_dir=self.output_dir,
            train_ratio=self.train_ratio,
            save_crops=self.save_crops,
            save_visualizations=self.save_vis
        )
        
        total_items = len(self.items)
        processed_count = 0
        total_plates_found = 0
        
        for idx, item in enumerate(self.items):
            if self._is_stopped:
                self.log_message.emit("⏹️ Auto-labeling stopped by user.")
                break
                
            self.progress_updated.emit(idx + 1, total_items)
            
            if item.media_type == 'image':
                try:
                    # Read image
                    img_bgr = cv2.imread(str(item.path))
                    if img_bgr is None:
                        continue
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    
                    detections = detector.predict(img_rgb)
                    total_plates_found += len(detections)
                    
                    if detections:
                        unique_name = getattr(item, 'unique_identifier', item.path.stem)
                        generator.save_sample(
                            base_name=unique_name,
                            image_rgb=img_rgb,
                            detections=detections
                        )
                        
                    annotated = detector.draw_detections(img_rgb, detections)
                    status_text = f"[{idx+1}/{total_items}] Image: {item.filename} | Plates: {len(detections)}"
                    self.frame_processed.emit(annotated, detections, status_text)
                    self.stats_updated.emit(processed_count + 1, total_plates_found)
                    processed_count += 1
                except Exception as e:
                    self.log_message.emit(f"Error on {item.filename}: {e}")
                    
            elif item.media_type == 'video':
                self.log_message.emit(f"🎬 Processing Video: {item.filename} ({getattr(item, 'root_path', item.path.parent).name})...")
                frame_step = self.sampling_cfg.get('frame_step', 15)
                max_frames = self.sampling_cfg.get('max_frames', None)
                
                for f_idx, timestamp, frame_rgb in VideoProcessor.extract_frames(
                    item.path, sample_step=frame_step, max_frames=max_frames
                ):
                    if self._is_stopped:
                        break
                        
                    detections = detector.predict(frame_rgb)
                    total_plates_found += len(detections)
                    
                    if detections:
                        unique_vid_name = getattr(item, 'unique_identifier', item.path.stem)
                        base_name = f"{unique_vid_name}_f{f_idx:06d}"
                        generator.save_sample(
                            base_name=base_name,
                            image_rgb=frame_rgb,
                            detections=detections
                        )
                        
                    annotated = detector.draw_detections(frame_rgb, detections)
                    status_text = f"[{idx+1}/{total_items}] Video: {item.filename} (Frame {f_idx}) | Plates: {len(detections)}"
                    self.frame_processed.emit(annotated, detections, status_text)
                    self.stats_updated.emit(processed_count + 1, total_plates_found)
                    processed_count += 1
                    
        # Generate data.yaml
        yaml_path = generator.generate_data_yaml()
        self.log_message.emit(f"✅ Generated dataset config: {yaml_path}")
        self.log_message.emit(f"🎉 Auto-labeling complete! Processed {processed_count} frames, labeled {total_plates_found} plates.")
        self.dataset_ready.emit(str(yaml_path))
        self.finished.emit()


class OWODLabelerTab(QWidget):
    """Tab 2: Open-World Object Detection & Dataset Generation."""
    
    dataset_created = pyqtSignal(str)  # emits yaml path
    
    def __init__(self, scanner: MediaScanner):
        super().__init__()
        self.scanner = scanner
        self.worker: Optional[AutoLabelWorker] = None
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # 1. Config Header Grid
        cfg_group = QGroupBox("⚙️ OWOD Model & Target Configuration")
        cfg_layout = QFormLayout(cfg_group)
        cfg_layout.setContentsMargins(12, 14, 12, 12)
        cfg_layout.setSpacing(8)
        
        # Model selector
        h_model = QHBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "yolov8s-worldv2.pt (Fast YOLO-World v2)",
            "yolov8m-worldv2.pt (Balanced YOLO-World v2)",
            "yolov8l-worldv2.pt (High-Accuracy YOLO-World v2)",
            "yolov8x-worldv2.pt (Maximum Accuracy YOLO-World v2)",
            "yolo11s.pt (YOLO11 Small Base)",
            "yolo11n.pt (YOLO11 Nano Base)",
            "Custom Model Checkpoint...",
            "Heuristic Fast CV Detector (No Download Required)"
        ])
        h_model.addWidget(self.combo_model, 1)
        
        self.btn_browse_model = QPushButton("📁 Custom...")
        self.btn_browse_model.setObjectName("btnSecondary")
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        h_model.addWidget(self.btn_browse_model)
        cfg_layout.addRow("Detection Model:", h_model)
        
        # Prompt
        self.edit_prompt = QLineEdit("license plate")
        self.edit_prompt.setPlaceholderText("e.g. license plate, vehicle registration plate, car plate")
        self.edit_prompt.setToolTip("Open-world text prompt for zero-shot detection")
        cfg_layout.addRow("Prompt Classes:", self.edit_prompt)
        
        # Sliders: Confidence & IoU
        h_sliders = QHBoxLayout()
        
        lbl_conf = QLabel("Confidence: 0.25")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(5, 95)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(lambda v: lbl_conf.setText(f"Confidence: {v/100:.2f}"))
        h_sliders.addWidget(lbl_conf)
        h_sliders.addWidget(self.slider_conf)
        
        lbl_iou = QLabel("IoU / NMS: 0.45")
        self.slider_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_iou.setRange(10, 95)
        self.slider_iou.setValue(45)
        self.slider_iou.valueChanged.connect(lambda v: lbl_iou.setText(f"IoU / NMS: {v/100:.2f}"))
        h_sliders.addWidget(lbl_iou)
        h_sliders.addWidget(self.slider_iou)
        
        # Device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda:0 (GPU)", "cpu"])
        h_sliders.addWidget(QLabel("Device:"))
        h_sliders.addWidget(self.combo_device)
        
        cfg_layout.addRow("Detection Parameters:", h_sliders)
        
        # Output directory
        h_out = QHBoxLayout()
        self.edit_output_dir = QLineEdit(str(Path("curated_lpr_dataset/yolo_license_plates").resolve()))
        h_out.addWidget(self.edit_output_dir, 1)
        self.btn_browse_out = QPushButton("📁 Browse...")
        self.btn_browse_out.setObjectName("btnSecondary")
        self.btn_browse_out.clicked.connect(self._on_browse_output)
        h_out.addWidget(self.btn_browse_out)
        cfg_layout.addRow("Output Dataset Dir:", h_out)
        
        # Output options
        h_opts = QHBoxLayout()
        self.chk_crops = QCheckBox("Save Plate Crops (for OCR/Inspection)")
        self.chk_crops.setChecked(True)
        h_opts.addWidget(self.chk_crops)
        
        self.chk_vis = QCheckBox("Save Visualized Bounding Box Images")
        self.chk_vis.setChecked(True)
        h_opts.addWidget(self.chk_vis)
        
        lbl_split = QLabel("Train/Val Split: 80% / 20%")
        self.slider_split = QSlider(Qt.Orientation.Horizontal)
        self.slider_split.setRange(50, 95)
        self.slider_split.setValue(80)
        self.slider_split.valueChanged.connect(lambda v: lbl_split.setText(f"Train/Val Split: {v}% / {100-v}%"))
        h_opts.addWidget(lbl_split)
        h_opts.addWidget(self.slider_split)
        
        cfg_layout.addRow("Export Options:", h_opts)
        layout.addWidget(cfg_group)
        
        # 2. Main Central Splitter (Left: Live Visualizer, Right: Live Logs & Counters)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        
        # Left: Live Preview Canvas
        grp_canvas = QGroupBox("👁️ Live OWOD Detection Preview")
        v_can = QVBoxLayout(grp_canvas)
        v_can.setContentsMargins(10, 14, 10, 10)
        
        self.lbl_canvas = QLabel("Live detected frames with license plate boxes will appear here...")
        self.lbl_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_canvas.setStyleSheet("background-color: #0b1120; border-radius: 6px; color: #64748b; font-style: italic;")
        self.lbl_canvas.setMinimumSize(420, 280)
        v_can.addWidget(self.lbl_canvas, 1)
        
        self.lbl_live_status = QLabel("Ready to start.")
        self.lbl_live_status.setStyleSheet("color: #38bdf8; font-weight: 500;")
        v_can.addWidget(self.lbl_live_status)
        
        splitter.addWidget(grp_canvas)
        
        # Right: Log & Stats Panel
        grp_logs = QGroupBox("📋 Activity Log & Detection Stats")
        v_log = QVBoxLayout(grp_logs)
        v_log.setContentsMargins(10, 14, 10, 10)
        v_log.setSpacing(8)
        
        # Stats Badges
        h_stats = QHBoxLayout()
        self.lbl_stat_processed = QLabel("Frames: 0")
        self.lbl_stat_processed.setStyleSheet("background-color: #1e293b; color: #38bdf8; padding: 6px; border-radius: 4px; font-weight: bold;")
        h_stats.addWidget(self.lbl_stat_processed)
        
        self.lbl_stat_plates = QLabel("Plates Found: 0")
        self.lbl_stat_plates.setStyleSheet("background-color: #1e293b; color: #34d399; padding: 6px; border-radius: 4px; font-weight: bold;")
        h_stats.addWidget(self.lbl_stat_plates)
        v_log.addLayout(h_stats)
        
        # Console Log Box
        self.txt_logs = QTextEdit()
        self.txt_logs.setReadOnly(True)
        self.txt_logs.setStyleSheet("background-color: #0b1120; font-family: 'Consolas', monospace; font-size: 11px;")
        v_log.addWidget(self.txt_logs, 1)
        
        splitter.addWidget(grp_logs)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)
        
        # 3. Bottom Action Bar & Progress
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(10)
        
        self.btn_start = QPushButton("▶️ Start Auto-Labeling")
        self.btn_start.setObjectName("btnSuccess")
        self.btn_start.setMinimumHeight(38)
        self.btn_start.clicked.connect(self._on_start_clicked)
        bot_bar.addWidget(self.btn_start, 1)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        bot_bar.addWidget(self.btn_stop)
        
        layout.addLayout(bot_bar)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

    def _on_browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Custom Model Checkpoint", "", "PyTorch Models (*.pt *.onnx *.engine)"
        )
        if file_path:
            self.combo_model.insertItem(0, file_path)
            self.combo_model.setCurrentIndex(0)

    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Dataset Directory", self.edit_output_dir.text())
        if folder:
            self.edit_output_dir.setText(folder)

    def _on_start_clicked(self):
        selected_items = self.scanner.get_selected_items()
        if not selected_items:
            QMessageBox.warning(self, "No Media Selected", "Please select folders, images, or videos in Tab 1 before starting.")
            return
            
        model_str = self.combo_model.currentText()
        if "(" in model_str:
            model_key = model_str.split(" ")[0].strip()
        else:
            model_key = model_str.strip()
            
        dev = "cuda:0" if "cuda" in self.combo_device.currentText().lower() else "cpu"
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.txt_logs.clear()
        
        parent_tab = self.parentWidget()
        sampling_cfg = {'frame_step': 15, 'max_frames': 100}
        if hasattr(parent_tab, "media_tab"):
            sampling_cfg = parent_tab.media_tab.get_sampling_config()
            
        self.worker = AutoLabelWorker(
            items=selected_items,
            sampling_cfg=sampling_cfg,
            model_name=model_key,
            prompt=self.edit_prompt.text().strip(),
            conf_thresh=self.slider_conf.value() / 100.0,
            iou_thresh=self.slider_iou.value() / 100.0,
            device=dev,
            output_dir=self.edit_output_dir.text().strip(),
            train_ratio=self.slider_split.value() / 100.0,
            save_crops=self.chk_crops.isChecked(),
            save_vis=self.chk_vis.isChecked()
        )
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.stats_updated.connect(self._on_stats_updated)
        self.worker.log_message.connect(self._on_log_message)
        self.worker.dataset_ready.connect(self._on_dataset_ready)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_stop_clicked(self):
        if self.worker:
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def _on_frame_processed(self, img_rgb: np.ndarray, detections: list, status_text: str):
        self.lbl_live_status.setText(status_text)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl_canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_canvas.setPixmap(pix)

    def _on_progress_updated(self, current: int, total: int):
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"%p% ({current}/{total} items)")

    def _on_stats_updated(self, processed: int, plates: int):
        self.lbl_stat_processed.setText(f"Frames: {processed}")
        self.lbl_stat_plates.setText(f"Plates Found: {plates}")

    def _on_log_message(self, msg: str):
        self.txt_logs.append(msg)

    def _on_dataset_ready(self, yaml_path: str):
        self.dataset_created.emit(yaml_path)

    def _on_worker_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
