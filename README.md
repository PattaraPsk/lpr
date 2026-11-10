# Computer Vision & License Plate Recognition (LPR) Project

An enterprise, edge-optimized Computer Vision framework for Thai License Plate Recognition (ALPR/ANPR), dataset preparation, CRNN/YOLO model training, and real-time inference.

---

## 📂 Project Architecture

```text
computer-vision-project/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── train.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiments.ipynb
├── src/
│   ├── data/
│   │   ├── prepare.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── model.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       ├── metrics.py
│       └── visualization.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── outputs/
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
└── scripts/
    ├── download_data.sh
    └── run_training.sh
```

### Quick Start

```bash
# 1. Prepare data splits
python -m src.data.prepare --source-dir data/raw --splits-dir data/splits

# 2. Run training with configs
python -m src.training.train --dataset-config configs/dataset.yaml --model-config configs/model.yaml --train-config configs/train.yaml

# 3. Evaluate checkpoint on validation split
python -m src.training.evaluate --checkpoint outputs/checkpoints/best_model.pt --split data/splits/val.txt

# 4. Run inference on single image
python -m src.inference.predict --image path/to/plate.jpg --checkpoint outputs/checkpoints/best_model.pt

# 5. Run unit tests
python -m unittest tests/test_data.py tests/test_model.py
```

---

# License Plate Reader (LPR) & OWOD Studio

An end-to-end repository for Thai License Plate Reader (LPR), dataset annotation, Open-World Object Detection (OWOD) auto-labeling, and YOLO architecture surgery/pruning.

---

## 📂 Workspace Folder Tree

```text
lpr/
├── .env.example                        # Decoupled environment & RTSP credential template (ISO 27001)
├── .gitignore                          # Git ignore rules (datasets, weights, caches)
├── AGENTS.md                           # Edge guardrails & engineering non-negotiables
├── pyproject.toml                      # PEP 517/621 & ISO 19770 software packaging metadata
├── README.md                           # Main workspace documentation
├── requirements.txt                    # Pinned edge runtime dependencies
├── demo_realtime.py                    # Root entry point for Real-Time Thai LPR Demo
├── run.bat                             # Global interactive launcher for all tools
├── run_demo_realtime.bat               # Shortcut: interactive Real-Time Thai LPR Demo runner
├── run_detection.bat                   # CAM01 / CCTV video detection runner
├── run_owod_studio.bat                 # Shortcut: launch OWOD Studio GUI
├── run_tests.bat                       # Automated ISO compliance test runner (ISO 29119)
├── run_track_lpr.bat                   # Shortcut: batch video tracker & snapshot extractor
├── train_detector.bat                  # Shortcut: train YOLO Plate & Vehicle Detector
├── train_reader.bat                    # Shortcut: train VisionAttentionOCR Plate Reader
├── autorun_owod.bat                    # Shortcut: run batch OWOD on dataset/
├── STANDARDS.md                        # Enterprise & ISO Standards Compliance Manual
│
├── models/                             # Consolidated neural model checkpoints
│   ├── detector/                       # Plate & Front/Rear YOLO models (yolov11n, yolov8s)
│   ├── reader/                         # Thai OCR models
│   │   ├── selen_ocr_v011/             # SelenOCR v0.1.1 (CCTV Resilient & Low-Res Invariant Reader)
│   │   ├── selen_ocr_v010/             # SelenOCR v0.1.0 (>95% Clean Synthetic Baseline)
│   │   ├── paddleocr_v3/               # PaddleOCR Baseline v3.0 (Single-Line 11-Token)
│   │   ├── easyocr_v3/                 # EasyOCR Baseline v3.0 (Single-Line 11-Token)
│   │   └── thai_plate_reader.pth       # Legacy Thai OCR model
│   └── MODEL_CATALOG.md                # Architecture, inputs, outputs, and validation metrics
│
├── annotations/                        # Version-controlled labeling progress & metadata (ISO 3166-2:TH / ISO 25012)
│   ├── plates_session.json             # Canonical DataManipulator labeling session (auto-mirrored)
│   ├── approved_plates.json            # ISO 3166-2:TH normalized approved plate metadata
│   ├── approved_plates.csv             # Tabular export for quick inspection & training
│   └── LABELING_PROGRESS.md            # Live labeling audit report and statistics
│
├── tests/                              # Automated test suites (ISO/IEC/IEEE 29119)
│   ├── test_alpr_package.py            # Unit tests for modular tools.alpr package
│   ├── test_demo_realtime.py           # Real-Time ALPR end-to-end integration tests
│   └── test_iso_compliance.py          # ISO 3166-2:TH, 8601, 25012, 5259, 27001, 42001 verification
│
├── dataset/                            # Local datasets & raw media (git-ignored)
│   ├── zip/                            # Compressed dataset archives (.zip)
│   ├── raw/                            # Source recordings (CCTV / Dashcam / MP4)
│   ├── snapshot/                       # Extracted frame snapshots & vehicle crops
│   ├── labeled/                        # Auto-labeled & detected datasets (YOLO / OCR)
│   ├── verified/                       # Human-curated & approved staging datasets
│   └── ready/                          # Release versions (YYYYMMDDhhmm/plate_folder/)
│
└── tools/                              # Tooling suites
    ├── README.md                       # Overview of tools directory
    ├── run.bat                         # Interactive selector menu (CLI/batch)
    ├── demo_realtime.py                # Backward-compatible real-time demo launcher
    ├── train_models.py                 # Unified detector & OCR model training hub
    ├── process_and_track_lpr.py        # Batch video LPR tracker & snapshot extractor
    ├── detect_cam_videos.py            # CAM01 Video detection & tracking
    │
    ├── alpr/                           # Modularized Real-Time Thai ALPR Edge Engine
    │   ├── __init__.py                 # Clean package public exports
    │   ├── config.py                   # Centralized model resolution, fonts & thresholds
    │   ├── stream.py                   # Thread-safe zero-latency frame reader
    │   ├── tracker.py                  # IoU/distance tracker & consensus voting
    │   ├── detector.py                 # Coarse-to-fine cascader & plate detector
    │   ├── hud.py                      # Thai Unicode font cache, badges & telemetry HUD
    │   └── demo.py                     # Interactive orchestration, hotkeys & playlist
    │
    ├── rtsp_collector/                 # ISO-Compliant RTSP Stream Ingestion Engine
    │   ├── cli.py                      # CLI tool (probe, record, verify, status)
    │   ├── collector.py                # Core stream capture engine & watchdog
    │   ├── validator.py                # ISO 25012 data quality & SHA-256 verifier
    │   ├── manifest.py                 # ISO 8601 lineage catalog & audit manager
    │   ├── security.py                 # ISO 27001 credential decoupling & masking
    │   ├── config.yaml                 # Ingestion parameters & storage configuration
    │   ├── run.bat                     # Quick Windows runner batch script
    │   └── README.md                   # Complete ISO standard documentation
    │
    ├── DataManipulator/                # Thai LPR Dataset Annotator & Reviewer GUI
    │   ├── app/                        # PyQt6 application components
    │   │   ├── check_view.py           # 9-column batch review & summary dialog
    │   │   ├── image_viewer.py         # Zoom, pan, and 1:1 image canvas
    │   │   ├── main_window.py          # Main annotator GUI window
    │   │   ├── plate_types.py          # Thai plate types definitions & PascalCase mapping
    │   │   ├── plate_type_combo.py     # Custom plate type combo box
    │   │   ├── provinces.py            # 77 Thai provinces + special symbols
    │   │   ├── searchable_combo.py     # Fuzzy/searchable dropdown widget
    │   │   └── session_manager.py      # Session persistence (.lpr_session.json)
    │   ├── create_sample_data.py       # Mock data generator (creates local sample_images/)
    │   ├── main.py                     # Entry point for DataManipulator
    │   ├── README.md                   # Detailed DataManipulator guide
    │   ├── run.bat                     # Windows batch runner for DataManipulator
    │   └── test_app.py                 # Comprehensive unit & GUI test suite
    │
    └── owod/                           # Open-World Object Detection & YOLO Pruning Studio
        ├── core/                       # Core algorithms & engines
        │   ├── dataset_generator.py    # YOLO format exporter (train/val splits, data.yaml)
        │   ├── media_scanner.py        # Recursive image/video discovery scanner
        │   ├── model_surgery.py        # Layer inspection, freezing, slicing, and pruning
        │   ├── owod_engine.py          # YOLO-World v2 zero-shot detection engine
        │   ├── trainer_engine.py       # YOLO fine-tuning engine with live training metrics
        │   └── video_processor.py      # Video frame extraction & sampling engine
        ├── ui/                         # PyQt6 multi-tab studio UI
        │   ├── main_window.py          # Main OWOD Studio window
        │   ├── media_selector_tab.py   # Media discovery tree & frame sampler
        │   ├── owod_labeler_tab.py     # Real-time OWOD inference & labeling canvas
        │   ├── model_surgery_tab.py    # Layer freezing, slicing, and pruning tab
        │   ├── trainer_tab.py          # Fine-tuning dashboard & metric plots
        │   └── styles.py               # Modern dark theme QSS stylesheet
        ├── autorun_dataset.py          # Standalone batch OWOD script (GPU/CUDA accelerated)
        ├── build_dlt_lpr_dataset.py    # DLT vehicle-to-plate classifier & dataset builder
        ├── inspect_sources.py          # Source inspection script
        ├── main.py                     # Entry point for OWOD Studio GUI
        ├── run.bat                     # Windows batch runner for OWOD Studio
        ├── test_cam01_orientation.py   # Camera orientation test script
        ├── test_detect.py              # Detection test script
        ├── test_matcher.py             # Matcher test script
        ├── test_owod_studio.py         # OWOD unit & integration test suite
        ├── test_rf_lpr.py              # Front/Back & plate detection test
        ├── test_samples.py             # Sample test script
        ├── visualize_dlt_dataset.py    # Dataset visualizer script
        └── visualize_lpr_batch.py      # Batch visualizer script
```

---

## 🚀 Quick Start & Launchers

### Option 1: Global Tool Launcher
Run `run.bat` at the repository root to choose an action from the interactive menu:
```cmd
run.bat
```
- `[0]` Launch **Real-Time Thai LPR Demo** (Webcam / RTSP / Video)
- `[1]` Launch **OWOD Studio GUI** (Auto-Labeling & YOLO Pruning)
- `[2]` Auto-Run **Batch OWOD** on all files in `dataset/` (GPU CUDA)
- `[3]` Launch **Thai LPR Dataset Annotator & Reviewer** (`DataManipulator`)
- `[4]` Launch **ISO-Standard RTSP Stream Collector** (`rtsp_collector`)
- `[5]` Run **CAM01 Video Detection & Auto-Tracking Runner**
- `[6]` Run **ISO Compliance & Quality Assurance Test Suite** (`run_tests.bat`)
- `[7]` Train **Plate Detector & Vehicle Classifier** (`train_detector.bat`)
- `[8]` Train **Thai Plate Reader OCR** (`train_reader.bat`)
- `[9]` Run **Batch LPR Video Tracking & Snapshot Extractor** (`run_track_lpr.bat`)

### Option 2: Direct Tool Launchers
- **Run ISO Standards Test Suite**:
  ```cmd
  run_tests.bat
  # or: python -m unittest discover tests
  ```
- **Real-Time ALPR Demo**:
  ```cmd
  run_demo_realtime.bat
  # or: python demo_realtime.py
  ```
- **Train Detector & OCR**:
  ```cmd
  train_detector.bat
  train_reader.bat
  ```
- **OWOD Studio GUI**:
  ```cmd
  run_owod_studio.bat
  # or: python tools/owod/main.py
  ```
- **Thai LPR Annotator (`DataManipulator`)**:
  ```cmd
  cd tools/DataManipulator
  run.bat
  # or: python main.py
  ```
- **Batch OWOD Labeling (Command Line)**:
  ```cmd
  autorun_owod.bat
  # or: python tools/owod/autorun_dataset.py
  ```
- **CAM01 / CCTV Video Detection & Auto-Tracking**:
  ```cmd
  run_detection.bat
  # or: python tools/detect_cam_videos.py --input-dir "dataset/Raw Video Record/CAM01"
  ```
- **DLT License Plate Dataset Builder**:
  ```cmd
  python tools/owod/build_dlt_lpr_dataset.py
  ```

---

## 🧠 State-of-the-Art OCR: SelenOCR v0.1.1 (CCTV Resilient & Edge Optimized)

**SelenOCR v0.1.1** is the production Thai License Plate OCR engine designed specifically for low-latency edge deployment (CPU / Edge GPU). It bridges the gap between clean physical plates and extreme optical degradation (20px–48px low-resolution crops, sensor noise, motion blur, and headlight bloom).

### Architecture Highlights
- **Backbone**: MobileNetV3-Large with Multi-Scale Feature Pyramid Network (FPN) fusing stages to $d_{\text{model}}=384$.
- **Sequence Modeling**: Adaptive Positional Sequence Pooling ($W'=40$) + 4-Layer Transformer Encoder (6 heads, feedforward 1024) + 2-Layer Bidirectional LSTM ($d_{\text{hidden}}=256$).
- **Multi-Task Heads**: Simultaneous CTC alphanumeric sequence head (145 classes), vehicle plate type head (13 classes), and province classification head (81 classes).
- **Single-Line 11-Token Formulation**: Single-line sequence representation `plateType d3 d4 province` ($\le 11$ tokens max: `start + plateType + 3 + 4 + 1 + stop`), strictly respecting the 41 DLT consonants and OCR remapping (`ฃ->ข`, `ฅ->ค`, `ฏ->ฎ`).
- **Memory & Speed**: ~16.2M parameters (~61.6 MB float32), inference latency **0.8–1.2 ms** (> 1,000 FPS) on RTX 4090 GPU / ~15–25 ms on Edge CPU.

### 70:30 Golden Ratio Curriculum (170,000 Images)
To prevent stroke representation collapse on look-alike Thai character loops (`ข`/`ช`, `ด`/`ต`, `ท`/`ฑ`), SelenOCR v0.1.1 was trained on an empirically calibrated 70:30 corpus:
- **Dataset v3.0 (Clean Anchor)**: 120,000 images (100k clean physical geometry $\le 5^\circ$ + 10k blur `- - - -` + 10k non-plate `+ + + +`).
- **Dataset v3.1 (CCTV & Low-Res Add-on)**: 50,000 images (48,817 synthetic downsampled 20px–48px with stroke dilation/erosion & compression + 1,183 authentic CCTV anchor plates).

### Cross-Dataset Empirical Benchmark Summary

| Benchmark Dataset | Samples | SelenOCR v0.1.0 (Exact Acc) | SelenOCR v0.1.1 (Exact Acc) | Delta | Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Anchor CCTV (`202610261140`)** | 1,183 | 4.06% | **96.70%** | **+92.64%** 🚀 | 1.22 ms | 822.8 |
| **Clean Synthetic Target (`v3.0`)** | 2,000 | 95.85% | **99.65%** | **+3.80%** | 1.02 ms | 979.3 |
| **Degraded Environmental (`dirty10-blur5-angle5`)** | 2,000 | 58.90% | **68.75%** | **+9.85%** | 0.84 ms | 1193.3 |
| **CCTV & Low-Res Add-on (`v3.1`)** | 2,000 | — | **73.35%** | *New Baseline* | 0.87 ms | 1148.5 |
| **Expanded Real CCTV/Mobile (`202611030335`)** | 2,000 | 4.10% | **15.60%** | **+11.50%** | 0.88 ms | 1136.5 |
| **Mobile Camera Synthetic** | 1,500 | 12.30% | **23.40%** | **+11.10%** | 0.91 ms | 1104.3 |
| **CCTV Profile Synthetic** | 1,500 | 0.80% | **13.53%** | **+12.73%** | 0.81 ms | 1229.6 |

### Quick CLI Commands for SelenOCR
```bash
# 1. Evaluate SelenOCR v0.1.1 across all project benchmark datasets
python tools/evaluate_selen_ocr_on_all_datasets.py --checkpoint models/reader/selen_ocr_v011/best_reader.pth --version v0.1.1

# 2. Train SelenOCR v0.1.1 on combined 170k corpus (warm-started from v0.1.0)
python tools/train_selen_ocr_v011.py --epochs 10 --batch-size 128 --lr 2.5e-4

# 3. Generate Clean Target Synthetic Dataset (Dataset v3.0, 120,000 images)
python tools/generate_v3_clean_synthetic_dataset.py

# 4. Generate CCTV & Low-Resolution Add-on Dataset (Dataset v3.1, 50,000 images)
python tools/generate_v31_cctv_addon_dataset.py --num-synthetic 48817 --num-workers 16
```

---

## 🛠️ Tool Suites Overview

### 1. ALPR Edge Engine (`tools/alpr/` & `demo_realtime.py`)
- **Zero-Latency Ingestion** (`stream.py`): Multi-threaded camera/RTSP frame reader with automatic frame-dropping to eliminate queue latency.
- **Full Inference Pipeline** (`detector.py`):
  - Plate & Front/Rear orientation detection (`YOLOv11n`).
  - Contextual vehicle classification (`YOLOv8s`).
  - Spatial multi-object tracking across frames (`tracker.py`).
  - Thai OCR reading (`Edge Thai OCR Reader`) with ISO 3166-2:TH province resolution.
- **Edge-AI HUD Overlay** (`hud.py`): High-speed cached Thai Unicode font rendering (Tahoma / Leelawadee) with corner brackets, real-time FPS/latency telemetry, and interactive controls (`[Q] Quit`, `[SPACE] Pause`, `[R] Flip 180°`, `[S] Snapshot`, `[T] Boxes`, `[H] HUD`, `[Drag Mouse] Set ROI`, `[X] Clear ROI`, `[N/P] Next/Prev VDO`).

### 2. Unified Model Training Hub (`tools/train_models.py`)
- Standalone CLI & script suite to train or fine-tune both the YOLO Plate & Vehicle Detector and the VisionAttentionOCR reader.
- Automatic candidate weight resolution from `models/` with fallback logic.

### 3. DataManipulator (`tools/DataManipulator/`)
- Desktop review and annotation GUI for Thai License Plate datasets.
- Manages standardized 5-element filename structure:
  ```text
  {name}_{plate_type}_{d3}_{d4}_{province}.jpg
  ```
- Supports all 10 Thai vehicle plate categories, Thai province search, duplicate cleanup (`_dup`), subfolder grouping, and 9-column check table review.

### 4. OWOD Studio (`tools/owod/`)
- **Zero-Shot Auto-Labeling**: Uses YOLO-World v2 models to detect license plates across image and video directories automatically.
- **YOLO Model Surgery & Pruning**: Inspect model layers, freeze backbone/neck, slice model depth, and apply structured or unstructured weight pruning.
- **Fine-Tuning**: Trains custom pruned models directly against auto-labeled license plate datasets with live training metrics and curves.
- **Orientation & Vehicle Context**: Supports Front/Back vehicle orientation models (`vehicle_plate_detector.pt`) and COCO vehicle context matching.

### 5. ISO RTSP Stream Collector (`tools/rtsp_collector/`)
- **Systematic CCTV & RTSP Ingestion**: Continuous 24/7 stream capture with auto-recovery and exponential backoff.
- **ISO Standard Compliance**:
  - **ISO 8601**: Standardized UTC/local timestamping and `CAM01/YYYY/MM/DD/` partitioning.
  - **ISO/IEC 14496-14**: Lossless H.264 stream copy (`-c:v copy`) into MP4 containers with FastStart `moov` atom placement.
  - **ISO/IEC 25012**: SHA-256 data integrity hashes and FFprobe container/frame validation.
  - **ISO/IEC 27001**: Credential decoupling via `.env` and automated secret masking in logs and manifests.
  - **ISO/IEC 20547-3**: Staging-to-Bronze Big Data landing zone architecture.

---

## 🏛️ International Standards Compliance

This repository enforces compliance across all data, vision, and streaming components:
- **ISO 3166-2:TH**: Official codes for all 77 Thai provinces (`TH-10` to `TH-96`) in `tools/DataManipulator/app/provinces.py`.
- **ISO/IEC 10646 & Unicode 15.0 NFC**: Canonical script normalization across all annotations and string lookups.
- **ISO 8601**: Standard UTC timestamps and ISO duration representation (`PT...S`).
- **ISO/IEC 25012**: SHA-256 data integrity hashes and atomic state writes.
- **ISO/IEC 5259-2:2023**: AI Data Quality auditing for YOLO datasets (`tools/owod/core/quality_auditor.py`).
- **ISO/IEC 27001**: Credential masking in URLs and user path sanitization.
- **ISO/IEC 42001**: AI Management System & strict exclusion of binary weights from version control.
- **ISO/IEC/IEEE 29119**: Automated test suite (`run_tests.bat`).

For full implementation details, see [STANDARDS.md](STANDARDS.md).
