# Thai License Plate Recognition (Thai LPR) & OWOD Studio

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20%7C%20PyTorch%20%7C%20YOLO--World-green.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20Edge-orange.svg)]()
[![Thai DLT](https://img.shields.io/badge/Standard-77%20Provinces%20Regex-purple.svg)]()

Production-grade, commercial Computer Vision monorepo for Thai License Plate Recognition (ALPR/ANPR), Open-World Object Detection (OWOD) auto-labeling, ISO-compliant RTSP video collection, and edge inference deployment.

---

## 📂 Workspace Architecture Overview

```text
lpr/
├── .env.example                        # Decoupled environment & RTSP credential template (ISO 27001)
├── .gitignore                          # Strict model weight, dataset, and artifact ignore rules
├── AGENTS.md                           # Edge guardrails & engineering non-negotiables
├── LPR_PIPELINE_STATUS.md              # Architecture audit and master migration status
├── README.md                           # This document
│
├── run.bat                             # Master interactive launcher for all tool suites
├── run_detection.bat                   # Batch CCTV video detection runner with JSON tracking
├── run_owod_studio.bat                 # 1-Click launcher: OWOD Studio GUI
├── autorun_owod.bat                    # 1-Click launcher: Batch GPU OWOD auto-labeling
├── run_pipeline.bat                    # 1-Click launcher: Edge Inference Pipeline benchmark
├── run_curate_dataset.bat              # 1-Click launcher: Dataset curation & SQLite registry
├── run_relabel_gui.bat                 # 1-Click launcher: Web-based plate relabeling GUI
│
├── pipeline/                           # Pillar 1: Production Edge Inference Pipeline
│   ├── configs/                        # Runtime YAML configs (detection, tracking, RTSP)
│   ├── models/                         # Local model weights & ONNX runtimes
│   ├── scripts/                        # Windows & Linux execution launchers
│   ├── src/                            # Modular Python package (capture, detector, ocr, pipeline, utils)
│   ├── tests/                          # Unit test suite & sample test media
│   ├── main.py                         # Unified CLI entrypoint (Stream, Video, Image, Benchmark)
│   ├── requirements.txt                # Pinned Python 3.10 dependencies
│   └── README.md                       # Complete pipeline documentation
│
├── tools/                              # Tool Suites & Ingestion Engines
│   ├── README.md                       # Overview of tools directory
│   ├── run.bat                         # Interactive tool selector menu
│   ├── detect_cam_videos.py            # Batch CCTV video detection runner with incremental tracking
│   │
│   ├── rtsp_collector/                 # ISO-Compliant RTSP Stream Ingestion Engine
│   │   ├── cli.py                      # CLI tool (probe, record, verify, status)
│   │   ├── collector.py                # Core stream capture engine & watchdog
│   │   ├── validator.py                # ISO 25012 data quality & SHA-256 verifier
│   │   ├── manifest.py                 # ISO 8601 lineage catalog & audit manager
│   │   ├── security.py                 # ISO 27001 credential decoupling & masking
│   │   ├── config.yaml                 # Ingestion parameters & storage configuration
│   │   ├── run.bat                     # Quick Windows runner batch script
│   │   └── README.md                   # Complete ISO standard documentation
│   │
│   ├── DataManipulator/                # Thai LPR Dataset Annotator & Reviewer GUI
│   │   ├── app/                        # PyQt6 application components
│   │   ├── main.py                     # Entry point for DataManipulator
│   │   ├── run.bat                     # Windows batch runner for DataManipulator
│   │   └── README.md                   # Detailed DataManipulator guide
│   │
│   ├── owod/                           # Open-World Object Detection & YOLO Pruning Studio
│   │   ├── core/                       # Core algorithms (OWOD engine, model surgery, trainer)
│   │   ├── ui/                         # PyQt6 multi-tab studio UI
│   │   ├── build_dlt_lpr_dataset.py    # DLT vehicle-to-plate classifier & dataset builder
│   │   ├── autorun_dataset.py          # Standalone batch OWOD script (GPU/CUDA accelerated)
│   │   ├── main.py                     # Entry point for OWOD Studio GUI
│   │   ├── run.bat                     # Windows batch runner for OWOD Studio
│   │   └── README.md                   # Detailed OWOD Studio guide
│   │
│   └── relabel_gui/                    # Web-based Interactive Plate Relabeling GUI
│       ├── server.py                   # FastAPI / Python backend server
│       ├── static/                     # HTML/CSS/JS web interface
│       └── README.md                   # Relabel GUI guide
│
├── training_detection/                 # Pillar 3: Plate Detection Model Training
│   ├── configs/                        # Training hyperparameters (AdamW, augmentations)
│   ├── prepare_dataset.py              # Dataset unpacker & data.yaml generator
│   ├── train.py                        # Ultralytics detector training engine
│   ├── export_onnx.py                  # PyTorch .pt -> ONNX converter with dynamic batching
│   ├── evaluate.py                     # mAP50, mAP50-95 & latency evaluator
│   ├── train_detector.bat              # 1-Click interactive training launcher
│   └── README.md                       # Detection training guide
│
├── training_ocr/                       # Pillar 4: Thai OCR Model Training & Synthetic Data
│   ├── synthetic_data/                 # 77-Province synthetic plate generator & augmentor
│   ├── character_bank/                 # Hough deskewing & CCA character glyph segmenter
│   ├── models/                         # PyTorch CRNN (CNN + BiLSTM + CTC) OCR training engine
│   ├── generate_synthetic.bat          # 1-Click synthetic plate generator
│   ├── train_ocr.bat                   # 1-Click interactive OCR trainer
│   └── README.md                       # OCR training guide
│
├── selen-lpr/                          # High-accuracy OCR & Classification Interface
│   ├── configs/                        # Categories and provincial configurations
│   ├── interfaces/                     # IO specs and external service interfaces
│   ├── src/                            # Vision-attention OCR & DLT rule engine
│   ├── tests/                          # Pipeline integration & model tests
│   └── tools/                          # Gemini auto-labeling & clip renderers
│
└── dataset/                            # Local datasets & raw media (git-ignored)
    ├── curated_lpr_dataset/            # Approved Thai LPR dataset images
    ├── yolo_owod_labeled_dataset/      # Auto-labeled YOLO dataset (train/val/crops)
    └── Raw Video Record/               # CCTV / Dashcam source recordings (ISO Partitioned)
```

---

## 🚀 Quick Start & Launchers

### Master Interactive Runner
Run `run.bat` at the repository root to launch any tool from the interactive menu:
```cmd
run.bat
```
- `[1]` Launch **OWOD Studio GUI** (Auto-Labeling & YOLO Pruning)
- `[2]` Auto-Run **Batch OWOD** on all files in `dataset/` (GPU CUDA)
- `[3]` Launch **Thai LPR Dataset Annotator & Reviewer** (`DataManipulator`)
- `[4]` Launch **ISO-Standard RTSP Stream Collector** (`rtsp_collector`)

### Dedicated Batch Launchers
| Workflow | Launcher Script | CLI Alternative |
| :--- | :--- | :--- |
| **Edge Pipeline Benchmark** | `run_pipeline.bat` | `python pipeline/main.py --mode benchmark` |
| **CCTV Video Detection & Tracking** | `run_detection.bat` | `python tools/detect_cam_videos.py --input-dir "dataset/Raw Video Record/CAM01"` |
| **OWOD Studio GUI** | `run_owod_studio.bat` | `python tools/owod/main.py` |
| **Batch OWOD Auto-Labeling** | `autorun_owod.bat` | `python tools/owod/autorun_dataset.py` |
| **Thai LPR Annotator (DataManipulator)** | `tools/DataManipulator/run.bat` | `python tools/DataManipulator/main.py` |
| **Web-Based Plate Relabeler** | `run_relabel_gui.bat` | `python tools/relabel_gui/server.py` |
| **DLT Dataset Builder** | — | `python tools/owod/build_dlt_lpr_dataset.py` |
| **Detection Training (YOLOv8/v11)** | `training_detection/train_detector.bat` | `python training_detection/train.py` |
| **OCR Training (CRNN + CTC)** | `training_ocr/train_ocr.bat` | `python -m training_ocr.models.train_ocr` |

---

## 🛠️ Core Functional Pillars

### 1. Production Edge Inference Pipeline (`pipeline/`)
The inference pipeline processes live RTSP video feeds, recordings, or single images with sub-50ms latency.
- **Detector**: ONNX Runtime or TensorRT accelerated YOLO models.
- **OCR**: Thai CRNN model decoding 77 provinces and alphanumeric sequences.
- **Validation**: Regex matching conforming to official Thailand DLT syntax.

For complete CLI options and configuration settings, see **[pipeline/README.md](file:///d:/Projects-anti/lpr/pipeline/README.md)**.

### 2. ISO RTSP Stream Collector (`tools/rtsp_collector/`)
Continuous 24/7 stream capture with auto-recovery and exponential backoff conforming to international standards:
- **ISO 8601**: Standardized UTC/local timestamping and `CAM01/YYYY/MM/DD/` partitioning.
- **ISO/IEC 14496-14**: Lossless H.264 stream copy into MP4 containers with FastStart `moov` atom placement.
- **ISO/IEC 25012**: SHA-256 data integrity hashes and FFprobe container/frame validation.
- **ISO/IEC 27001**: Credential decoupling via `.env` and automated secret masking in logs and manifests.

For detailed usage, see **[tools/rtsp_collector/README.md](file:///d:/Projects-anti/lpr/tools/rtsp_collector/README.md)**.

### 3. Open-World Object Detection & YOLO Surgery Studio (`tools/owod/`)
- **Zero-Shot Auto-Labeling**: Uses YOLO-World v2 models to detect license plates across image and video directories automatically.
- **YOLO Model Surgery & Pruning**: Inspect model layers, freeze backbone/neck, slice model depth, and apply structured or unstructured weight pruning.
- **Orientation & Vehicle Matching**: Supports Front/Back vehicle orientation models (`rear_front_best.pt` / `RAER-TNORF`) and COCO vehicle context matching.
- **DLT Dataset Builder**: `build_dlt_lpr_dataset.py` creates clean YOLO datasets with DLT vehicle classes (`car_plate`, `motorcycle_plate`, `truck_plate`, `bus_plate`).

For detailed usage, see **[tools/owod/README.md](file:///d:/Projects-anti/lpr/tools/owod/README.md)**.

### 4. CCTV Detection Runner & Incremental Video Tracker (`tools/detect_cam_videos.py`)
- Automatically traverses surveillance recordings, rotates upside-down feeds (180°), and executes multi-stage vehicle, plate, and orientation detection.
- Maintains atomic state in `dataset/processed_videos_tracker.json` to prevent re-processing identical video files across runs.

### 5. DataManipulator & Relabel GUI (`tools/DataManipulator/` & `tools/relabel_gui/`)
- **DataManipulator**: PyQt6 desktop annotator managing standardized 5-element filenames (`{name}_{plate_type}_{d3}_{d4}_{province}.jpg`), Thai province search, and duplicate cleanup.
- **Relabel GUI**: Lightweight browser interface for rapid visual verification and correction of cropped plate datasets.

### 6. Model Training Suites (`training_detection/` & `training_ocr/`)
- **Detection**: Custom YOLO training pipeline with automated ONNX export and validation against real test media.
- **OCR**: Synthetic generation covering all 77 Thai provinces, character bank extraction via Hough deskewing and CCA, and PyTorch CRNN CTC training.

---

## 🔒 Engineering Guardrails & System Prerequisites

- **Python Runtime**: Python 3.10.x (64-bit) recommended.
- **CUDA Acceleration**: NVIDIA GPU (RTX 3060/4060/4090) with CUDA 12.4 support; CPU ONNX fallback supported.
- **Model Artifacts Policy**: Raw weights (`.pt`, `.onnx`, `.engine`, `.torchscript`) are strictly git-ignored and must never be committed to git.
- **Storage & Lineage**: Raw media, crops, and datasets reside under `dataset/` and are tracked via ISO 8601 catalogs.
