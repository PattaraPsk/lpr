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
├── run.bat                             # Global interactive launcher for all tools
├── run_detection.bat                   # CAM01 / CCTV video detection runner
├── run_owod_studio.bat                 # Shortcut: launch OWOD Studio GUI
├── run_tests.bat                       # Automated ISO compliance test runner (ISO 29119)
├── autorun_owod.bat                    # Shortcut: run batch OWOD on dataset/
├── STANDARDS.md                        # Enterprise & ISO Standards Compliance Manual
│
├── tests/                              # Automated test suites (ISO/IEC/IEEE 29119)
│   └── test_iso_compliance.py          # ISO 3166-2:TH, 8601, 25012, 5259, 27001, 42001 verification
│
├── dataset/                            # Local datasets & raw media (git-ignored)
│   ├── curated_lpr_dataset/            # Approved Thai LPR dataset images
│   ├── yolo_owod_labeled_dataset/      # Auto-labeled YOLO dataset (train/val/crops)
│   └── Raw Video Record/               # CCTV / Dashcam source recordings (ISO Partitioned)
│
└── tools/                              # Tooling suites
    ├── README.md                       # Overview of tools directory
    ├── run.bat                         # Interactive selector menu (CLI/batch)
    ├── detect_cam_videos.py            # CAM01 Video detection & tracking
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
- `[1]` Launch **OWOD Studio GUI** (Auto-Labeling & YOLO Pruning)
- `[2]` Auto-Run **Batch OWOD** on all files in `dataset/` (GPU CUDA)
- `[3]` Launch **Thai LPR Dataset Annotator & Reviewer** (`DataManipulator`)
- `[4]` Launch **ISO-Standard RTSP Stream Collector** (`rtsp_collector`)
- `[5]` Run **CAM01 Video Detection & Auto-Tracking Runner**
- `[6]` Run **ISO Compliance & Quality Assurance Test Suite** (`run_tests.bat`)

### Option 2: Direct Tool Launchers
- **Run ISO Standards Test Suite**:
  ```cmd
  run_tests.bat
  # or: python -m unittest discover tests
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

## 🛠️ Tool Suites Overview

### 1. DataManipulator (`tools/DataManipulator/`)
- Desktop review and annotation GUI for Thai License Plate datasets.
- Manages standardized 5-element filename structure:
  ```text
  {name}_{plate_type}_{d3}_{d4}_{province}.jpg
  ```
- Supports all 10 Thai vehicle plate categories, Thai province search, duplicate cleanup (`_dup`), subfolder grouping, and 9-column check table review.

### 2. OWOD Studio (`tools/owod/`)
- **Zero-Shot Auto-Labeling**: Uses YOLO-World v2 models to detect license plates across image and video directories automatically.
- **YOLO Model Surgery & Pruning**: Inspect model layers, freeze backbone/neck, slice model depth, and apply structured or unstructured weight pruning.
- **Fine-Tuning**: Trains custom pruned models directly against auto-labeled license plate datasets with live training metrics and curves.
- **Orientation & Vehicle Context**: Supports Front/Back vehicle orientation models (`rear_front_best.pt` / `RAER-TNORF`) and COCO vehicle context matching.

### 3. ISO RTSP Stream Collector (`tools/rtsp_collector/`)
- **Systematic CCTV & RTSP Ingestion**: Continuous 24/7 stream capture with auto-recovery and exponential backoff.
- **ISO Standard Compliance**:
  - **ISO 8601**: Standardized UTC/local timestamping and `CAM01/YYYY/MM/DD/` partitioning.
  - **ISO/IEC 14496-14**: Lossless H.264 stream copy (`-c:v copy`) into MP4 containers with FastStart `moov` atom placement.
  - **ISO/IEC 25012**: SHA-256 data integrity hashes and FFprobe container/frame validation.
  - **ISO/IEC 27001**: Credential decoupling via `.env` and automated secret masking in logs and manifests.
  - **ISO/IEC 20547-3**: Staging-to-Bronze Big Data landing zone architecture.

### 4. Video Detection & DLT Dataset Builder (`tools/detect_cam_videos.py` & `tools/owod/build_dlt_lpr_dataset.py`)
- **Incremental Video Processing**: Deduplicates already-processed CCTV clips using an atomic JSON tracker (`dataset/processed_videos_tracker.json`).
- **DLT Plate Classification**: Maps COCO vehicles (car, motorcycle, truck, bus) to Thailand DLT plate categories with geometric aspect ratio and plausibility filtering.
- **ISO 5259 Data Quality Reports**: Automatically generates `data_quality_report.json` with class imbalance, box containment, and FIPS SHA-256 verification.

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
