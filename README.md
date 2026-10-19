# License Plate Reader (LPR) & OWOD Studio

An end-to-end repository for Thai License Plate Reader (LPR), dataset annotation, Open-World Object Detection (OWOD) auto-labeling, and YOLO architecture surgery/pruning.

---

## 📂 Workspace Folder Tree

```text
lpr/
├── .gitignore                          # Git ignore rules (datasets, weights, caches)
├── README.md                           # Main workspace documentation
├── run.bat                             # Global interactive launcher for all tools
├── run_owod_studio.bat                 # Shortcut: launch OWOD Studio GUI
├── autorun_owod.bat                    # Shortcut: run batch OWOD on dataset/
│
├── dataset/                            # Local datasets & raw media (git-ignored)
│   ├── curated_lpr_dataset/            # Approved Thai LPR dataset images
│   ├── yolo_owod_labeled_dataset/      # Auto-labeled YOLO dataset (train/val/crops)
│   └── Raw Video Record/               # CCTV / Dashcam source recordings (ISO Partitioned)
│
└── tools/                              # Tooling suites
    ├── README.md                       # Overview of tools directory
    ├── run.bat                         # Interactive selector menu (CLI/batch)
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
        ├── main.py                     # Entry point for OWOD Studio GUI
        ├── README.md                   # Detailed OWOD Studio guide
        ├── run.bat                     # Windows batch runner for OWOD Studio
        └── test_owod_studio.py         # OWOD unit & integration test suite
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

### Option 2: Direct Tool Launchers
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

### 3. ISO RTSP Stream Collector (`tools/rtsp_collector/`)
- **Systematic CCTV & RTSP Ingestion**: Continuous 24/7 stream capture with auto-recovery and exponential backoff.
- **ISO Standard Compliance**:
  - **ISO 8601**: Standardized UTC/local timestamping and `CAM01/YYYY/MM/DD/` partitioning.
  - **ISO/IEC 14496-14**: Lossless H.264 stream copy (`-c:v copy`) into MP4 containers with FastStart `moov` atom placement.
  - **ISO/IEC 25012**: SHA-256 data integrity hashes and FFprobe container/frame validation.
  - **ISO/IEC 27001**: Credential decoupling via `.env` and automated secret masking in logs and manifests.
  - **ISO/IEC 20547-3**: Staging-to-Bronze Big Data landing zone architecture.
