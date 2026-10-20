# Thai LPR Project - Master Pipeline Status & Architecture Audit

**Target System**: Commercial Edge Thai License Plate Recognition (ALPR/ANPR)  
**Root Directory**: `d:\Projects-anti\lpr`  
**Target Runtime**: Python 3.10.x (64-bit) / Windows & Linux Edge / NVIDIA CUDA & CPU  
**Audit Date**: October 2026  

---

## 1. Full Workspace Audit & Inventory

### 1.1 Directory & Asset Inventory

| Subdirectory / Asset | Item Count | Key Files & Artifacts | Primary Role / Classification |
| :--- | :--- | :--- | :--- |
| **`zero_shot_license_plate_detection/`** | 65,472 | `main.py`, `crop_characters.py`, `font_analyzer.py`, `augment_license_plate.py`, `organize_crop_folders.py`, `generate_sample.py`, `fonts/` | **Core Vision & LPR Suite**: Zero-shot OWLv2/YOLO-World plate detector, Hough/CLAHE/projection character segmentation, font morphology analyzer, 77-province synthetic generator, and `AA`–`ZZ` dataset packager. |
| **`license_plate_sorter/`** | 31,116 | `main.py`, `requirements.txt`, `setup.bat`, `run.bat` | **VLM Plate Classifier & Sorter**: Qwen2-VL-2B/7B Vision-Language model classifying real physical plates vs fake digital ad graphics/Canva mockups vs auction plates, with OCR renaming. |
| **`batch_image_filter_suite/`** | 39,824 | `main.py`, `web_server.py`, `modules/filter_engine.py`, `modules/checkpoint.py`, `modules/file_router.py` | **Dataset Curation**: High-throughput Pillow byte-stream aspect ratio & dimension filtering with transactional JSONL checkpoint resume. *(Note: Contains nested duplicate directory `batch_image_filter_suite/batch_image_filter_suite/`).* |
| **`01_thai_lpr_cpu_edge/`** | 8,438 | `setup.bat`, `run.bat`, `requirements.txt`, `models/`, `data/`, `logs/` | **CPU Edge Runtime**: Enforces Python 3.10.11 for ONNX Runtime / FastAPI / PySide edge deployment. |
| **`lpr-dataset/`** | 3,254 | `fetch_thai_plates.py`, `setup.bat`, `thai_license_plate_fetch.bat`, 5 YOLOv11 dataset zip archives | **Dataset Harvester**: Scrapers for PlatesMania Thailand, Roboflow ALPR API, and local YOLO dataset cropping. |
| **`fb_image_downloader/`** | 7,586 | `fb_downloader/fb_fullres.py`, `fb_downloader/main.py`, `browser_session/` | **Social Scraper**: Automated Selenium crawler for vehicle photo extraction. |
| **`selen-lpr/`** | 1 | `io.txt` | **Interface Stub**: Interface placeholder (`Input: image / Output: s_x,s_y,s_w,s_h,d3,d4,province`). |
| **`02_lx_pt_patrol_cv/`** | 1 | `requirements.txt` | **Stale Workspace**: Former patrol anomaly detection directory (cleared). |
| **`scripts/`** | 2 | `check_environment.py`, `pipeline_manifest.json` | **Diagnostic & Metadata Tools**: Hardware and environment diagnostic tools. |
| **Root Archives & Zip Files** | 6 archives | `LPR-20261012T021756Z-1-001.zip` (636 MB), `batch_image_filter_suite.zip` (793 MB), `fb_image_downloader.zip`, `license_plate_sorter.zip`, `zero_shot_license_plate_detection.zip` | **Raw Media & Archive Copies**: Test videos (9 Thai dashcam MP4 files) and compressed backups. |

---

### 1.2 Pipeline Component Classification

#### A. Core Inference Code
- **Plate Detection & Localization**:
  - OWLv2 Open-Vocabulary Detector (`zero_shot_license_plate_detection/main.py`): Hugging Face `google/owlv2-base-patch16-ensemble` with queries `["license plate", "vehicle registration plate", "number plate"]`.
  - YOLO-World Zero-Shot Detector (`zero_shot_license_plate_detection/main.py`): Ultralytics `yolov8s-worldv2.pt`.
- **Character Segmentation & Morphology**:
  - `zero_shot_license_plate_detection/crop_characters.py`: Hough Line plate deskewing ($-25^\circ$ to $+25^\circ$), CLAHE adaptive contrast normalization, horizontal projection profiling splitting top alphanumeric glyphs from province text, Connected Component Analysis (CCA) contour extraction, screw/bolt filtering.
  - `zero_shot_license_plate_detection/font_analyzer.py`: Stroke width estimation via Euclidean distance transform, aspect ratio profiling (digits: ~0.58, Thai: ~0.72), intra-glyph/letter-to-number spacing ratios.
- **VLM Classification & OCR**:
  - `license_plate_sorter/main.py`: Qwen2-VL-2B/7B Instruct model performing 4-way classification and extracting Thai consonants, numbers, and province text.
- **Temporal Tracking & Centroid Smoothing**:
  - `SimpleBoxTracker`: Real-time IoU matching + Exponential Moving Average (EMA, $\alpha=0.75$) centroid smoothing to eliminate bounding box flicker.

#### B. Stream & Video Ingestion Logic
- **Real-Time Stream Capture**:
  - Wraps OpenCV `cv2.VideoCapture` supporting RTSP streams, USB webcams (`device=0`), and local MP4/AVI video files.
  - Frame-dropping logic: non-blocking queues (`in_queue.full()`) and output queue eviction (`out_queue.qsize() > 2`) to eliminate latency buffer accumulation.
- **Batch Dataset & Video Ingestion**:
  - `batch_image_filter_suite/modules/filter_engine.py`: Multi-worker Pillow byte stream pre-inspection.
  - `zero_shot_license_plate_detection/main.py`: `ThreadPoolExecutor` prefetch queue.
  - Real test media: 9 authentic Thai street dashcam videos extracted in `LPR-20261012T021756Z-1-001.zip`.

#### C. Thai Language Post-Processing & Regex Parsing Rules
- **Syntax Formats**:
  - Modern private car format: `[1-9][ก-ฮ]{2} [1-9][0-9]{0-3}` (e.g. `1กข 8920`, `9วย 4512`).
  - Classic vehicle format: `[ก-ฮ]{2} [1-9][0-9]{0-3}` (e.g. `กข 1234`).
  - Motorcycle / Special formats: `[1-9][ก-ฮ]{2} [0-9]{1-4}` or `[ก-ฮ]{3} [0-9]{1-4}`.
- **Provincial Validation**:
  - Canonical dictionary of all 77 official Thai provinces codified in `augment_license_plate.py`.
- **Filename Parsing & Sanitization**:
  - Windows safe-name sanitization: `re.sub(r'[\\/*?:"<>|]', "", text)`.
  - Structured naming standard: `{3digit}_{4digit}_{province}_{folderID}{imgID}.jpg`.

#### D. Dependency & Environment State
- Monorepo standard: **Python 3.10.x (64-bit)**.
- Hardware Target: NVIDIA GeForce RTX 4090 (24 GB VRAM) with CUDA 12.4 PyTorch wheels, and fallback to CPU ONNX Runtime.
- Scattered environments: 6 distinct `.venv` / `venv` directories totaling $>20\text{ GB}$ of duplicated libraries.

#### E. Redundant, Duplicated & Temporary Files
- Nested directory duplication: `batch_image_filter_suite/batch_image_filter_suite/`.
- Temporary session files: `fb_image_downloader/fb_downloader/browser_session/` (Chrome cache, metrics, SQLite DBs).
- Zip archives present alongside extracted contents (`.zip` files totaling $>1.5\text{ GB}$).
- Empty/stale folder: `02_lx_pt_patrol_cv/`.

---

## 2. Identified Bottlenecks, Broken Imports & Structural Debt

1. **Siloed & Fragmented Architecture**:
   - The repository has operated as 5 separate disconnected projects. There is no top-level importable Python package `src/`.
   - A developer or client cannot do `from src.pipeline import LPRPipeline` or `from src.detector import PlateDetector`.
2. **Missing Decoupled Inference Pipeline**:
   - Detection, OCR, character segmentation, stream reading, and regex formatting are entangled in monolithic scripts rather than clean, single-responsibility modules.
3. **Hardcoded Configurations**:
   - Video file paths, RTSP endpoints, model checkpoints, confidence thresholds, and crop margins are hardcoded in CLI arguments and script bodies rather than reading from central YAML configuration files.
4. **Offline Edge Model Deployment Gap**:
   - Models are dynamically downloaded from Hugging Face or Ultralytics on first run. An edge client running in an air-gapped or low-bandwidth deployment will fail without a localized `models/` registry.
5. **Virtual Environment Bloat**:
   - 6 redundant virtual environments consume disk space and contain machine-specific absolute path links in `pyvenv.cfg`.
6. **Commercial Delivery Debt**:
   - No standardized `main.py` CLI supporting Stream, Video, Image, and Benchmark modes.
   - Missing unit tests (`tests/`) and sample test media fixtures.
   - Missing client-facing `README.md` with benchmark numbers and 1-minute quickstart.

---

## 3. Proposed Professional Package Architecture

The proposed structure unifies all sub-projects into a clean, decoupled, commercial-grade package:

```
lpr/
├── configs/                            # Centralized YAML Configurations
│   ├── config.default.yaml             # Detection thresholds, OCR model names, crop padding
│   └── camera_streams.yaml             # RTSP camera feed URLs, FPS limits, reconnect intervals
│
├── models/                             # Model Weights & Runtime Registries
│   ├── README.md                       # Download links, SHA256 checksums, ONNX export guide
│   └── (local weights: .onnx / .pt)   # Excluded from git via .gitignore
│
├── scripts/                            # Operational Launchers & Environment Setup
│   ├── setup_env.bat                   # 1-Click Windows Python 3.10 venv setup
│   ├── setup_env.sh                    # 1-Click Linux venv setup
│   ├── run_live_stream.bat             # Production RTSP streaming launcher
│   ├── run_video.bat                   # Local video file processor
│   ├── run_benchmark.bat               # High-throughput CPU/GPU benchmark runner
│   └── check_environment.py            # Diagnostic tool (Python, CUDA, VRAM, imports)
│
├── src/                                # Clean, Importable Production Python Package
│   ├── __init__.py                     # Package entry and version metadata
│   ├── capture/                        # Video & RTSP Stream Handling
│   │   ├── __init__.py
│   │   ├── stream.py                   # Multi-threaded RTSP capture with frame-dropping queue
│   │   └── video_reader.py             # File and batch image iterator
│   ├── detector/                       # Vehicle & License Plate Localization
│   │   ├── __init__.py
│   │   ├── base_detector.py            # Abstract Base Class for detectors
│   │   ├── yolo_detector.py            # Ultralytics YOLO-World / YOLOv8 plate detector
│   │   └── onnx_detector.py            # Lightweight CPU-optimized ONNX Runtime detector
│   ├── ocr/                            # Thai Character & Province Recognition
│   │   ├── __init__.py
│   │   ├── segmenter.py                # Hough deskewing, CLAHE, projection band split, CCA
│   │   ├── recognizer.py               # Character classifier & province recognizer (ONNX / VLM)
│   │   └── font_engine.py              # DLT font geometry analyzer & NotoSansThai renderer
│   ├── pipeline/                       # End-to-End Pipeline Orchestration
│   │   ├── __init__.py
│   │   ├── lpr_pipeline.py             # Frame -> Plate -> Crop -> OCR -> Formatter pipeline
│   │   └── types.py                    # Dataclasses: BoundingBox, PlateReading, LPRResult
│   └── utils/                          # Cross-Cutting Utilities
│       ├── __init__.py
│       ├── config_loader.py            # YAML configuration parser
│       ├── formatter.py                # Thai plate regex sanitizer (หมวดอักษร + ตัวเลข + จังหวัด)
│       ├── visualizer.py               # Bounding box HUD, labels, and footprint drawing
│       └── metrics.py                  # Latency, FPS, and throughput telemetry
│
├── tests/                              # Verification Suite & Sample Test Media
│   ├── __init__.py
│   ├── test_detector.py                # Unit test for plate detection
│   ├── test_ocr.py                     # Unit test for character segmentation & OCR
│   ├── test_formatter.py               # Unit test for Thai plate syntax validation
│   └── samples/                        # Extracted test images and sample video clip
│
├── data/                               # Organized Dataset & Storage (Optional / Symlinked)
│   ├── raw/                            # Ingested street captures & crawler downloads
│   ├── curated/                        # Filtered passed images
│   └── synthetic/                      # Generated 77-province augmented plates
│
├── .agentsignore                       # Exclusions for weights, logs, and datasets
├── .gitignore                          # Git exclusions for venv, caches, and media
├── requirements.txt                    # Pinned, conflict-free Python 3.10 dependencies
├── main.py                             # Unified CLI Entrypoint (Stream, Video, Image, Benchmark)
└── README.md                           # Commercial Client Deliverable Documentation
```

---

## 4. Implementation & Verification Summary (Completed)

All phases have been executed, tested, and verified:

### Phase 1: Audit & Inventory — COMPLETE
- Cataloged all 6 submodules, datasets, model files, and loose zip archives.
- Identified and resolved hardcoded paths, duplicated environments, and monolithic scripts.

### Phase 2: Professional Package Layout — COMPLETE
- Created clean standard monorepo: `configs/`, `models/`, `scripts/`, `src/`, `tests/`, `archive/`.
- Moved all root zip archives into `archive/` to keep root tidy.
- Populated `configs/config.default.yaml`, `configs/camera_streams.yaml`, and `models/README.md`.

### Phase 3: Environment & Dependency Standardization — COMPLETE
- Standardized `requirements.txt` into cleanly commented functional blocks (Inference, CV, Utilities) enforcing Python 3.10 compatibility.
- Implemented cross-platform one-click virtual environment setup scripts: `scripts/setup_env.bat` and `scripts/setup_env.sh`.
- Created production runtime launchers: `scripts/run_live_stream.bat`, `scripts/run_video.bat`, and `scripts/run_benchmark.bat`.

### Phase 4: Modular Code Refactoring — COMPLETE
- Implemented pure dataclasses (`BoundingBox`, `PlateReading`, `LPRResult`, `StreamMetrics`) in `src/pipeline/types.py`.
- Implemented non-blocking RTSP stream reader with automatic stale frame dropping in `src/capture/stream.py`.
- Implemented multi-backend detector abstraction (`BaseDetector`, `YOLOWorldDetector`, `ONNXDetector`) in `src/detector/`.
- Implemented character segmentation (`segmenter.py`), font morphology engine (`font_engine.py`), and Thai plate recognizer (`recognizer.py`) in `src/ocr/`.
- Implemented official 77-province Thai DLT regex sanitizer and fuzzy matcher in `src/utils/formatter.py`.
- Implemented HUD visualizer (`visualizer.py`) and high-precision telemetry metrics (`metrics.py`) in `src/utils/`.
- Implemented unified pipeline orchestrator with IoU + EMA temporal centroid smoothing in `src/pipeline/lpr_pipeline.py`.
- Implemented production CLI entrypoint `main.py` supporting `stream`, `video`, `image`, and `benchmark` modes.

### Phase 5: Verification, Benchmarks & Delivery — COMPLETE
- **Unit Test Suite**: 10 unit tests across `test_formatter.py`, `test_detector.py`, `test_ocr.py`, and `test_pipeline.py`.
  - Result: `Ran 10 tests in 111.739s - OK (100% Pass)`.
- **Benchmark Performance**: Tested on 1080p authentic Thai dashcam video (`tests/samples/sample.mp4`):
  - Result: **22.40 FPS**, **44.63 ms** average latency per frame.
- **Client Documentation**: Delivered commercial-grade [README.md](file:///d:/Projects-anti/lpr/README.md) with architecture diagrams, quickstart guides, CLI reference, and Thai DLT specification.

---

## 5. Phase 6: 4-Pillar Monorepo Architecture Restructuring (Completed)

Per user request, the workspace was consolidated into exactly 4 functional pillars:

1. **`pipeline/`**: Dedicated self-contained inference pipeline package (`src/`, `configs/`, `models/`, `scripts/`, `tests/`, `main.py`, `.venv/`, `requirements.txt`).
2. **`dataset/`**: Automated dataset harvesting (`downloaders/`), aspect ratio & corruption filtering (`filtering/`), Qwen2-VL automated classification & labeling (`labeling/`), and verified plate storage (`storage/`).
3. **`training_detection/`**: YOLOv8/v11 license plate detection training engine (`train.py`, `prepare_dataset.py`, `export_onnx.py`, `evaluate.py`, `datasets/`).
4. **`training_ocr/`**: Thai character recognition suite with 77-province synthetic plate augmentation (`synthetic_data/`), character morphology extraction (`character_bank/`), and CRNN + CTC model training (`models/`).
5. **`archive/`**: Preserved storage for legacy prototypes (`01_thai_lpr_cpu_edge/`, `selen-lpr_old/`, `zero_shot_license_plate_detection/`, and old backup zips).

---

## 6. Phase 7: Pure PyTorch `selen-lpr`, Full Verification & Video Delivery (Completed)

### A. Pure PyTorch Architecture (Zero External Package Lock-in)
- **Zero EasyOCR & Zero Ultralytics**: Re-implemented complete inference pipeline purely using standard `torch`, `torchvision`, `cv2`, `numpy`, and `PIL`.
- **Stand-alone Models**:
  - `selen-lpr/models/detector/plate_detector.torchscript` (10.66 MB) — Native TorchScript detector with pure `torchvision.ops.nms`.
  - `selen-lpr/models/detector/plate_detector.onnx` (10.67 MB) — Edge-optimized ONNX runtime.
  - `selen-lpr/models/ocr/thai_plate_attn_ocr.pt` (55.25 MB) — Pure PyTorch GRCNN + BiLSTM + Attention Decoder (187 classes).
- **IP Protection & Tool Isolation**: Reusable transfer scripts isolated strictly in `selen-lpr/tools/` (`transfer_ocr_weights.py`, `export_detector.py`, `render_lpr_video_clip.py`).

### B. Unit Test Suite (100% OK — 17/17 Passing)
- **`selen-lpr/tests/`**: 7/7 tests passed in 1.068s:
  - `test_detection.py` (Pure PyTorch NMS & bounding box scaling)
  - `test_classification.py` (DLT category classification across all 8 vehicle types + Lao/English cross-border)
  - `test_ocr_model.py` (Standalone forward pass, vocabulary encoding/decoding)
  - `test_pipeline_integration.py` (End-to-end adaptive pipeline integration)
- **`pipeline/tests/`**: 10/10 tests passed in 5.217s:
  - Formatter, detector, OCR, and end-to-end streaming orchestrator tests.
- **Combined**: 17 unit tests, 100% passing.

### C. Video Passage Extraction & Annotation Deliverables
- **Footage Extraction**: Extracted 9 test recordings from `archive/LPR-20261012T021756Z-1-001.zip` to `dataset/storage/videos/LPR/`.
- **Clipped MP4 Outputs**:
  - **`runs/car_passage_lpr_annotated.mp4`** (5.58 MB) — High-impact 2-second passing vehicle clip (Frames 225–285, White Honda HR-V passing over speed bump).
  - **`runs/full_passage_lpr_annotated.mp4`** (26.9 MB) — Complete 10-second multi-vehicle passage reel (Frames 0–300, 38 plate detections).
- **On-Video Telemetry & HUD Typography**:
  - Rendered with dual-language TrueType font (`tahomabd.ttf` + `NotoSansThai-Bold.ttf`) eliminating missing glyphs.
  - Printed on every frame:
    1. Plate Number (`d3 + d4` in bold yellow e.g. `9ฎ 90`, `บฎ 0`, `ฃถ 202`)
    2. Province Name in Thai (`province` e.g. `กาญจนบุรี`, `ชัยนาท`, `ภูเก็ต`)
    3. Official DLT Vehicle Type (`cat_desc` e.g. `Private Cars`, `Private Pickups`, `Motorcycles`)
    4. Normalized Scale Coordinates: $(s\_x, s\_y, s\_w, s\_h)$
    5. Detection Confidence percentage (`Conf: 70.7%`)
    6. System Status Bar: Frame number, Timestamp, Real-time status badge



