# Agent Context Index: Thai LPR Edge

## 1. Architecture & Tech Stack
- **Runtime:** Python >= 3.10 (CPU Edge / Workstation), PyTorch >= 2.0, Torchvision >= 0.15
- **Vision & ML:** Ultralytics (YOLOv8s, YOLOv8-Worldv2, YOLOv11n), OpenCV (>= 4.8), Deep Text Rec (ResNet+BiLSTM+Attn)
- **UI & Ingestion:** PyQt6 (>= 6.5), FFmpeg/FFprobe subprocessing
- **Configs & Env:** `pyproject.toml`, `.env` / `.env.example`, `tools/rtsp_collector/config.yaml`
- **Entry Points:** `run.bat` (CLI launcher), `run_tests.bat` (test suite), `tools/detect_cam_videos.py` (CCTV inference), `tools/owod/main.py` (OWOD Studio GUI), `tools/DataManipulator/main.py` (Annotation GUI), `tools/rtsp_collector/cli.py` (RTSP capture)

## 2. Directory & Responsibility Map
- `annotations/`: Tracked review sessions (`plates_session.json`) and exports (`approved_plates.json`/`.csv`).
- `models/`: Model catalog (`MODEL_CATALOG.md`) and checkpoints (`detector/`, `reader/`). No raw weights committed.
- `tools/detect_cam_videos.py`: Production CCTV video inference, DLT plate mapping, tracking cache (`dataset/processed_videos_tracker.json`).
- `tools/DataManipulator/`: PyQt6 dataset curation tool; `app/` houses UI, ISO lookups, and session serialization.
- `tools/owod/core/`: Domain detection logic; `owod_engine.py` (YOLO/OWOD), `model_surgery.py` (prune/freeze), `quality_auditor.py` (ISO 5259 audit), `dataset_generator.py` (YOLO split).
- `tools/owod/ui/`: PyQt6 tabs for OWOD labeling, model surgery, dataset building, and fine-tuning.
- `tools/rtsp_collector/`: RTSP capture daemon; `collector.py` (stream worker), `validator.py` (integrity), `manifest.py` (ISO 8601 audit), `security.py` (ISO 27001 masking).
- `tests/`: ISO/IEC/IEEE 29119 compliance test suite (`test_iso_compliance.py`).
- `dataset/` & `runs/`: Git-ignored local video records, extracted crops, and training runs.
- `archive/` & `batch_image_filter_suite/`: Deprecated legacy code; exclude from active workflows.

## 3. Critical Interfaces & Data Boundaries
- **Plate Filename Schema:** `{name}_{plate_type}_{d3}_{d4}_{province}.ext` (5 components; parsed in `DataManipulator/app/session_manager.py`).
- **DLT Classes (0-3):** `[car_plate, motorcycle_plate, truck_plate, bus_plate]` mapped from COCO vehicle IDs `{2:0, 3:1, 7:2, 5:3}` in `detect_cam_videos.py` & `build_dlt_lpr_dataset.py`.
- **YOLO BBox Format:** Normalized `(class_id, xc, yc, w, h)` with $[0.0, 1.0]$ bounds; audited in `quality_auditor.py`.
- **Core Domain Logic (Inspect/Modify):**
  - Detection / Pipeline: `tools/detect_cam_videos.py`, `tools/owod/core/owod_engine.py`
  - Model Surgery / Pruning: `tools/owod/core/model_surgery.py`
  - Stream Ingestion & Watchdog: `tools/rtsp_collector/collector.py`, `tools/rtsp_collector/validator.py`
  - Standards & Normalization: `tools/DataManipulator/app/provinces.py`, `tools/owod/core/quality_auditor.py`
- **Boilerplate / Helpers (Avoid Crawling):** UI layouts (`tools/*/ui/*`, `tools/DataManipulator/app/*_view.py`), mock data scripts (`create_sample_data.py`), visualizers (`visualize_*.py`), one-off test scripts (`inspect_*.py`).

## 4. Invariants & Conventions
- **Text & Geo Standards:** All Thai text must be Unicode NFC (`unicodedata.normalize('NFC', text)`). Provinces must map to ISO 3166-2:TH (`TH-10` to `TH-96`) via `provinces.py`. Timestamps must be ISO 8601 UTC.
- **Model Governance:** NEVER commit binary weights (`.pt`, `.onnx`, `.engine`, `.pth`). Document changes in `models/MODEL_CATALOG.md`. Deterministic random seed: `42`.
- **Security:** Sanitize user paths and mask RTSP credentials (`admin:***@host`) using `tools/rtsp_collector/security.py`.
- **State Writes:** All runtime metadata and session JSON files must use atomic write-and-replace (`.tmp` -> replace) to prevent corruption.
