# LPR Tools Directory

This folder contains the core dataset manipulation, ingestion, and labeling suites:

1. **[`DataManipulator`](DataManipulator/)**:
   - Desktop GUI for Thai License Plate (LPR) Dataset Annotation, Review, Subfolder Grouping, and Filename Renaming.
   - Run via `tools/DataManipulator/run.bat` or `python tools/DataManipulator/main.py`.

2. **[`owod`](owod/)**:
   - Open-World Object Detection (OWOD) License Plate Studio & YOLO Model Pruning Tool (zero-shot labeling, model surgery, layer freezing, structured/unstructured pruning, and fine-tuning).
   - Contains `build_dlt_lpr_dataset.py` for automated DLT plate dataset generation.
   - Run via `tools/owod/run.bat` or `python tools/owod/main.py` (or root `run_owod_studio.bat`).

3. **[`rtsp_collector`](rtsp_collector/)**:
   - ISO-compliant continuous 24/7 RTSP stream capture, integrity validation, and storage partitioning.
   - Run via `tools/rtsp_collector/run.bat` or `python tools/rtsp_collector/cli.py`.

4. **[`detect_cam_videos.py`](detect_cam_videos.py)**:
   - Batch CCTV video detection runner with incremental JSON tracking and vehicle-plate matching.
   - Run via root `run_detection.bat` or `python tools/detect_cam_videos.py`.
