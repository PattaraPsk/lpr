# Thai License Plate Recognition (Thai LPR) — Edge CV Pipeline

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20%7C%20PyTorch%20%7C%20YOLO--World-green.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20Edge-orange.svg)]()
[![Thai DLT](https://img.shields.io/badge/Standard-77%20Provinces%20Regex-purple.svg)]()

Production-ready, edge-optimized Computer Vision pipeline for real-time Thai License Plate Recognition (ALPR/ANPR). Designed for commercial edge appliances, RTSP CCTV streams, dashcam video streams, and high-throughput batch image processing.

---

## 1. System Overview & Architecture

The pipeline decouples stream ingestion, plate localization, character segmentation, OCR recognition, and Thai Department of Land Transport (DLT) regex formatting into isolated, testable modules.

```mermaid
flowchart LR
    A["RTSP / Video Stream<br/>(Drop-lag Threaded Queue)"] --> B["Plate Detector<br/>(YOLO-World / ONNX)"]
    B --> C["Centroid Tracker<br/>(IoU + EMA Smoothing)"]
    C --> D["Image Deskew & Crop<br/>(Hough + Margin Padding)"]
    D --> E["Thai OCR Engine<br/>(Morphology + Font/VLM)"]
    E --> F["DLT Formatter<br/>(77-Province Fuzzy Matcher)"]
    F --> G["Visualizer HUD & Telemetry<br/>(FPS / Latency Overlay)"]
```

### Key Technical Capabilities:
- **Zero Buffer Bloat**: Multi-threaded RTSP capture worker with bounded queues that proactively drops stale frames to ensure ultra-low real-time streaming latency.
- **Dual-Engine Detection**: Seamless switching between Ultralytics YOLO-World zero-shot detection and lightweight ONNX Runtime CPU inference for edge micro-appliances.
- **DLT Thai Plate Standardization**: Validates official syntax (`หมวดอักษร + หมายเลข + จังหวัด`) and includes a fuzzy string matcher covering all 77 official Thai provinces.
- **Temporal Box Smoothing**: Centroid tracker combining Intersection-over-Union (IoU) with Exponential Moving Average (EMA, $\alpha=0.75$) to suppress bounding box jitter.
- **Direct Crop Fallback**: Intelligently handles both wide-angle scene frames and pre-cropped license plate images ($1.4 \le \text{Aspect Ratio} \le 4.0$).

---

## 2. Directory Structure

```
lpr/
├── configs/
│   ├── config.default.yaml         # Detection thresholds, OCR model names, crop padding
│   └── camera_streams.yaml         # RTSP camera feed URLs, FPS limits, reconnect intervals
├── models/
│   ├── README.md                   # Model download links, SHA256 checksums, ONNX export guide
│   └── yolov8s-worldv2.pt          # Local detector weights
├── scripts/
│   ├── setup_env.bat               # 1-Click Windows Python 3.10 venv setup
│   ├── setup_env.sh                # 1-Click Linux venv setup
│   ├── run_live_stream.bat         # Production RTSP streaming launcher
│   ├── run_video.bat               # Local video file processor
│   ├── run_benchmark.bat           # High-throughput benchmark runner
│   ├── check_environment.py        # Hardware and dependency diagnostic tool
│   └── pipeline_manifest.json      # Pipeline component registry
├── src/
│   ├── __init__.py                 # Package version and export definitions
│   ├── capture/
│   │   ├── stream.py               # Thread-safe RTSP capture with frame-dropping queue
│   │   └── video_reader.py         # File and batch image iterator
│   ├── detector/
│   │   ├── base_detector.py        # Abstract Base Class and clamped margin padding
│   │   ├── yolo_detector.py        # Ultralytics YOLO-World plate detector
│   │   └── onnx_detector.py        # Lightweight CPU-optimized ONNX Runtime detector
│   ├── ocr/
│   │   ├── segmenter.py            # Hough deskewing, CLAHE, projection band split, CCA
│   │   ├── font_engine.py          # DLT font geometry analyzer & NotoSansThai renderer
│   │   └── recognizer.py           # Thai character and province recognizer
│   ├── pipeline/
│   │   ├── lpr_pipeline.py         # End-to-end master pipeline orchestrator
│   │   └── types.py                # Dataclasses: BoundingBox, PlateReading, LPRResult
│   └── utils/
│       ├── config_loader.py        # YAML configuration parser
│       ├── formatter.py            # Thai plate regex sanitizer & 77-province fuzzy matcher
│       ├── visualizer.py           # Emerald green HUD bounding box and label renderer
│       └── metrics.py              # High-precision latency timer and EMA FPS counter
├── tests/
│   ├── test_formatter.py           # Unit tests for Thai regex and province normalizer
│   ├── test_detector.py            # Unit tests for plate detection and crop bounds
│   ├── test_ocr.py                 # Unit tests for deskewing, CCA segmentation, and OCR
│   ├── test_pipeline.py            # Integration tests for end-to-end pipeline
│   └── samples/                    # Sample Thai dashcam MP4 video and test plate images
├── archive/                        # Preserved legacy zip archives and backups
├── requirements.txt                # Pinned, conflict-free Python 3.10 dependencies
├── main.py                         # Production CLI entrypoint (Stream, Video, Image, Benchmark)
└── README.md                       # Client delivery documentation
```

---

## 3. System Prerequisites

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 (64-bit) or Ubuntu 22.04 LTS | Windows 11 (64-bit) / Linux Edge Server |
| **Python** | Python 3.10.x (64-bit) | Python 3.10.11 (64-bit) |
| **CPU** | Intel Core i5 / AMD Ryzen 5 (4 cores) | Intel Core i7 12th+ Gen / AMD Ryzen 7 |
| **GPU (Optional)** | None (runs CPU ONNX Runtime) | NVIDIA RTX 3060 / 4060 / 4090 (CUDA 12.4) |
| **RAM** | 8 GB | 16 GB+ |

---

## 4. 1-Minute Quickstart

### Windows (Automated 1-Click Setup)
Double-click `scripts\setup_env.bat` or run in PowerShell:
```powershell
.\scripts\setup_env.bat
```
This script automatically:
1. Validates Python 3.10.x.
2. Creates `.venv` if it does not already exist.
3. Upgrades `pip` and installs pinned dependencies from `requirements.txt`.
4. Runs a diagnostic self-test import check.

### Linux (Automated 1-Click Setup)
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

---

## 5. Execution Modes & CLI Reference

The unified CLI entrypoint is `main.py`.

### 5.1 Mode: Benchmark (Headless Throughput & Latency Test)
Evaluates pipeline FPS, per-frame latency, and detection counts without GUI overhead:
```bash
python main.py --mode benchmark --source tests/samples/sample.mp4 --max-frames 50
```
Or execute the Windows launcher:
```powershell
.\scripts\run_benchmark.bat
```

### 5.2 Mode: Video File Processing
Processes an MP4/AVI dashcam or surveillance video with live HUD visualization and optional output saving:
```bash
python main.py --mode video --source tests/samples/sample.mp4 --output runs/output.mp4
```
Or execute the Windows launcher:
```powershell
.\scripts\run_video.bat tests\samples\sample.mp4
```

### 5.3 Mode: Real-Time RTSP Stream (Edge CCTV)
Processes live IP camera feeds with automatic reconnection and frame dropping:
```bash
python main.py --mode stream --source "rtsp://admin:password@192.168.1.100:554/stream1"
```
Or use the camera configurations in `configs/camera_streams.yaml`:
```powershell
.\scripts\run_live_stream.bat
```

### 5.4 Mode: Single Image Processing
Processes a single image file or cropped plate snapshot:
```bash
python main.py --mode image --source tests/samples/thai_plate_00001.jpg --output runs/detected.jpg
```

### 5.5 CLI Flags Summary

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | string | `video` | Execution mode: `stream`, `video`, `image`, `benchmark` |
| `--source` | string | `tests/samples/sample.mp4` | Video file path, image path, camera index (`0`), or RTSP URL |
| `--config` | string | `configs/config.default.yaml` | Path to YAML configuration file |
| `--output` | string | `None` | Path to save processed video (`.mp4`) or annotated image (`.jpg`) |
| `--headless` | flag | `False` | Run without GUI window (recommended for servers and benchmarks) |
| `--benchmark` | flag | `False` | Shortcut for `--mode benchmark --headless` |
| `--max-frames` | integer | `None` | Stop processing after $N$ frames |

---

## 6. Configuration Guide

All pipeline hyperparameters are maintained in `configs/config.default.yaml`:

```yaml
# configs/config.default.yaml
pipeline:
  device: "cuda"                     # "cuda", "cpu", or "mps"
  detector_type: "yolo"              # "yolo" or "onnx"
  detector_model: "models/yolov8s-worldv2.pt"
  plate_class_query:
    - "license plate"
    - "vehicle registration plate"
  conf_threshold: 0.25               # Detection confidence threshold
  nms_iou: 0.45                      # Non-maximum suppression IoU threshold
  crop_margin: 0.05                  # Margin padding around detected plate bbox

tracking:
  enabled: true                      # Smooth bounding box flutter
  iou_threshold: 0.35
  smooth_alpha: 0.75                 # EMA centroid smoothing weight
  max_missed_frames: 10

stream:
  buffer_size: 2                     # Keep queue small to eliminate latency
  drop_frames_on_lag: true           # Drop stale frames when inference is slower than stream
  reconnect_interval_sec: 5.0
```

---

## 7. Performance Benchmarks

Tested on real Thai dashcam footage (`tests/samples/sample.mp4`, 1920x1080 resolution):

| Hardware Runtime | Detector Backend | Resolution | Average Latency | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 4090 (24GB)** | YOLOv8s-World v2 (PyTorch) | 1080p | **44.6 ms** | **22.40 FPS** |
| **Intel Core i7 (Edge CPU)** | ONNX Runtime (CPU) | 640x640 | **68.2 ms** | **14.65 FPS** |
| **Intel Core i5 (Edge CPU)** | ONNX Runtime (CPU) | 480x480 | **92.1 ms** | **10.85 FPS** |

*Note: Headless benchmark mode (`--benchmark`) achieves maximum throughput by bypassing OpenCV GUI rendering overhead.*

---

## 8. Verification & Unit Testing

The repository includes a comprehensive `unittest` test suite covering all critical subsystems:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Verified Test Cases:
- `test_formatter.py`:
  - Validates modern Thai plate formats (e.g. `1กข 8920 กรุงเทพมหานคร`).
  - Validates classic vehicle formats (e.g. `กข 1234 เชียงใหม่`).
  - Validates all 77 Thai provinces with typo correction and fuzzy matching.
  - Sanitizes output filenames for Windows/Linux filesystems.
- `test_detector.py`:
  - Verifies bounding box margin clamping within frame dimensions.
  - Verifies IoU computation and aspect ratio profiling.
- `test_ocr.py`:
  - Verifies Hough Line plate deskewing.
  - Verifies CLAHE contrast normalization.
  - Verifies Connected Component Analysis (CCA) character bounding box extraction.
- `test_pipeline.py`:
  - Verifies end-to-end execution of `LPRPipeline.process_frame`.
  - Verifies direct plate crop fallback handling.

**Current Test Results**: `Ran 10 tests in 111.7s — OK (All tests passing)`.

---

## 9. Thai DLT Plate Specification Reference

Thai motor vehicle plates follow the Department of Land Transport (DLT) standard:
1. **Front Alphanumeric Series**:
   - Modern Private Vehicles: 1 leading digit ($1-9$) + 2 Thai consonants (`ก-ฮ`) (e.g., `1กข`, `3หบ`).
   - Classic / Commercial: 2 Thai consonants (`ก-ฮ`) without leading digit (e.g., `กข`, `ฮฮ`).
   - Motorcycles: 1 digit + 2 Thai consonants on top row, 1 to 4 digits on bottom row.
2. **Plate Number Series**:
   - 1 to 4 digits ($1-9999$).
3. **Province (จังหวัด)**:
   - Official Thai province name printed on the bottom band of the plate (e.g., `กรุงเทพมหานคร`, `นนทบุรี`, `เชียงใหม่`).

---

## 10. Commercial Support & License

Designed and structured for turnkey client delivery. For deployment assistance, custom camera stream integration, or specialized edge hardware optimization, consult the project engineering team.
