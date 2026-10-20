# Thai License Plate Recognition (Thai LPR) — Monorepo

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Inference](https://img.shields.io/badge/Inference-ONNX%20%7C%20PyTorch%20%7C%20YOLO--World-green.svg)](https://github.com/ultralytics/ultralytics)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20Edge-orange.svg)]()
[![Thai DLT](https://img.shields.io/badge/Standard-77%20Provinces%20Regex-purple.svg)]()

Production-ready, commercial-grade Computer Vision monorepo for Thai License Plate Recognition (ALPR/ANPR). Decoupled into four core functional pillars: real-time edge inference pipeline, automated dataset harvesting & VLM labeling, license plate detection training, and OCR character recognition training.

---

## 1. Monorepo Architecture Overview

The repository is organized into four clean, isolated directories:

```
lpr/
├── pipeline/                       # Pillar 1: Production Edge Inference Pipeline
│   ├── configs/                    # Runtime YAML configs (detection, tracking, RTSP)
│   ├── models/                     # Local model weights & ONNX runtimes
│   ├── scripts/                    # Windows & Linux execution launchers
│   ├── src/                        # Modular Python package (capture, detector, ocr, pipeline, utils)
│   ├── tests/                      # Unit test suite & sample test media
│   ├── main.py                     # Unified CLI entrypoint (Stream, Video, Image, Benchmark)
│   ├── requirements.txt            # Pinned Python 3.10 dependencies
│   └── README.md                   # Complete pipeline documentation
│
├── dataset/                        # Pillar 2: Dataset Acquisition, Filtering & Labeling
│   ├── downloaders/                # Harvester for PlatesMania, Roboflow & Facebook crawler
│   ├── filtering/                  # High-throughput aspect ratio & quality filter suite
│   ├── labeling/                   # Qwen2-VL Vision-Language plate classifier & labeler
│   ├── storage/                    # Verified plates & raw dataset pools
│   ├── run_download.bat            # 1-Click downloader launcher
│   ├── run_filter.bat              # 1-Click filter launcher
│   ├── run_label_sorter.bat        # 1-Click VLM sorter launcher
│   └── README.md                   # Dataset engineering guide
│
├── training_detection/             # Pillar 3: Plate Detection Model Training
│   ├── datasets/                   # Raw YOLOv8/v11 dataset archives (.zip)
│   ├── configs/                    # Training hyperparameters (AdamW, augmentations)
│   ├── prepare_dataset.py          # Dataset unpacker & data.yaml generator
│   ├── train.py                    # Ultralytics detector training engine
│   ├── export_onnx.py              # PyTorch .pt -> ONNX converter with dynamic batching
│   ├── evaluate.py                 # mAP50, mAP50-95 & latency evaluator
│   ├── train_detector.bat          # 1-Click interactive training launcher
│   └── README.md                   # Detection training guide
│
├── training_ocr/                   # Pillar 4: Thai OCR Model Training & Synthetic Data
│   ├── synthetic_data/             # 77-Province synthetic plate generator & augmentor
│   ├── character_bank/             # Hough deskewing & CCA character glyph segmenter
│   ├── models/                     # PyTorch CRNN (CNN + BiLSTM + CTC) OCR training engine
│   ├── generate_synthetic.bat      # 1-Click synthetic plate generator
│   ├── train_ocr.bat               # 1-Click interactive OCR trainer
│   └── README.md                   # OCR training guide
│
├── archive/                        # Preserved legacy prototypes & backups
│   ├── 01_thai_lpr_cpu_edge/       # Archived CPU edge prototype
│   ├── selen-lpr/                  # Archived stub
│   ├── zero_shot_license_plate_detection/ # Archived zero-shot experiments
│   └── root_scripts/               # Archived root launchers
│
├── run_pipeline.bat                # 1-Click root launcher into pipeline/
├── .agentsignore
├── .gitignore
├── LPR_PIPELINE_STATUS.md          # Architecture audit and migration status
└── README.md                       # This document
```

---

## 2. Pillar 1: Production Inference Pipeline (`pipeline/`)

The inference pipeline takes live RTSP video feeds, dashcam recordings, or single images and extracts normalized Thai license plate readings with sub-50ms latency.

### Quick Run:
```powershell
# Run benchmark from root
.\run_pipeline.bat

# Or run directly from pipeline/
cd pipeline
python main.py --mode video --source tests/samples/sample.mp4
python main.py --mode benchmark --max-frames 50
python main.py --mode stream --source "rtsp://camera_ip:554/stream"
```

For complete CLI options and configuration settings, see **[pipeline/README.md](file:///d:/Projects-anti/lpr/pipeline/README.md)**.

---

## 3. Pillar 2: Dataset Handling & Labeling (`dataset/`)

The dataset engineering suite automates the lifecycle from web ingestion to verified training crops:
1. **Download**: Pulls plate photos from PlatesMania Thailand, Roboflow ALPR APIs, or automated vehicle crawlers (`downloaders/`).
2. **Filter**: Discards corrupted or non-plate aspect ratios ($1.4 \le \text{AR} \le 4.0$) via multi-threaded Pillow inspection (`filtering/`).
3. **VLM Label & Sort**: Classifies plates using Qwen2-VL into real plates, digital ad mockups, auction plates, or non-plates with OCR text extraction (`labeling/`).

### Quick Run:
```powershell
cd dataset
.\run_download.bat       # Download plates
.\run_filter.bat         # Aspect ratio filter
.\run_label_sorter.bat   # VLM classification & labeling
```

For detailed usage, see **[dataset/README.md](file:///d:/Projects-anti/lpr/dataset/README.md)**.

---

## 4. Pillar 3: Detection Model Training (`training_detection/`)

Trains custom YOLOv8 / YOLOv11 license plate detectors and automatically exports optimized `.onnx` models into `pipeline/models/`:

### Quick Run:
```powershell
cd training_detection
.\train_detector.bat
```
- Option 1: Unpack and prepare datasets from `datasets/*.zip`.
- Option 2: Train YOLOv8s plate detector (100 epochs on GPU).
- Option 3: Export best weights to ONNX format with dynamic batching and opset 17.

For detailed usage, see **[training_detection/README.md](file:///d:/Projects-anti/lpr/training_detection/README.md)**.

---

## 5. Pillar 4: OCR Model Training (`training_ocr/`)

End-to-end character recognition training covering all 77 Thai provinces:
1. **Synthetic Plate Augmentation**: Generates realistic plates with TrueType fonts, perspective warps, motion blur, and dirt patterns (`synthetic_data/`).
2. **Character Segmentation**: Hough transform plate deskewing, CLAHE, projection splitting, and Connected Component Analysis (CCA) glyph extraction (`character_bank/`).
3. **CRNN + CTC Training**: PyTorch CNN + Bidirectional LSTM model with CTC Loss, decoding Thai consonants, digits, and provinces (`models/`).

### Quick Run:
```powershell
cd training_ocr
.\generate_synthetic.bat    # Generate 77-province synthetic plates
.\train_ocr.bat             # Train CRNN OCR model
```

For detailed usage, see **[training_ocr/README.md](file:///d:/Projects-anti/lpr/training_ocr/README.md)**.

---

## 6. System Prerequisites

- **OS**: Windows 10/11 (64-bit) or Ubuntu 22.04 LTS
- **Python**: Python 3.10.x (64-bit)
- **Hardware**: NVIDIA RTX 3060/4060/4090 (CUDA 12.4) or Edge CPU (Intel Core i5/i7)
- **Dependencies**: Managed per pillar; master virtualenv located at `pipeline/.venv`.
