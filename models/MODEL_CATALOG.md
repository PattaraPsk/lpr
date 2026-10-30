# AI Model Catalog & Governance Lineage (ISO/IEC 42001:2023)

This catalog specifies model lineages, architectures, tasks, input dimensions, classes, and performance benchmarks for all models deployed in the Thai License Plate Recognition (LPR) & OWOD Studio pipeline.

---

## 🏛️ Active Production Models

### 1. Unified Vehicle Orientation & License Plate Detector
- **Model Key:** `vehicle-plate-detector-yolov11n`
- **Primary Checkpoint:** [`models/detector/vehicle_plate_yolov11n.pt`](file:///d:/Projects-anti/lpr/models/detector/vehicle_plate_yolov11n.pt)
- **Convenience Aliases:**
  - `models/detector/vehicle_plate_detector.pt`
  - `vehicle_plate_detector.pt` (root hardlink)
- **Architecture:** YOLOv11n (Nano Detection Head)
- **Input Resolution:** `640x640`
- **Weight Size:** ~5.5 MB (Edge-optimized)
- **Class Map:**
  - `0`: `Front` (Vehicle approaching / front orientation)
  - `1`: `LicensePlate` (Thai License Plate bounding box)
  - `2`: `Rear` (Vehicle departing / rear orientation)
- **Training Provenance:**
  - Training Run: `train2` (515 epochs completed, early stop patience 100)
  - Dataset: `Vehicle-Plate-LPR.v12c.yolov8`
- **Evaluation Metrics (Validation Set):**
  - **mAP@50:** **99.36%** (0.99361)
  - **mAP@50-95:** **90.40%** (0.90402)
  - **Precision:** **98.43%** (0.98428)
  - **Recall:** **99.23%** (0.99228)

---

### 2. Thai License Plate Text Reader (OCR)
- **Model Key:** `thai-lpr-reader` *(Active Production Checkpoint)*
- **Primary Checkpoint:** [`models/reader/thai_plate_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/thai_plate_reader.pth)
- **Training Artifact:** [`runs/train/reader/best_reader.pth`](file:///d:/Projects-anti/lpr/runs/train/reader/best_reader.pth)
- **Architecture:** Deep Text Recognition Benchmark (ResNet/CNN Feature Extraction + BiLSTM Sequence Modeling + Attention Decoder, 187 tokens)
- **Weight Size:** ~57.9 MB (SHA256: `CC57C17F8CA6AC3DF21DF634539F4FEEAD7C5795B2FC1CDECF1BB4D41D35560C`)
- **Task:** Thai alphanumeric character segmentation & province text transcription
- **Input:** Rectified license plate crops (normalized aspect ratio H=64, W=600, NormalizePAD)
- **Training Dataset (ISO 42001 Traceability):**
  - Anchor Verified Ground Truth: `dataset/ready/202610261140` (1,183 plates)
  - Mobile Verified Ground Truth: `dataset/snapshot/mobile` (69 plates)
  - CCTV & Mobile Multi-Condition Synthetic: 17,248 + 1,500 plates (3D perspective warp, blur, noise, dirty layers, graphic auction backgrounds)
  - Total Training Corpus: **21,120 plates** (30 epochs, AdamW + AMP Mixed Precision FP16 on RTX 4090)
- **Evaluation Benchmarks (vs. Baseline):**
  - **Real Verified Plates Exact Match:** **100.0%** (15/15) vs. Baseline 80.0% (12/15)
  - **CCTV Edge Synthetic Benchmark:** **100.0%** (20/20) vs. Baseline 0.0% (0/20)
  - Resolves subtle character confusions (`1กข` vs `1กช`, `1กจ` vs `2กจ`, `1กศ` vs `1ศศ`) and handles 3-tier motorcycle layouts and heavy commercial trucks seamlessly.

---

### 3. Public EasyOCR Thai Foundation Reader (Open-Source Licensed)
- **Model Key:** easyocr-thai-reader *(Open-Source Public Lineage)*
- **Primary Checkpoint:** [models/reader/easyocr_thai/best_reader.pth](file:///d:/Projects-anti/lpr/models/reader/easyocr_thai/best_reader.pth)
- **Base Provenance:** Official JaidedAI EasyOCR Thai Pre-trained Weights (	hai.pth, Apache 2.0 / MIT)
- **Architecture:** ResNet Feature Extraction + 2-layer Bidirectional LSTM + CTC Linear Head (177 Thai glyph classes)
- **Parameters:** 53,809,297 (~205 MB float32, ~54 MB INT8 quantized)
- **Inference Speed:** > 1,500 FPS on RTX 4090 GPU / ~50 ms on CPU Edge
- **Training Corpus (ISO 42001 Traceability):**
  - Manually Labeled Ground Truth: dataset/labeled/manually_labeled/plates (6,168 plates)
  - Synthetic CCTV: dataset/dataset-cctv-synthetic/plates (18,437 plates)
  - Synthetic Mobile: dataset/dataset-mobile-synthetic/plates (1,500 plates)
  - Verified Benchmark: dataset/ready/202610261140 & mobile (1,249 plates)
  - Verified Staged/Curated: dataset/verified/ (2,292 plates)
  - Total Training Corpus: **29,720 balanced plates** (max 60 per prefix across 8,456 distinct prefixes)
- **Evaluation Performance:**
  - Training CTC loss descended monotonically from **9.32** to **1.15** in Epoch 1.
  - Successfully recognizes Thai consonants (ฒ, ต, ว, น, จ), digits, and provincial names (กรุงเทพมหานคร).

---

### 4. SelenLPR Multi-Task Attention Reader (selen-lpr-0.0.1 Generation)
- **Model Key:** `selen-lpr-0.0.1` *(Next-Gen Edge Architecture)*
- **Primary Checkpoint:** [`models/reader/selen-lpr-0.0.1/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/selen-lpr-0.0.1/best_reader.pth)
- **Training Artifact:** [`runs/train/selen_lpr_0.0.1/best_reader.pth`](file:///d:/Projects-anti/lpr/runs/train/selen_lpr_0.0.1/best_reader.pth)
- **Architecture:** `SelenLPRv2Edge`: Pretrained `MobileNetV3-Small` Backbone (ImageNet-1K, tuned horizontal strides H'=8, W'=48) + Two-Band Spatial Feature Slicing (Upper 62.5% for registration characters, Lower 37.5% for province) + Multi-Head Self-Attention Transformer Sequence Encoder (256-dim, 4 heads) + Multi-Head Slot Character Decoders (8 slot queries: 4 for `d3`, 4 for `d4`) + Direct Lower Band Province Classifier (79 classes) + Global Feature Type Classifier (11 classes)
- **Parameters:** 3,818,937 (~14.57 MB float32, ~7.28 MB FP16) - **75% smaller** than legacy baseline (57.9 MB)
- **Cryptographic Hash (SHA-256):** `7C07C588491569D0CD59073255CCFFCFC18372FD348B47CBBF2F17826C414A30`
- **Inference Speed:** > 2,000 FPS on RTX 4090 GPU / ~3-5 ms on CPU Edge
- **Target Outputs (End-to-End Simultaneous):**
  1. `plate_type`: 11 DLT vehicle plate categories (`WhiteBlackCar`, `WhiteGreen`, `YellowBlack`, `WhiteBlackMotor`, etc.)
  2. `d3`: Prefix Thai consonants and leading digits (1..4 characters)
  3. `d4`: Registration numbers (1..4 digits)
  4. `province`: 78 Thailand administrative subdivisions + Betong
- **Loss Function & Training Protocol:** `SelenMultiTaskLoss` with dual modes:
  - Mode 1: Clamped Kendall homoscedastic uncertainty (`s_d3`, `s_d4` clamped in `[-2.5, 1.2]`, task scaling `d3=2.5`, `d4=2.0`, label smoothing `0.02`).
  - Mode 2: Static weighted sum (`1.0*type + 1.0*prov + 3.0*d3 + 3.0*d4 + 0.10*consistency`).
  - PyTorch AMP mixed precision (`torch.amp.autocast("cuda")` + `GradScaler`) on RTX 4090.
- **Training Corpus (33,400 clean plates, 8,429 distinct prefixes):**
  - Manually Labeled Ground Truth: `dataset/labeled/manually_labeled/plates` (6,168 plates)
  - Synthetic CCTV: `dataset/dataset-cctv-synthetic/plates` (18,437 plates)
  - Synthetic Mobile: `dataset/dataset-mobile-synthetic/plates` (1,500 plates)
  - Verified Benchmark: `dataset/ready/202610261140` & `mobile` (1,249 plates)
  - Verified Staged/Curated: `dataset/verified/` (2,292 plates)
- **Continuation Run Schedule (500 Epochs):**
  - Resumes from epoch 51 using checkpoint `models/reader/selen-lpr-0.0.1/best_reader.pth`.
  - Checkpoint bundles saved to `runs/train/selen_lpr_0.0.1/last_checkpoint.pth` and `best_checkpoint.pth` for full state recovery.
- **Evaluation Benchmark (50-Epoch Baseline Checkpoint):**
  - **Plate Type Accuracy:** **97.21%** (PASSED ALPR standard >= 90%)
  - **Province Classification Accuracy:** **95.44%** (PASSED ALPR standard >= 85%)
  - **Exact 4-Way Match:** Initial convergence lifting to 8.54% at epoch 44, targeted for >90% across 500 epochs under stabilized sequence gradient flow.

---

### 5. Context Vehicle Detector (DLT Classification)
- **Model Key:** `yolov8s-vehicle`
- **Primary Checkpoint:** [`models/detector/yolov8s.pt`](file:///d:/Projects-anti/lpr/models/detector/yolov8s.pt) (also linked at root)
- **Architecture:** YOLOv8s
- **Weight Size:** ~22.5 MB
- **Classes Filtered:** COCO Vehicle classes:
  - `2`: `car` -> DLT Category `0` (`car_plate`)
  - `3`: `motorcycle` -> DLT Category `1` (`motorcycle_plate`)
  - `7`: `truck` -> DLT Category `2` (`truck_plate`)
  - `5`: `bus` -> DLT Category `3` (`bus_plate`)
- **Task:** Provides vehicle context bounding box to match and associate detected license plates into Thailand DLT categories.

---

### 6. Zero-Shot Open-Vocabulary Detector (OWOD Studio)
- **Model Key:** `yolov8s-worldv2`
- **Primary Checkpoint:** [`models/detector/yolov8s-worldv2.pt`](file:///d:/Projects-anti/lpr/models/detector/yolov8s-worldv2.pt) (also linked at root)
- **Architecture:** YOLO-World v2 Small
- **Weight Size:** ~26.6 MB
- **Task:** Open-world zero-shot proposal generation via dynamic text prompt (`"license plate"`).

---

### 7. Edge Baseline Detector
- **Model Key:** `yolov8n-base`
- **Primary Checkpoint:** [`models/detector/yolov8n.pt`](file:///d:/Projects-anti/lpr/models/detector/yolov8n.pt) (also linked at root)
- **Architecture:** YOLOv8n
- **Weight Size:** ~6.5 MB
- **Task:** Minimum latency baseline for CPU-constrained edge hardware.

---

## 🗄️ Model Directory Layout

```text
models/
├── detector/
│   ├── vehicle_plate_yolov11n.pt        # Vehicle orientation & license plate detector
│   ├── vehicle_plate_detector.pt        # Standard detector link
│   ├── yolov8s.pt                       # COCO vehicle detector
│   ├── yolov8s-worldv2.pt               # YOLO-World v2 zero-shot detector
│   └── yolov8n.pt                       # Lightweight edge base
│
├── reader/
│   ├── easyocr_thai/
│   │   └── best_reader.pth              # Public EasyOCR fine-tuned foundation reader
│   ├── thai_plate_reader.pth            # Thai Character & Province OCR
│   └── selen-lpr-0.0.1/
│       └── best_reader.pth              # SelenLPR multi-task edge model
│
└── MODEL_CATALOG.md                     # This governance record
```

*(Note: `model/` is an NTFS junction pointing to `models/` for 100% backward compatibility with singular/plural path references).*
