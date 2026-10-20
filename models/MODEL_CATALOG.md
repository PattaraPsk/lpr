# AI Model Catalog & Governance Lineage (ISO/IEC 42001:2023)

This catalog specifies model lineages, architectures, tasks, input dimensions, classes, and performance benchmarks for all models deployed in the Thai License Plate Recognition (LPR) & OWOD Studio pipeline.

---

## 🏛️ Active Production Models

### 1. Unified Front/Rear Orientation & License Plate Detector
- **Model Key:** `front-rear-lpr-yolov11n`
- **Primary Checkpoint:** [`models/detector/Front-Rear-LPR.combinev1.yolov11n/train2/weights/best.pt`](file:///d:/Projects-anti/lpr/models/detector/Front-Rear-LPR.combinev1.yolov11n/train2/weights/best.pt)
- **Convenience Aliases:**
  - `models/detector/front_rear_lpr_yolov11n.pt`
  - `models/detector/rear_front_best.pt`
  - `rear_front_best.pt` (root hardlink)
- **Architecture:** YOLOv11n (Nano Detection Head)
- **Input Resolution:** `640x640`
- **Weight Size:** ~5.5 MB (Edge-optimized)
- **Class Map:**
  - `0`: `Front` (Vehicle approaching / front orientation)
  - `1`: `LicensePlate` (Thai License Plate bounding box)
  - `2`: `Rear` (Vehicle departing / rear orientation)
- **Training Provenance:**
  - Training Run: `train2` (515 epochs completed, early stop patience 100)
  - Dataset: `Front-Rear-LPR.v12c.yolov8`
- **Evaluation Metrics (Validation Set):**
  - **mAP@50:** **99.36%** (0.99361)
  - **mAP@50-95:** **90.40%** (0.90402)
  - **Precision:** **98.43%** (0.98428)
  - **Recall:** **99.23%** (0.99228)

---

### 2. Thai License Plate Text Reader (OCR)
- **Model Key:** `thai-lpr-reader-a1.3.4.1`
- **Primary Checkpoint:** [`models/reader/a1.3.4.1/best_norm_ED.pth`](file:///d:/Projects-anti/lpr/models/reader/a1.3.4.1/best_norm_ED.pth)
- **Convenience Alias:** `models/reader/thai_plate_reader.pth`
- **Architecture:** Deep Text Recognition Benchmark (ResNet/CNN Feature Extraction + BiLSTM Sequence Modeling + Attention Decoder)
- **Weight Size:** ~57.9 MB
- **Task:** Thai alphanumeric character segmentation & province text transcription
- **Input:** Rectified license plate crops (normalized aspect ratio)

---

### 3. Context Vehicle Detector (DLT Classification)
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

### 4. Zero-Shot Open-Vocabulary Detector (OWOD Studio)
- **Model Key:** `yolov8s-worldv2`
- **Primary Checkpoint:** [`models/detector/yolov8s-worldv2.pt`](file:///d:/Projects-anti/lpr/models/detector/yolov8s-worldv2.pt) (also linked at root)
- **Architecture:** YOLO-World v2 Small
- **Weight Size:** ~26.6 MB
- **Task:** Open-world zero-shot proposal generation via dynamic text prompt (`"license plate"`).

---

### 5. Edge Baseline Detector
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
│   ├── Front-Rear-LPR.combinev1.yolov11n/
│   │   ├── train2/
│   │   │   ├── weights/
│   │   │   │   ├── best.pt              # [Active] 99.36% mAP50 Front/Rear/Plate
│   │   │   │   └── last.pt
│   │   │   ├── args.yaml
│   │   │   ├── results.csv
│   │   │   └── *.png (curves & confusion matrices)
│   │   └── train22/                     # Validation run plots
│   ├── front_rear_lpr_yolov11n.pt       # Direct link to best.pt
│   ├── rear_front_best.pt               # Direct link to best.pt
│   ├── yolov8s.pt                       # COCO vehicle detector
│   ├── yolov8s-worldv2.pt               # YOLO-World v2 zero-shot detector
│   └── yolov8n.pt                       # Lightweight edge base
│
├── reader/
│   ├── a1.3.4.1/
│   │   └── best_norm_ED.pth             # [Active] Thai Character & Province OCR
│   └── thai_plate_reader.pth            # Direct link to best_norm_ED.pth
│
└── MODEL_CATALOG.md                     # This governance record
```

*(Note: `model/` is an NTFS junction pointing to `models/` for 100% backward compatibility with singular/plural path references).*
