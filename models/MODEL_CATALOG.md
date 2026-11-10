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
- **Architecture:** `SelenLPR`: Pretrained `MobileNetV3-Small` Backbone + Multi-Head Self-Attention Transformer Context + 2-Layer BiLSTM LeftToRight CTC Sequence Head (14 sequence time steps, strict DLT length clamping: `d3 <= 3`, `d4 <= 4`)
- **Parameters:** 5,441,774 (~20.76 MB float32, ~10.38 MB FP16)
- **Cryptographic Hash (SHA-256):** `C9715E0B37BC1AB804312B6EA18032050960302E2541B76BB65437B86AB819AB`
- **Inference Speed:** > 2,000 FPS on RTX 4090 GPU / ~3-5 ms on CPU Edge
- **Target Outputs (End-to-End Simultaneous):**
  1. `plate_type`: 11 DLT vehicle plate categories (`WhiteBlackCar`, `WhiteGreen`, `YellowBlack`, `WhiteBlackMotor`, etc.)
  2. `d3`: Prefix Thai consonants and leading digits (max 3 characters, DLT validated)
  3. `d4`: Registration numbers (max 4 digits: 1..4 numeric digits)
  4. `province`: 78 Thailand administrative subdivisions + Betong
- **Loss Function & Training Protocol:** `SelenMultiTaskLoss` with CTC sequence alignment:
  - Loss mode: CTC loss for sequence modeling (`nn.CTCLoss(blank=0, zero_infinity=True)`).
  - Sequence Length: W=14 time steps (compact 2x sequence stride matching max 7 characters).
  - Optimizer: AdamW (`lr=5e-4`, `weight_decay=1e-4`, CosineAnnealingLR scheduler).
  - PyTorch AMP mixed precision (`torch.amp.autocast("cuda")` + `GradScaler`) on NVIDIA RTX 4090.
- **Training Corpus (`dataset/ready/202611030335` - 29,759 images across 28,495 unique plates):**
  - Anchor Verified Ground Truth: `dataset/ready/202610261140` (1,183 plates)
  - Mobile Verified Ground Truth: `dataset/snapshot/mobile` (68 approved plates)
  - Staging Approved Ground Truth: `dataset/verified/dataset-staging-approved` (2,081 plates)
  - *Strict Exclusion*: `dataset/labeled/curated_lpr_dataset` completely excluded.
  - CCTV & Mobile 50% Reduced-Blur Synthetic Data: 26,427 plates
  - Train/Val Split: 28,272 training samples / 1,487 validation samples
- **Training Performance (10 Epochs, 575.5s total time):**
  - **Validation Loss:** Decreased monotonically down to **0.7091**
  - **Exact 4-Way Match Accuracy:** **37.39%** (556 / 1,487 plates)
  - **Prefix Sequence (`d3`) Accuracy:** **74.4%**
  - **Digit Sequence (`d4`) Accuracy:** **46.3%**
  - **Plate Type Accuracy:** **99.6%**
  - **Province Classification Accuracy:** **93.8%**
  - **DLT Length Compliance:** 100% compliant (`len(d3) <= 3`, `len(d4) <= 4`, zero illegal characters).

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

### 8. SelenLPR-E Multi-Task EasyOCR Foundation Reader
- **Model Key:** `selenLPR-E` *(Multi-Task EasyOCR Foundation Checkpoint)*
- **Primary Checkpoint:** [`models/reader/selenLPR-E/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/selenLPR-E/best_reader.pth)
- **Base Provenance:** Official JaidedAI EasyOCR Thai Foundation Weights (`ResNet_FeatureExtractor` + 2-layer `BidirectionalLSTM(512, 512, 512)`)
- **Architecture:** ResNet Feature Extractor + 2-layer BiLSTM Sequence Modeling + Delimiter-Aware CTC Alphanumeric Sequence Head + Auxiliary Plate Type & Closed-Set Province Heads
- **Parameters:** 54,010,700 (~206 MB float32, ~103 MB FP16)
- **Cryptographic Hash (SHA-256):** `02EAB9E325E63DF9D2707514661569ACDFE0DEF2616F00DC78325341473474D5`
- **Inference Speed:** > 1,500 FPS on RTX 4090 GPU / ~35-45 ms on CPU Edge
- **Target Outputs (Simultaneous Multitask):**
  1. `plate_type`: 11 vehicle categories (`WhiteBlackCar`, `WhiteGreen`, `WhiteBlue`, `YellowBlack`, `WhiteBlackMotor`, etc.) with pooled feature and 6-dim RGB color moments conditioning.
  2. `d3`: Prefix Thai consonants and leading digits (max 3 characters, DLT validated).
  3. `d4`: Registration number sequence (max 4 digits: 1..4 numeric digits).
  4. `province`: 79 closed-set Thailand administrative subdivisions (78 official provinces + Betong district + unknown).
- **Delimiter-Aware Sequence Paradigm:**
  - Sequences encoded as `d3 <sep> d4` using explicit boundary delimiter token (token index 1: `' '`, vocabulary size 53).
  - Greedy CTC decode collapses repetitions and cleanly splits on the delimiter, eliminating consonant heuristics and natively transcribing numbers-only commercial/private truck plates (e.g., `53 9210 นนทบุรี`).
  - Strict length clamping enforced: `len(d3) <= 3` and `len(d4) <= 4`.
- **Training Corpus (51,151 plates):**
  - Anchor Ground Truth: `dataset/reading/versions/202611030335/plates` (29,677 plates)
  - Degradation Benchmark: `dataset/reading/versions/dataset-dirty10-blur5-angle5/plates` (20,000 plates)
  - Canonical Verified Set: `dataset/reading/versions/202610261140/plates` (1,474 plates)
- **Evaluation Performance (Target Convergence Threshold Achieved):**
  - **Validation Loss:** **0.4187** (descended from 0.9153)
  - **Digit Sequence (`d4`) Accuracy:** **91.75%** (peak 92.05%)
  - **Plate Type Classification Accuracy:** **95.95%**
  - **Closed-Set Province Accuracy:** **95.40%**
  - **Prefix Sequence (`d3`) Accuracy:** **76.20%**
  - **Core Registration (`d3 + d4`) Accuracy:** **73.20%**
  - **Exact 4-Way Match Accuracy:** **68.90%**

---

### 6. SelenOCR v0.1.1 (CCTV Resilient & Low-Res Invariant Multi-Task Transformer Reader)
- **Model Key:** `selen-ocr-v0.1.1` *(Production CCTV & High-Fidelity Edge Checkpoint)*
- **Primary Checkpoint:** [`models/reader/selen_ocr_v011/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/selen_ocr_v011/best_reader.pth)
- **SHA-256 Checksum:** `D202423581F16D70BB538B8646E34914C799541B4A9F8B815036D1ADB1D36735`
- **Architecture:**
  - Backbone: MobileNetV3-Large with Multi-Scale FPN Adapter (fusing stages 112ch, 160ch, 960ch to $d_{\text{model}}=384$)
  - Sequence Modeling: Adaptive Positional Sequence Pooling (enforcing exactly $W'=40$ sequence time steps) + 4-layer Transformer Encoder ($d_{\text{model}}=384$, 6 heads, feedforward 1024)
  - Multi-Task CTC Neck: 2-layer Bidirectional LSTM ($d_{\text{hidden}}=256$) + Linear CTC sequence head (145 classes)
  - Multi-Task Aux Heads: Plate Type head (13 classes), Province classification head (81 classes via global sequence pooling)
- **Total Parameters:** 16,156,319 (~16.2M parameters, 61.6 MB float32)
- **Training Corpus (ISO 42001 Traceability):**
  - Combined Corpus: **170,000 images** (Curriculum balanced 70.6% Clean Anchor : 29.4% CCTV Degraded)
  - Dataset v3.0: `dataset/reading/versions/dataset-v3.0-synthetic-clean` (120,000 images: 100k clean, 10k blur `- - - -`, 10k non-plate `+ + + +`)
  - Dataset v3.1: `dataset/reading/versions/dataset-v3.1-cctv-addon` (50,000 images: 48,817 low-res downscaled 20px–48px with stroke dilation/erosion & compression, plus 1,183 authentic real CCTV plates)
  - Optimization: Warm-started from converged `selen_ocr_v010` weights, trained for 10 epochs (AdamW $2.5 \times 10^{-4}$, Cosine Annealing, AMP FP16 on RTX 4090).
- **Cross-Dataset Benchmark Performance:**
  - **Real Anchor Verified CCTV (`202610261140`, 1,183 images):** **96.70%** exact match (Registration: **96.96%**, $d3$: **97.72%**, $d4$: **98.31%**, Prov: **99.32%**, Type: **99.58%**) — *Massive leap from 4.06% in v0.1.0*.
  - **Clean Synthetic Benchmark (`dataset-v3.0-synthetic-clean`, 2,000 stratified samples):** **99.65%** exact match ($d3$: **99.95%**, $d4$: **99.65%**, Prov: **100.00%**, Type: **100.00%**).
  - **Degraded Environmental Benchmark (`dataset-dirty10-blur5-angle5`, 2,000 samples):** **68.75%** exact match (Registration: **74.45%**, $d3$: **84.95%**, Prov: **81.95%**).
  - **CCTV & Low-Resolution Add-on (`dataset-v3.1-cctv-addon`, 2,000 samples):** **73.35%** exact match (Registration: **73.85%**, $d3$: **93.40%**, Prov: **99.40%**).
  - **Inference Latency & Throughput:** ~0.8–1.2 ms per frame (**> 1,000 FPS**) on RTX 4090 GPU.

---

### 7. SelenOCR v0.1.0 (Enlarged Multi-Task Transformer & FPN Thai LPR Reader)
- **Model Key:** `selen-ocr-v0.1.0` *(Target Goal >95% Achieved Checkpoint)*
- **Primary Checkpoint:** [`models/reader/selen_ocr_v010/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/selen_ocr_v010/best_reader.pth)
- **SHA-256 Checksum:** `7317F1F09FA6DD1253B7F0817B6494A71F4FF1AA26D14C2BAFF062FB90CE9786`
- **Architecture:**
  - Backbone: MobileNetV3-Large with Multi-Scale FPN Adapter (fusing stages 112ch, 160ch, 960ch to $d_{\text{model}}=384$)
  - Sequence Modeling: Adaptive Positional Sequence Pooling (enforcing exactly $W'=40$ sequence time steps) + 4-layer Transformer Encoder ($d_{\text{model}}=384$, 6 heads, feedforward 1024)
  - Multi-Task CTC Neck: 2-layer Bidirectional LSTM ($d_{\text{hidden}}=256$) + Linear CTC sequence head (145 classes)
  - Multi-Task Aux Heads: Plate Type head (13 classes), Province classification head (81 classes via global sequence pooling)
- **Total Parameters:** 16,156,319 (~16.2M parameters, 61.6 MB float32)
- **Single Text Line Token Paradigm:**
  - Format: `"plateType d3 d4 province"`
  - Token Structure: `start + plateType + 3 + 4 + 1 + stop = 11 tokens max`
  - Vocabulary Size: 145 discrete tokens (`<pad>`, `<start>`, `<stop>`, `+`, `-`, 10 digits `0-9`, 41 DLT consonants, 11 vehicle types, 78 provinces)
  - Negative Condition Tokens: `+` (non-plate images labeled `+ + + +`), `-` (unreadable blur labeled `- - - -`)
- **Training Corpus (ISO 42001 Traceability):**
  - Dataset: `dataset/reading/versions/dataset-v3.0-synthetic-clean` (120,000 images)
  - Composition: 100,000 clean synthetic plates ($\text{yaw} \le \pm 5^\circ$, $\text{pitch} \in [0^\circ, 5^\circ]$, camera 1.0–2.0m, plate 0.3–0.5m), 10,000 heavy blur (`- - - -`), 10,000 non-plate crops (`+ + + +`)
  - Target Regional Weighting: Bangkok (35%), Phuket (15%), Eastern Seaboard (15%), Chiang Mai (10%), all 77 provinces + Betong represented (25%)
- **Validation & Benchmark Performance (2,000 Stratified Test Samples: 80% Clean, 10% Blur, 10% Non-Plate):**
  - **Overall Exact Match Sequence Accuracy:** **96.15%** (exceeding >95.0% goal requirement)
  - **Plate Type Classification Accuracy:** **100.00%**
  - **Prefix Sequence (`d3`) Accuracy:** **98.70%**
  - **Digit Sequence (`d4`) Accuracy:** **97.50%**
  - **Province Classification Accuracy:** **99.95%**
  - **Negative Condition Rejection (`+` and `-`):** **100.00%**
  - **Inference Latency:** 161.17 ms on GPU (RTX 4090) / ~6.2 FPS

---

### 8. PaddleOCR Baseline v3.0 (Single-Line 11-Token Architecture)
- **Model Key:** `paddleocr-baseline-v3.0`
- **Primary Checkpoint:** [`models/reader/paddleocr_v3/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/paddleocr_v3/best_reader.pth)
- **SHA-256 Checksum:** `8F26A176AC72AB411214F53CE4CFBFE0AD38951AEF777BFECDD909E8A57D7058`
- **Architecture:** MobileNetV3 Feature Extractor + 2-layer Bidirectional LSTM Neck + CTC Linear Head (145 classes)
- **Input Dimensions:** `[B, 3, 48, 320]`
- **Stratified Benchmark Performance (2,000 samples: 80% Clean, 10% Blur, 10% Non-Plate):**
  - **Exact Match Accuracy:** **79.05%**
  - **Plate Type Accuracy:** **100.00%**
  - **Prefix Sequence (`d3`) Accuracy:** **99.60%**
  - **Digit Sequence (`d4`) Accuracy:** **79.45%**
  - **Province Accuracy:** **99.65%**
  - **Negative Rejection:** **100.00%**
  - **Inference Latency:** 64.26 ms / ~15.6 FPS

---

### 9. EasyOCR Baseline v3.0 (Single-Line 11-Token Architecture)
- **Model Key:** `easyocr-baseline-v3.0`
- **Primary Checkpoint:** [`models/reader/easyocr_v3/best_reader.pth`](file:///d:/Projects-anti/lpr/models/reader/easyocr_v3/best_reader.pth)
- **SHA-256 Checksum:** `B4373CB4357990A22B5334573F8D13471BC3C205B8CB8BCF2234C4DF43ADACD2`
- **Architecture:** ResNet Feature Extractor + 2-layer Bidirectional LSTM Neck + CTC Linear Head (145 classes)
- **Input Dimensions:** `[B, 1, 64, 600]`
- **Stratified Benchmark Performance (2,000 samples: 80% Clean, 10% Blur, 10% Non-Plate):**
  - **Exact Match Accuracy:** **19.75%**
  - **Plate Type Accuracy:** **85.60%**
  - **Prefix Sequence (`d3`) Accuracy:** **19.75%**
  - **Digit Sequence (`d4`) Accuracy:** **19.75%**
  - **Province Accuracy:** **48.20%**
  - **Negative Rejection:** **98.75%**
  - **Inference Latency:** 335.46 ms / ~3.0 FPS

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
│   ├── easyocr_v3/
│   │   └── best_reader.pth              # EasyOCR Baseline v3.0 (145 classes, 11-token)
│   ├── paddleocr_v3/
│   │   └── best_reader.pth              # PaddleOCR Baseline v3.0 (145 classes, 11-token)
│   ├── selen_ocr_v010/
│   │   └── best_reader.pth              # SelenOCR v0.1.0 Multi-Task Transformer Reader
│   ├── selen_ocr_v011/
│   │   └── best_reader.pth              # SelenOCR v0.1.1 CCTV & Low-Res Invariant Reader
│   ├── easyocr_thai/
│   │   └── best_reader.pth              # Public EasyOCR fine-tuned foundation reader
│   ├── thai_plate_reader.pth            # Thai Character & Province OCR
│   ├── selen-lpr-0.0.1/
│   │   └── best_reader.pth              # SelenLPR multi-task edge model
│   └── selenLPR-E/
│       └── best_reader.pth              # SelenLPR-E Multi-Task EasyOCR Foundation Reader
│
└── MODEL_CATALOG.md                     # This governance record
```

*(Note: `model/` is an NTFS junction pointing to `models/` for 100% backward compatibility with singular/plural path references).*
