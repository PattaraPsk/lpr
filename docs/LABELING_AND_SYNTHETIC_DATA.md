# Thai LPR: Labeling Process, Synthetic Data Generation & Standards Compliance
## Technical Specification & Operational Manual

This document details the end-to-end data engineering architecture, labeling workflows, synthetic generation pipelines, and international standards compliance for the Thai License Plate Recognition (ALPR/ANPR) Edge System.

---

## 📑 Table of Contents
1. [Executive Summary & Architectural Context](#1-executive-summary--architectural-context)
2. [Labeling Process & Annotation Engineering](#2-labeling-process--annotation-engineering)
   - [2.1 Data Ingestion & Auto-Crop Pipeline](#21-data-ingestion--auto-crop-pipeline)
   - [2.2 AI-Assisted Pre-Annotation (VLM & Edge Reader)](#22-ai-assisted-pre-annotation-vlm--edge-reader)
   - [2.3 Human-in-the-Loop Verification (DataManipulator GUI)](#23-human-in-the-loop-verification-datamanipulator-gui)
   - [2.4 5-Element File Naming & Annotation Scheme](#24-5-element-file-naming--annotation-scheme)
   - [2.5 Rejection Classes & Ambiguity Handling](#25-rejection-classes--ambiguity-handling)
   - [2.6 Session Persistence & Ground-Truth Serialization](#26-session-persistence--ground-truth-serialization)
3. [Synthetic Data Generation Pipeline](#3-synthetic-data-generation-pipeline)
   - [3.1 Motivation & The Real-World Domain Gap](#31-motivation--the-real-world-domain-gap)
   - [3.2 Calibrated Vector Font Engine (DLT Suite)](#32-calibrated-vector-font-engine-dlt-suite)
   - [3.3 3D Physical Camera & Perspective Calibration](#33-3d-physical-camera--perspective-calibration)
   - [3.4 Environmental Degradation & Optical Artifacts](#34-environmental-degradation--optical-artifacts)
   - [3.5 The 70:30 Golden Ratio Curriculum (170k Corpus)](#35-the-7030-golden-ratio-curriculum-170k-corpus)
   - [3.6 Empirical Validation & Benchmark Breakdown](#36-empirical-validation--benchmark-breakdown)
4. [International & Enterprise Standards Compliance Matrix](#4-international--enterprise-standards-compliance-matrix)
   - [4.1 Standards Traceability Matrix](#41-standards-traceability-matrix)
   - [4.2 ISO/IEC 42001:2023 (AI Management System)](#42-isoiec-420012023-ai-management-system)
   - [4.3 ISO/IEC 5259-2:2023 (AI Data Quality Measures)](#43-isoiec-5259-22023-ai-data-quality-measures)
   - [4.4 ISO/IEC 25012:2008 (Data Integrity & SHA-256 Provenance)](#44-isoiec-250122008-data-integrity--sha-256-provenance)
   - [4.5 ISO 3166-2:TH & Unicode 15.0 NFC Normalization](#45-iso-3166-2th--unicode-150-nfc-normalization)
   - [4.6 Thailand DLT Statutory Vehicle Plate Specifications](#46-thailand-dlt-statutory-vehicle-plate-specifications)
   - [4.7 ISO/IEC 27001:2022 (Privacy & Path Sanitization)](#47-isoiec-270012022-privacy--path-sanitization)

---

## 1. Executive Summary & Architectural Context

Thai License Plate Recognition poses unique computer vision challenges:
- **Complex Orthography**: 41 valid consonants, distinct closed loops (`ข` vs `ช`, `ด` vs `ต`, `ท` vs `ฑ`), tone marks, and 77 administrative provinces plus the special administrative district Betong.
- **Physical Variations**: Multiple plate colorways (private cars, commercial trucks, motorcycle formats with stacked alphanumeric rows, and public transport).
- **Extreme Optical Degradation**: Low-resolution CCTV cameras (20px–48px vertical plate crops), perspective tilts, vehicle mounting angles, motion blur, and headlight bloom during nighttime conditions.
- **Extreme Class Imbalance**: Real-world traffic is heavily concentrated in Bangkok and major metropolitan corridors, resulting in severe data scarcity for regional provinces.

To achieve production-grade accuracy (>95% on authentic CCTV streams) without prohibitive manual labeling costs, the system integrates a **semi-automated human-in-the-loop annotation pipeline** with a **physics-calibrated parametric synthetic generation engine** operating under rigorous ISO/IEC AI governance standards.

---

## 2. Labeling Process & Annotation Engineering

The data annotation architecture bridges raw edge surveillance video and cryptographically verified training datasets.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Stream & Video Ingestion"]
        RTSP["CCTV / RTSP Stream (tools/rtsp_collector)"]
        VIDEO["Recorded Video (MP4 / H.264)"]
    end

    subgraph AutoExtract ["2. Detection & Bounding Extraction"]
        YOLO["Vehicle & Plate Detector (YOLO-World / YOLOv8/v11)"]
        TRACK["Tracking & Snapshot Extractor (tools/process_and_track_lpr.py)"]
    end

    subgraph PreLabel ["3. AI Pre-Annotation Engine"]
        VLM["Offline VLM / PaddleOCR-VL (tools/offline_vlm_annotator.py)"]
        SELEN["SelenOCR Fast Predictor (tools/label_unverified_with_selen.py)"]
        HASH["Cryptographic SHA-256 Deduplication (ISO/IEC 25012)"]
    end

    subgraph HumanLoop ["4. Human-in-the-Loop Curation"]
        GUI["DataManipulator GUI (tools/DataManipulator)"]
        DIFF["Side-by-Side Current vs AI Prediction Grid"]
        HOTKEY["Rapid Verification (Enter / Ctrl+R / Ctrl+N / Ctrl+D)"]
    end

    subgraph MasterData ["5. Standardized Serialization"]
        SESSION["Session State (annotations/plates_session.json)"]
        APPROVED["Approved JSON / CSV (annotations/approved_plates.json)"]
        FILES["Standardized 5-Element Filenames"]
    end

    Ingestion --> AutoExtract
    AutoExtract --> PreLabel
    PreLabel --> HumanLoop
    HumanLoop --> MasterData
```

### 2.1 Data Ingestion & Auto-Crop Pipeline
1. **Continuous Capture (`tools/rtsp_collector/`)**:
   - Collects 24/7 RTSP camera streams conforming to ISO/IEC 14496-14 and ISO/IEC 27001 (credential masking).
   - Generates segmented MP4 files with verified moov-atom container integrity.
2. **Object Detection & Tracking (`tools/process_and_track_lpr.py`)**:
   - Coarse vehicle bounding boxes and fine license plate crops are extracted using fine-tuned YOLO detectors.
   - Spatial consensus voting aggregates consecutive video frames to select the highest-resolution, sharpest candidate plate snapshot for annotation.

### 2.2 AI-Assisted Pre-Annotation (VLM & Edge Reader)
Prior to manual review, unannotated plate crops pass through automated pre-labeling engines:
- **Offline VLM Annotator (`tools/offline_vlm_annotator.py`)**:
  - Leverages PaddleOCR-VL in FP16 precision.
  - Automatically parses text into 3 distinct logical tokens: $d3$ (series letters), $d4$ (number), and $province$.
  - Applies deterministic dictionary matching against official Thailand DLT provinces and active consonants.
- **Deduplication Engine**:
  - Computes chunked SHA-256 hashes (`chunk_size = 65536`) for every image payload to eliminate duplicate frames across surveillance clips.

### 2.3 Human-in-the-Loop Verification (`DataManipulator` GUI)
Human annotators verify and correct draft labels through a dedicated desktop application (`tools/DataManipulator/`):
- **Live Comparison Grid**: Displays side-by-side comparison between Current values and AI Predictions with visual diff highlights (`Matched ✅`, `Differences ⚠️`).
- **Selective Field Replacement**:
  - Single-field override buttons (`⬅ d3`, `⬅ d4`, `⬅ Prov`).
  - Batch replacement shortcut `Ctrl+R` to instantly populate all fields with model predictions when confident.
- **Subfolder Grouping & Recursive Inspection**:
  - Hierarchically navigates dataset trees with folder-level progress statistics.
- **Searchable Autocomplete**:
  - Fuzzy-search dropdown for all 77 Thai provinces and Betong mapped to ISO 3166-2:TH codes.

### 2.4 5-Element File Naming & Annotation Scheme
Every verified image is atomically renamed and indexed according to the standard 5-element schema:

$$\text{\{name\}\_\{plate\_type\}\_\{d3\}\_\{d4\}\_\{province\}.jpg}$$

#### Field Definitions:
1. **`name`**: Unique sample identifier or video clip origin (e.g., `AA09`, `CAM01_00123`).
2. **`plate_type`**: Vehicle and plate classification category:
   - `WhiteBlackCar`: Private passenger vehicles ($\le 7$ seats).
   - `WhiteBlackMotor`: Standard private motorcycles.
   - `WhiteBlue`: Vans, minibuses, and passenger vehicles ($> 7$ seats).
   - `WhiteGreen`: Private commercial pickups and light trucks.
   - `YellowBlack`: Public taxis and commercial hire vehicles.
   - `YellowBlue`: Fixed-route public buses and songthaews.
   - `YellowGreen`: Public three-wheeled tuk-tuks.
   - `GreenWhite`: Airport, hotel, and luxury limousine transport.
   - `RedBlack`: Temporary dealer registration.
   - `OrangeBlack`: Agricultural tractors and heavy machinery.
3. **`d3`**: Front alphanumeric series characters (e.g., `1กข`, `ขก`, `หม`, `3กส`).
4. **`d4`**: Numeric registration digits (1 to 4 digits, e.g., `1`, `400`, `1168`, `9999`).
5. **`province`**: Canonical Thai administrative subdivision name conforming to ISO 3166-2:TH (e.g., `กรุงเทพมหานคร`, `เชียงใหม่`, `สงขลา`).

### 2.5 Rejection Classes & Ambiguity Handling
To train robust classification and avoid model hallucinations on out-of-distribution frames:
- **Unreadable / Degraded Plates (`- - - -`)**:
  - Filename format: `{name}_-_-_-_-_.jpg`
  - Assigned when optical degradation, extreme motion blur, or severe damage prevents human recognition.
- **Non-Plate Negative Samples (`+ + + +`)**:
  - Filename format: `{name}_+_+_+_+.jpg`
  - Assigned to false-positive detector crops (e.g., car grilles, badges, headlamps, bumpers) to train model background rejection.

### 2.6 Session Persistence & Ground-Truth Serialization
- **Session File (`annotations/plates_session.json`)**:
  - Maintains state, current review index, modification timestamps (ISO 8601 UTC), reviewer approval status (`approved`, `pending`, `skipped`), and ISO 3166-2 codes.
- **Atomic Persistence**:
  - Implements atomic file write-and-replace using temporary files (`.tmp` $\to$ `.json`) to prevent data corruption during system crashes.
- **Ground Truth Export**:
  - Exported to `annotations/approved_plates.json` and tabular `annotations/approved_plates.csv` for direct consumption by training pipelines.

---

## 3. Synthetic Data Generation Pipeline

Real-world datasets suffer from insurmountable sample deficiencies: rare provinces have negligible representation, and extreme low-light CCTV footage cannot be gathered in sufficient volume without privacy and cost constraints. The synthetic generation pipeline synthetically models the full physical imaging chain.

### 3.1 Motivation & The Real-World Domain Gap
Early baseline models trained purely on clean synthetic plates failed catastrophically on surveillance footage (achieving only **4.06% exact match** on raw CCTV test sets). Analysis revealed three core domain discrepancies:
1. **Sub-pixel Loop Erosion & Dilation**: Character loops in Thai glyphs (e.g., `ข` vs `ช`) collapse into solid blobs under low CCTV resolution ($<32\text{px}$).
2. **Camera Perspective**: True surveillance cameras are mounted overhead ($1.0\text{m} - 2.0\text{m}$) at non-zero pitch angles.
3. **Sensor Noise & Macroblocking**: H.264/H.265 compression artifacts destroy fine stroke gradients.

### 3.2 Calibrated Vector Font Engine (DLT Suite)
Standard public Thai fonts fail to replicate the exact aspect ratios, stroke widths, and corner radiuses of physical embossed DLT dies.
- **Vector Font Calibration (`tools/build_font_suite_v10.py`)**:
  - Generates specialized TrueType font suites (`rbz-licenseplate-*.ttf`):
    - `rbz-licenseplate-id.ttf`: Standard automobile embossed letters and digits.
    - `rbz-licenseplate-prov.ttf`: Province sub-line glyphs.
    - `rbz-licenseplate-large.ttf`: Heavy truck/trailer typography.
    - `rbz-licenseplate-yellow.ttf`: Transport/commercial specific character weights.
  - Ensures accurate character glyph proportions and eliminates stroke clipping during rendering.

### 3.3 3D Physical Camera & Perspective Calibration
The generator (`tools/realistic_plate_generator.py`, `tools/generate_v3_clean_synthetic_dataset.py`) implements a calibrated 3D geometric transformation model:

| Physical Parameter | Calibrated Value Range | Physical Justification |
| :--- | :--- | :--- |
| **Focal Length** | $4.0\text{ mm}$ | Typical wide-angle commercial security camera lens. |
| **Camera Mounting Height** | $1.0\text{ m} - 2.0\text{ m}$ | Standard entry gate / pole / gantry surveillance height. |
| **Plate Height** | $0.3\text{ m} - 0.5\text{ m}$ | Vehicle bumper and motorcycle tail mounting range. |
| **Pitch Angle ($\theta_x$)** | $0^\circ \text{ to } 5^\circ$ | Downward viewing angle from pole to plate. |
| **Yaw Angle ($\theta_y$)** | $-5^\circ \text{ to } +5^\circ$ | Vehicle lane offset and approach trajectory. |
| **Roll Angle ($\theta_z$)** | $-2^\circ \text{ to } +2^\circ$ | Road crown slope and vehicle suspension tilt. |

### 3.4 Environmental Degradation & Optical Artifacts
To simulate field degradation, generated plate dies are processed through the degradation engine (`tools/dirty_plate_synthesizer.py`, `tools/generate_v31_cctv_addon_dataset.py`):
1. **Low-Resolution Downscaling**:
   - Randomly downsamples plates to vertical heights between **20px and 48px**, followed by bicubic upsampling to input resolution ($64 \times 192\text{px}$).
2. **Morphological Variations**:
   - **Dilation ($k = 3 \times 3$)**: Simulates over-inking, night-time headlight bloom, and retroreflective overexposure.
   - **Erosion ($k = 3 \times 3$)**: Simulates paint chipping, weathered embossing, and fading.
3. **Surveillance Compression Artifacts**:
   - JPEG compression quality degradation ($Q \in [25, 65]$).
   - H.264 macroblocking and discrete cosine transform (DCT) block quantization.
4. **Sensor & Motion Noise**:
   - Directional linear horizontal motion streaks (vehicle movement).
   - Additive Gaussian noise ($\sigma \in [5, 25]$) and Poisson photon shot noise.
5. **Physical Occlusions & Mounting Artefacts**:
   - Random insertion of mounting bolt screws, shadows cast by vehicle bumpers, mud splashes, and edge margin jitter ($5\% - 15\%$).

### 3.5 The 70:30 Golden Ratio Curriculum (170k Corpus)
To train **SelenOCR v0.1.1**, the dataset was compiled following an empirically calibrated 70:30 ratio balancing clean structural representation against real-world degradation:

```mermaid
pie title 70:30 Corpus Composition (170,000 Images)
    "Clean Physical Plates (v3.0)" : 100000
    "Severe Blur Rejection (- - - -)" : 10000
    "Non-Plate Negative Crops (+ + + +)" : 10000
    "Synthetic CCTV & Low-Res (v3.1)" : 48817
    "Authentic CCTV Anchor Plates" : 1183
```

#### Composition Breakdown:
1. **Dataset v3.0 (Clean Anchor — 120,000 images, ~70%)**:
   - **100,000 Clean Plates**: Balanced across regional traffic density:
     - Bangkok Metropolitan Area: 35% (35,000)
     - Phuket: 15% (15,000)
     - Eastern Seaboard (Chonburi, Rayong, Chachoengsao): 15% (15,000)
     - Chiang Mai: 10% (10,000)
     - Remaining 74 provinces + Betong: 25% (25,000) — **guaranteeing 100% provincial coverage**.
   - **10,000 Severe Blur Samples (`- - - -`)**: Inculcates unreadable rejection.
   - **10,000 Non-Plate Crops (`+ + + +`)**: Inculcates false-positive rejection.
2. **Dataset v3.1 (CCTV & Low-Res Add-on — 50,000 images, ~30%)**:
   - **48,817 Synthetic CCTV Plates**: Downsampled to $20\text{px} - 48\text{px}$ with morphological dilation/erosion and compression artifacts.
   - **1,183 Authentic CCTV Anchor Plates (`202610261140`)**: Real field-captured plates acting as the empirical domain bridge.

### 3.6 Empirical Validation & Benchmark Breakdown
Training on this combined corpus resolved the real-world domain gap without inducing catastrophic forgetting on clean plates:

| Benchmark Dataset | Test Samples | Baseline SelenOCR v0.1.0 | Production SelenOCR v0.1.1 | Performance Delta | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Anchor CCTV (`202610261140`)** | 1,183 | 4.06% | **96.70%** | **+92.64%** 🚀 | 1.22 ms |
| **Clean Synthetic Target (`v3.0`)** | 2,000 | 95.85% | **99.65%** | **+3.80%** | 1.02 ms |
| **Degraded Environmental (`dirty10-blur5-angle5`)** | 2,000 | 58.90% | **68.75%** | **+9.85%** | 0.84 ms |
| **CCTV & Low-Res Add-on (`v3.1`)** | 2,000 | — | **73.35%** | *New Baseline* | 0.87 ms |
| **Mobile Camera Synthetic** | 1,500 | 12.30% | **23.40%** | **+11.10%** | 0.91 ms |

---

## 4. International & Enterprise Standards Compliance Matrix

The repository strictly enforces international quality, data integrity, and AI governance standards across all modules and scripts.

### 4.1 Standards Traceability Matrix

| Standard | Domain & Scope | Enforcement Modules | Compliance Mechanism |
| :--- | :--- | :--- | :--- |
| **ISO/IEC 42001:2023** | Artificial Intelligence Management System (AIMS) | `CHANGELOG.md`<br>`models/MODEL_CATALOG.md`<br>`tools/iso42001_logger.py`<br>`AGENTS.md` | Formal lifecycle change logging (Clause 8.4, Annex A.6.7), standardized model lineage cards (Annex A.8), operational event logging (Annex A.9.3), and zero unverified model deployments. |
| **ISO/IEC 5259-2:2023** | AI Data Quality Measures for ML | `tools/owod/core/quality_auditor.py`<br>`tools/owod/core/dataset_generator.py` | Automated dataset auditing: bounding box syntactic boundaries ($0 \le x_c, y_c, w, h \le 1$), degenerate zero-area checks, split ratio compliance, and class balance reporting. |
| **ISO/IEC 25012:2008** | Data Quality Model (Data Integrity) | `tools/detect_cam_videos.py`<br>`tools/rtsp_collector/manifest.py`<br>`tools/DataManipulator/app/session_manager.py` | Cryptographic SHA-256 calculation for all media payloads, immutable manifest ledgers, and atomic file write patterns (`.tmp` $\to$ target). |
| **ISO 3166-2:TH** | Country Subdivision Codes (Thailand) | `tools/DataManipulator/app/provinces.py`<br>`tools/selen_token_codec.py` | Complete mapping of all 77 Thai provinces and Betong to formal administrative identifiers (`TH-10` to `TH-96`). |
| **ISO/IEC 10646 / Unicode 15.0** | Universal Coded Character Set (NFC) | `tools/DataManipulator/app/provinces.py`<br>`tools/owod/core/owod_engine.py` | Mandatory canonical composition (`unicodedata.normalize('NFC', text)`) across all text ingestion and serialization layers. |
| **ISO 8601:2019** | Date, Time & Duration Representation | `tools/rtsp_collector/`<br>`tools/detect_cam_videos.py`<br>`tools/iso42001_logger.py` | Extended UTC representation (`YYYY-MM-DDTHH:MM:SSZ`) and standardized duration tokens (`PT1M30S`). |
| **ISO/IEC 27001:2022** | Information Security & Privacy Protection | `tools/rtsp_collector/security.py`<br>`tools/detect_cam_videos.py` | Mandatory masking of RTSP streaming credentials (`user:***@host`) and user filesystem paths in operational logs. |
| **ISO/IEC 14496-14 / ITU-T H.264** | MP4 Video File Format Integrity | `tools/rtsp_collector/validator.py`<br>`tools/detect_cam_videos.py` | Moov-atom container verification, corrupted frame handling, and real-time buffer frame-dropping logic. |
| **ISO/IEC/IEEE 29119** | Software Testing & Quality Assurance | `tests/test_iso_compliance.py`<br>`tools/DataManipulator/test_app.py` | Automated test suite verifying province lookups, NFC normalization, SHA-256 hashes, and dataset quality gates. |
| **Thailand DLT Statutory Specifications** | Thailand Vehicle Registration Regulations | `AGENTS.md`<br>`STANDARDS.md`<br>`tools/alpr/`<br>`tools/selen_token_codec.py` | Strict enforcement of the 41 approved DLT consonants, exclusion of 3 prohibited glyphs (`ฃ`, `ฅ`, `ฏ`), OCR remap rules, and absence of vowels on standard plates. |

### 4.2 ISO/IEC 42001:2023 (AI Management System)
Compliance is embedded into the development workflow:
1. **Change Control Ledger (`CHANGELOG.md`)**:
   - Every modification to models, datasets, or pipelines must record an entry conforming to ISO 42001 with: Record Identifier, Date (UTC), Component, Change Type, ISO 42001 Clause, Objective, Risk Assessment, and Verification Gate.
2. **Model Catalog (`models/MODEL_CATALOG.md`)**:
   - Comprehensive model cards documenting checkpoint architectures, SHA-256 hashes, input/output tensors, training hyperparameters, and benchmark metrics.
3. **Deterministic Training & Auditing**:
   - Deterministic random seeding (`torch.manual_seed(42)`, `np.random.seed(42)`).
   - Operational event logging emitting ISO 8601 timestamps and sanitized inference parameters.

### 4.3 ISO/IEC 5259-2:2023 (AI Data Quality Measures)
Automated dataset quality audits produce `data_quality_report.json` assessing:
- **Syntactic Validity**: Verification that all bounding box coordinates satisfy $0.0 \le x_{\text{center}}, y_{\text{center}}, w, h \le 1.0$.
- **Geometric Sanity**: Detection and pruning of zero-area or inverted bounding boxes.
- **Data Balance & Representation**: Evaluation of class distribution across vehicle types and regional provinces to measure and mitigate dataset bias.

### 4.4 ISO/IEC 25012:2008 (Data Integrity & SHA-256 Provenance)
Data integrity is preserved through cryptographic controls:
- **Payload Verification**:
  ```python
  def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
      hasher = hashlib.sha256()
      with open(file_path, "rb") as f:
          for chunk in iter(lambda: f.read(chunk_size), b""):
              hasher.update(chunk)
      return hasher.hexdigest().upper()
  ```
- **Crash-Resilient State Persistence**:
  Session files are written to `.tmp` files and verified before atomic OS-level replacement to prevent zero-byte corruptions.

### 4.5 ISO 3166-2:TH & Unicode 15.0 NFC Normalization
- All 77 provinces and Betong are indexed to ISO 3166-2 administrative codes (e.g., `TH-10` for Bangkok, `TH-50` for Chiang Mai, `TH-90` for Songkhla).
- Ingestion and matching modules enforce Unicode NFC normalization:
  ```python
  import unicodedata
  normalized_text = unicodedata.normalize("NFC", raw_text.strip())
  ```
  This eliminates multi-byte decomposition anomalies between decomposed Thai vowels (`เ` + `เ` vs `แ`) and tone mark sequences.

### 4.6 Thailand DLT Statutory Vehicle Plate Specifications
In strict accordance with Thailand Department of Land Transport (DLT) regulations:
- **41 Approved Thai Consonants**:
  `ก ข ค ฆ ง จ ฉ ช ซ ฌ ญ ฎ ฐ ฑ ฒ ณ ด ต ถ ท ธ น บ ป ผ ฝ พ ฟ ภ ม ย ร ล ว ศ ษ ส ห ฬ อ ฮ`
  *(Note: Letters `ซ`, `ห`, and `ป` are actively used in standard series: e.g., `1กซ`, `1กป`, `หม`, `ปข`).*
- **3 Strictly Excluded Consonants**:
  - `ฃ` (Kho Khuat) — Obsolete character; prohibited in vehicle registration.
  - `ฅ` (Kho Khon) — Obsolete character; prohibited in vehicle registration.
  - `ฏ` (To Patak) — Excluded by DLT to prevent optical ambiguity with `ฎ` (Do Chada).
- **OCR Post-Processing Remap Rules**:
  ```python
  OCR_REMAP_RULES = {
      "ฃ": "ข",
      "ฅ": "ค",
      "ฏ": "ฎ",
  }
  ```
- **Absence of Vowels / Tone Marks**:
  Standard Thai alphanumeric registration plates strictly contain **no vowels or tone marks** (excluding modern 2022+ special auction vanity plates).

### 4.7 ISO/IEC 27001:2022 (Privacy & Path Sanitization)
All streaming collectors and operational logs automatically redact sensitive credentials and personal directory structures:
- RTSP URLs are sanitized: `rtsp://admin:secret@192.168.1.50/stream` $\to$ `rtsp://admin:***@192.168.1.50/stream`.
- Host system usernames and local file paths are sanitized across logs and audit outputs (`C:/Users/***/lpr/...`).

---

## 5. Summary & Operational Checklist

| Stage | Key Tool / Script | Input | Output Artifact | Verification Standard |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion** | `tools/rtsp_collector/` | RTSP / IP Camera | Segmented MP4 + Manifest | ISO 14496-14 / ISO 27001 |
| **Tracking** | `tools/process_and_track_lpr.py` | Video Files | Plate Snapshots | ISO 25012 (SHA-256) |
| **Pre-Annotation** | `tools/offline_vlm_annotator.py` | Image Folder | Draft Labels (`.json`) | DLT Standard / Unicode NFC |
| **Human Review** | `tools/DataManipulator/` | Draft Labels | `approved_plates.json` | ISO 3166-2:TH / ISO 5259 |
| **Clean Synthetic**| `tools/generate_v3_clean_synthetic_dataset.py` | DLT Fonts & Rules | 120,000 Plates (v3.0) | ISO 5259 (Balance: 77 Prov) |
| **CCTV Synthetic** | `tools/generate_v31_cctv_addon_dataset.py` | Low-Res & Noise Models | 50,000 Plates (v3.1) | Realistic Physical Imaging |
| **Compliance Test**| `run_tests.bat` | Repository | Test Report | ISO/IEC/IEEE 29119 |
