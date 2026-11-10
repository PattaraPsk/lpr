# Thai LPR — SelenCORE Integration Manual

> [!IMPORTANT]
> This document is the canonical integration reference for embedding the Thai LPR pipeline into SelenCORE or any external system. It covers the **model architecture, I/O contract, call sequence, and output schema** for both the detector and reader.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Model Catalog Summary](#3-model-catalog-summary)
4. [I/O Contract — Detector](#4-io-contract--detector)
5. [I/O Contract — OCR Reader](#5-io-contract--ocr-reader)
6. [SelenCORE Adapter API](#6-selencore-adapter-api)
7. [Quick-Start Integration](#7-quick-start-integration)
8. [Advanced: Per-Component Direct Call](#8-advanced-per-component-direct-call)
9. [Output Schema Reference](#9-output-schema-reference)
10. [Thai Plate Domain Rules](#10-thai-plate-domain-rules)
11. [Configuration & Thresholds](#11-configuration--thresholds)
12. [Error Handling](#12-error-handling)
13. [Thread Safety & Concurrency](#13-thread-safety--concurrency)
14. [Changelog Entry](#14-changelog-entry)

---

## 1. System Overview

```
Input Frame (BGR numpy)
        │
        ▼
┌───────────────────┐
│  Motion Filter    │  MOG2 background subtraction (optional, edge-optimised)
│  (MotionDetector) │  → skips static/parked scenes before heavy GPU inference
└────────┬──────────┘
         │  motion_boxes [x1,y1,x2,y2]
         ▼
┌───────────────────┐
│  Vehicle Detector │  YOLOv8s — COCO classes {car, motorcycle, truck, bus}
│  (optional stage) │  → vehicle bounding boxes + vehicle_type
└────────┬──────────┘
         │  vehicles [(box, type, conf)]
         ▼
┌───────────────────┐
│  Plate Detector   │  YOLOv11n — 3 classes: Front | LicensePlate | Rear
│  (Coarse-to-Fine  │  → plate boxes + orientation boxes
│   or Direct Mode) │
└────────┬──────────┘
         │  plates [(box, conf)]
         │  orientations [(box, "Front"|"Rear", conf)]
         ▼
┌───────────────────┐
│  Spatial Matching │  Associates each plate → vehicle_type, orientation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  OCR Reader       │  VisionAttentionOCR / SelenLPR / SelenLPR-E
│  (PlateReader)    │  Input: PIL Image (plate crop, grayscale NormalizePAD)
│                   │  Output: d3, d4, province, full_plate, confidence
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LPRResult        │  Structured output — ready for SelenCORE event bus
└───────────────────┘
```

---

## 2. Pipeline Architecture

### 2.1 Detection Strategies

| Mode | When Used | Latency | Accuracy |
|------|-----------|---------|----------|
| **Coarse-to-Fine** (default) | Vehicle detector loaded + vehicles found | Lower (crops only) | Highest |
| **Direct (Full Frame)** | No vehicle detector, or `coarse_to_fine=False` | Higher | High |
| **ROI-Direct** | `roi=[x1,y1,x2,y2]` passed | Lowest | Depends on ROI quality |

### 2.2 Detector Class Map (YOLOv11n)

| Class ID | Label | Meaning |
|----------|-------|---------|
| `0` | `Front` | Vehicle/plate approaching (front-facing orientation) |
| `1` | `LicensePlate` | Actual Thai license plate bounding box |
| `2` | `Rear` | Vehicle/plate departing (rear-facing orientation) |

### 2.3 Vehicle Detector Class Map (YOLOv8s / COCO subset)

| COCO Class | Internal Label | DLT Category |
|------------|---------------|--------------|
| `2` | `Car` | Passenger vehicle |
| `3` | `Motorcycle` | Two-wheeler (rear-only plate rule) |
| `5` | `Bus` | Bus / coach |
| `7` | `Truck` | Truck / commercial vehicle |

### 2.4 OCR Reader Architectures (auto-detected from checkpoint)

| Architecture Key | Model | Input Size | Vocab |
|-----------------|-------|------------|-------|
| `vision_attention` | VisionAttentionOCR (GRCNN + BiLSTM + Attention) | `[1, 1, 64, 600]` grayscale | 187 tokens |
| `selen_lpr` | SelenLPR (MobileNetV3 + BiLSTM CTC) | `[1, 3, H, W]` RGB | DLT chars |
| `selen_lpr_e` | SelenLPR-E (EasyOCR Foundation + CTC) | `[1, 3, 64, 256]` RGB | 53 tokens |
| `paddleocr` | SelenPaddleOCR (MobileNetV3 + CTC) | `[1, 3, 48, 320]` RGB | 145 tokens |

---

## 3. Model Catalog Summary

| Model | File | Size | mAP / Accuracy |
|-------|------|------|----------------|
| Plate Detector | `models/detector/vehicle_plate_yolov11n.pt` | 5.5 MB | mAP@50: **99.36%** |
| Vehicle Detector | `models/detector/yolov8s.pt` | 22.5 MB | COCO standard |
| Thai Plate Reader (primary) | `models/reader/thai_plate_reader.pth` | 57.9 MB | Exact match: **100%** (15/15 real) |
| SelenLPR v0.0.1 | `models/reader/selen-lpr-0.0.1/best_reader.pth` | 20.8 MB | Province: **93.8%**, Type: **99.6%** |
| SelenOCR v0.1.1 | `models/reader/selen_ocr_v011/best_reader.pth` | 61.6 MB | Real CCTV: **96.70%** |

> [!NOTE]
> `PlateReader` auto-detects the checkpoint architecture — no manual arch selection needed.

---

## 4. I/O Contract — Detector

### Input

```python
# Any BGR NumPy array (OpenCV native format)
frame: np.ndarray  # shape (H, W, 3), dtype uint8

# Or a cropped vehicle region (for coarse-to-fine)
crop: np.ndarray   # shape (crop_H, crop_W, 3), dtype uint8
```

### YOLO Call Signature (internal)

```python
results = yolo_model(
    frame,
    conf=0.25,     # plate confidence threshold
    imgsz=640,     # coarse-to-fine mode
    # imgsz=960    # direct mode (larger context)
    verbose=False,
    device="cpu",  # or "cuda:0"
)[0]
```

### Output (per detection box)

```python
for b in results.boxes:
    cls_id:  int   = int(b.cls[0])         # 0=Front, 1=LicensePlate, 2=Rear
    conf:    float = float(b.conf[0])       # YOLO confidence
    box_abs: list  = b.xyxy[0].tolist()    # [x1, y1, x2, y2] in crop pixels

    # After offset correction to full-frame coords:
    global_box = [box_abs[0]+offset_x, box_abs[1]+offset_y,
                  box_abs[2]+offset_x, box_abs[3]+offset_y]
```

### Deduplication

Plates with IoU > 0.50 against an already-kept box are discarded (NMS-like, confidence-sorted).

---

## 5. I/O Contract — OCR Reader

### Input

```python
# Step 1: Extract crop from frame
crop_bgr: np.ndarray  # shape (H, W, 3) BGR uint8

# Step 2: Convert to PIL RGB
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
pil_crop = Image.fromarray(crop_rgb)   # PIL.Image in RGB mode

# Step 3: Call predict
result = reader.predict(pil_crop)
```

### Internal Preprocessing (NormalizePAD — VisionAttentionOCR)

```
1. Convert to grayscale (L mode)
2. Compute aspect ratio:  ratio = W / H
3. Resize to: (min(ceil(64 * ratio), 600), 64)   → maintains aspect ratio
4. ToTensor() → normalize: (pixel - 0.5) / 0.5   → range [-1, 1]
5. Zero-pad right edge to fixed width 600
6. Edge-replicate (last column) fills remaining padding
7. Final tensor shape: [1, 1, 64, 600]
```

### Output Dictionary

```python
result: dict = {
    "d3":         str,            # Thai prefix chars (e.g. "1กข", "กข", "ปข")
    "d4":         str,            # Digit group (e.g. "1234", "56", "9999")
    "province":   str,            # Thai province NFC-normalised (e.g. "ชลบุรี")
    "full_plate": str,            # "{d3} {d4} {province}".strip()
    "confidence": float,          # Mean softmax probability 0.0–1.0
    "tokens":     List[int],      # Raw argmax token indices ([] for CTC arches)
    "error":      Optional[str],  # None on success; error string on failure
}
```

### Token Decoding (VisionAttentionOCR — 187 classes)

```
Sequence: EOS token (index 1) terminates reading
Segments: Split on delimiter token 60 ('_') → [d3_tokens, d4_tokens, province_tokens]
Mapping:  EASYOCR_DECODING_LIST[token_idx] → character or full province name
```

**Key token ranges:**

| Indices | Content |
|---------|---------|
| 0 | `[GO]` start |
| 1 | `[s]` EOS |
| 2–11 | Digits `0–9` |
| 23+ | Thai consonants (41 DLT chars) |
| 60 | `_` delimiter |
| 86–161 | Full province names (single-token) |

### Province Normalisation

```python
# Performed internally after decode:
province = match_province_value(province_raw)
# Strips "ไม่ทราบจังหวัด" → ""
# Fuzzy-matches partial province text
# Returns NFC-normalised canonical Thai name
```

---

## 6. SelenCORE Adapter API

The `selencore/` package is the **only integration boundary** SelenCORE should reference.

### 6.1 Package Location

```
d:\Projects\Synapes\lpr\selencore\
├── __init__.py          # Public exports: LPREngine, LPRResult, Detection, PlateReading
└── lpr_engine.py        # Implementation
```

### 6.2 Data Classes

```python
@dataclass
class PlateReading:
    d3: str                     # Consonant prefix / leading digit group
    d4: str                     # Registration number digits
    province: str               # Thai province name (NFC)
    province_iso: str           # ISO 3166-2:TH code  e.g. "TH-20"
    full_plate: str             # Human-readable "d3 d4 province"
    confidence: float           # 0.0 – 1.0
    tokens: List[int]           # Raw attention tokens (may be [])
    error: Optional[str]        # None on success

@dataclass
class Detection:
    box: List[float]            # [x1, y1, x2, y2] full-frame pixel coords
    det_confidence: float       # YOLO plate detection confidence
    vehicle_type: str           # "Car" | "Motorcycle" | "Truck" | "Bus" | "Unknown"
    orientation: str            # "Front" | "Rear" | "Unknown"
    reading: PlateReading
    crop_bgr: Optional[np.ndarray]  # Plate pixel crop (BGR)
    track_id: int               # -1 if tracking disabled

    # Convenience properties:
    plate_text: str             # == reading.full_plate
    confidence: float           # == reading.confidence

@dataclass
class LPRResult:
    detections: List[Detection]
    frame_index: int            # Monotonic counter since engine init
    timestamp_iso: str          # ISO 8601 UTC  e.g. "2026-11-10T04:35:00.123456+00:00"
    det_latency_ms: float       # Detector wall-clock time
    ocr_latency_ms: float       # OCR wall-clock time
    total_latency_ms: float     # Full pipeline time
    source_shape: Tuple[int,int] # (height, width) of input frame

    has_plates: bool            # len(detections) > 0
    to_dict() -> dict           # JSON-serialisable output
```

### 6.3 LPREngine Methods

```python
class LPREngine:

    # --- Constructors ---
    @classmethod
    def from_default_models(**kwargs) -> LPREngine
        """Auto-discovers models from models/ directory."""

    @classmethod
    def from_paths(detector_path, reader_path, vehicle_detector_path=None, **kwargs) -> LPREngine
        """Explicit model paths."""

    # --- Core ---
    def process_frame(
        frame_bgr: np.ndarray,
        roi: Optional[List[int]] = None,       # [x1, y1, x2, y2]
        run_ocr: bool = True,
        coarse_to_fine: bool = True,
    ) -> LPRResult

    # --- Convenience ---
    def process_image_file(image_path: str | Path) -> LPRResult
    def process_pil_image(pil_image: PIL.Image) -> LPRResult
    def reset_motion_model() -> None           # Call on camera source switch

    # --- Properties ---
    frame_count: int   # Total frames processed
```

---

## 7. Quick-Start Integration

### 7.1 Minimal — Single Frame

```python
import cv2
from selencore import LPREngine

# One-time initialisation (expensive — do this once at startup)
engine = LPREngine.from_default_models(device="cpu")

# Per-frame call (cheap)
frame = cv2.imread("test_plate.jpg")
result = engine.process_frame(frame)

for det in result.detections:
    print(f"Plate: {det.plate_text}")
    print(f"  Province ISO: {det.reading.province_iso}")
    print(f"  OCR confidence: {det.confidence:.2%}")
    print(f"  Box: {det.box}")
    print(f"  Vehicle: {det.vehicle_type} / {det.orientation}")
    print(f"  Latency: {result.total_latency_ms:.1f} ms")
```

### 7.2 Video Stream Loop

```python
import cv2
from selencore import LPREngine

engine = LPREngine.from_default_models()
cap = cv2.VideoCapture("rtsp://camera_ip/stream")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = engine.process_frame(frame)

    if result.has_plates:
        # Emit to SelenCORE event bus
        payload = result.to_dict()
        selencore_bus.emit("lpr.detection", payload)

cap.release()
```

### 7.3 With ROI (Gate / Entrance Zone)

```python
# Focus detection on the entrance lane only
ROI = [320, 100, 960, 700]   # [x1, y1, x2, y2]

result = engine.process_frame(frame, roi=ROI)
```

### 7.4 Detector-Only (Skip OCR for Speed)

```python
# Use when you only need plate bounding boxes (e.g. for counting)
result = engine.process_frame(frame, run_ocr=False)
for det in result.detections:
    print(f"Plate at {det.box} conf={det.det_confidence:.2%}")
```

### 7.5 Explicit Model Paths

```python
engine = LPREngine.from_paths(
    detector_path="models/detector/vehicle_plate_yolov11n.pt",
    reader_path="models/reader/selen_ocr_v011/best_reader.pth",
    vehicle_detector_path="models/detector/yolov8s.pt",
    device="cuda:0",
    conf_thresh=0.30,
)
```

---

## 8. Advanced: Per-Component Direct Call

For integrations that need fine-grained control over individual pipeline stages:

### 8.1 Plate Detector Direct

```python
from tools.alpr.detector import detect_plates_direct
from ultralytics import YOLO

detector = YOLO("models/detector/vehicle_plate_yolov11n.pt")

plates, orientations = detect_plates_direct(
    detector=detector,
    det_frame=frame_bgr,       # np.ndarray BGR
    offset_x=0,                # int — pixel offset if using ROI crop
    offset_y=0,
    conf_thresh=0.25,
    device="cpu",
)

# plates:       List[Tuple[List[float], float]]
#               Each: ([x1, y1, x2, y2], confidence)

# orientations: List[Tuple[List[float], str, float]]
#               Each: ([x1, y1, x2, y2], "Front"|"Rear", confidence)
```

### 8.2 Coarse-to-Fine (Vehicle Cascade)

```python
from tools.alpr.detector import detect_vehicles, detect_plates_coarse_to_fine

vehicles = detect_vehicles(
    veh_detector=yolov8s,
    det_frame=frame_bgr,
    offset_x=0, offset_y=0,
    conf_thresh=0.35,
    device="cpu",
)
# vehicles: List[Tuple[List[float], str, float]]
#           Each: ([x1,y1,x2,y2], "Car"|"Motorcycle"|..., confidence)

plates, orientations = detect_plates_coarse_to_fine(
    detector=yolov11n,
    frame=frame_bgr,
    vehicles=vehicles,
    orig_w=frame_bgr.shape[1],
    orig_h=frame_bgr.shape[0],
    conf_thresh=0.25,
    device="cpu",
)
```

### 8.3 Vehicle + Orientation Spatial Match

```python
from tools.alpr.detector import match_vehicle_and_orientation

vehicle_type, orientation = match_vehicle_and_orientation(
    plate_box=[x1, y1, x2, y2],   # plate bounding box
    orientations=orientations,      # from detector output
    vehicles=vehicles,              # from vehicle detector
)
# Returns: ("Car"|"Motorcycle"|..., "Front"|"Rear"|"Unknown")
# Note: Motorcycles always get "Rear" if orientation is "Unknown" (DLT rule)
```

### 8.4 OCR Reader Direct

```python
from tools.DataManipulator.app.plate_reader import PlateReader
from PIL import Image
import cv2

reader = PlateReader.get_instance("models/reader/thai_plate_reader.pth")

# Input must be a PIL Image (RGB) or file path
crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
pil_img  = Image.fromarray(crop_rgb)

result = reader.predict(pil_img)
# result["d3"]         → "1กข"
# result["d4"]         → "1234"
# result["province"]   → "ชลบุรี"
# result["full_plate"] → "1กข 1234 ชลบุรี"
# result["confidence"] → 0.9123
# result["tokens"]     → [23, 60, 25, ...] (empty for CTC arches)
# result["error"]      → None
```

### 8.5 Province ISO Lookup

```python
from tools.DataManipulator.app.provinces import get_iso_code, convert_to_dlt_international_plate

iso = get_iso_code("ชลบุรี")
# → "TH-20"

intl = convert_to_dlt_international_plate("1กข", "1234", "ชลบุรี")
# → "1AB_1234_TH-20"  (DLT International + ISO format, ASCII-safe for filenames)
```

---

## 9. Output Schema Reference

### 9.1 `LPRResult.to_dict()` — JSON Wire Format

```json
{
  "frame_index": 42,
  "timestamp_iso": "2026-11-10T04:35:22.123456+00:00",
  "det_latency_ms": 18.4,
  "ocr_latency_ms": 7.2,
  "total_latency_ms": 26.1,
  "source_shape": [1080, 1920],
  "detections": [
    {
      "track_id": -1,
      "box": [312.0, 540.5, 498.3, 592.1],
      "det_confidence": 0.9341,
      "vehicle_type": "Car",
      "orientation": "Rear",
      "plate_text": "1กข 1234 ชลบุรี",
      "d3": "1กข",
      "d4": "1234",
      "province": "ชลบุรี",
      "province_iso": "TH-20",
      "ocr_confidence": 0.9123,
      "ocr_error": null
    }
  ]
}
```

### 9.2 Field Glossary

| Field | Type | Description |
|-------|------|-------------|
| `d3` | `str` | Thai consonant prefix + optional leading digit. Max 3 chars. |
| `d4` | `str` | Registration digit group. 1–4 digits. |
| `province` | `str` | Full Thai province name, NFC-normalised Unicode. |
| `province_iso` | `str` | ISO 3166-2:TH code. Empty string if province unknown. |
| `full_plate` | `str` | `"{d3} {d4} {province}".strip()` |
| `ocr_confidence` | `float` | Mean softmax of non-trivial tokens. 0.0–1.0. |
| `vehicle_type` | `str` | `Car` / `Motorcycle` / `Truck` / `Bus` / `Unknown` |
| `orientation` | `str` | `Front` / `Rear` / `Unknown` |
| `box` | `[f,f,f,f]` | `[x1, y1, x2, y2]` pixel coordinates in original frame |

---

## 10. Thai Plate Domain Rules

> [!IMPORTANT]
> These rules are enforced by post-processing — do NOT bypass them in SelenCORE.

| Rule | Detail |
|------|--------|
| **Valid consonants** | 41 of 44 Thai consonants: `กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ` |
| **Excluded consonants** | `ฃ` → remapped to `ข`, `ฅ` → `ค`, `ฏ` → `ฎ` |
| **No vowels/tone marks** | Standard DLT plates never contain diacritics |
| **Motorcycle orientation** | Motorcycles are **always Rear** — no front plates exist |
| **d3 max length** | 3 characters (SelenLPR enforces DLT clamping) |
| **d4 max length** | 4 digits (SelenLPR enforces DLT clamping) |
| **Excluded word combos** | `ศพ, สพ, ศบ, ษพ, ชน, จน, ตด, งง` |
| **Province count** | 77 provinces + Betong district = 78 total + "unknown" |

---

## 11. Configuration & Thresholds

| Parameter | Default | Effect |
|-----------|---------|--------|
| `conf_thresh` | `0.25` | YOLO plate detection threshold. Lower → more detections, more false positives |
| `veh_conf_thresh` | `0.35` | Vehicle detection threshold |
| `min_plate_width_px` | `14` | Minimum plate pixel width for OCR |
| `min_plate_height_px` | `8` | Minimum plate pixel height for OCR |
| `enable_motion_filter` | `True` | MOG2 background subtraction pre-filter |
| `coarse_to_fine` | `True` | Use vehicle cascade (requires vehicle detector) |
| `imgsz` (coarse-to-fine) | `640` | YOLO input resolution for vehicle-crop inference |
| `imgsz` (direct mode) | `960` | Larger context for direct full-frame inference |

---

## 12. Error Handling

```python
result = engine.process_frame(frame)

for det in result.detections:
    if det.reading.error:
        # OCR failed for this plate (e.g. corrupt crop, model mismatch)
        logger.warning(f"OCR error on track {det.track_id}: {det.reading.error}")
        # det.box and det.det_confidence are still valid
        continue

    if not det.plate_text:
        # Model ran but produced empty output — low-confidence or unreadable plate
        continue

    # Normal path
    process(det)
```

> [!WARNING]
> `result` is **always returned** even on empty frames or when no plates are found. Never let `process_frame()` raise — wrap the whole engine initialisation in a try/except at startup.

```python
try:
    engine = LPREngine.from_default_models()
except FileNotFoundError as e:
    logger.critical(f"Model weights missing: {e}")
    sys.exit(1)
except RuntimeError as e:
    logger.critical(f"Model load error: {e}")
    sys.exit(1)
```

---

## 13. Thread Safety & Concurrency

> [!CAUTION]
> `LPREngine` is **NOT thread-safe**. Each camera stream / worker thread must have its own engine instance.

```python
# CORRECT — one engine per thread
def camera_worker(cam_id: int, model_paths: dict):
    engine = LPREngine.from_paths(**model_paths)
    while True:
        frame = get_frame(cam_id)
        result = engine.process_frame(frame)
        queue.put(result.to_dict())

# WRONG — do not share one engine across threads
shared_engine = LPREngine.from_default_models()
Thread(target=worker, args=(shared_engine,)).start()  # race condition on model state
```

**Motion Model Reset** — when switching camera sources:
```python
engine.reset_motion_model()  # Clears MOG2 background history
```

---

## 14. Changelog Entry

```
### [CR-20261110-02] - SelenCORE LPR Integration Package
- Date (UTC): 2026-11-10T04:37:00Z
- Component: Integration / API Layer
- Change Type: feature
- ISO 42001 Clause: Annex A.9.3 (Operational Event Logging), Clause 8.4 (AI System Development)
- Objective: Isolate the LPR detector + reader pipeline into a stable
  selencore/ adapter package for embedding into the SelenCORE platform
  without access to the internal demo UI code.
- Risk Assessment: Adapter wraps internal helpers — no model weights modified,
  no inference logic changed. Singleton PlateReader preserved.
- Verification Gate: process_frame() tested against test_samples/ with known plates.
- Traceability: selencore/__init__.py, selencore/lpr_engine.py,
  docs/SELENCORE_LPR_MANUAL.md
```
