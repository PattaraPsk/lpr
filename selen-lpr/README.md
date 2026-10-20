# Selen-LPR: Commercial Thai License Plate Recognition & Vehicle Classification

Pure PyTorch, production-grade Computer Vision pipeline for Thai License Plate Recognition (ALPR/ANPR) and vehicle classification across all 8 Department of Land Transport (DLT) categories, plus cross-border logistics support.

```
selen-lpr/
├── configs/
│   ├── categories.yaml           # Definitions for all 8 DLT vehicle types, colors, & legal consonants
│   └── detector_config.yaml      # Confidence thresholds, adaptive ROI settings, and input resolution
│
├── models/
│   ├── detector/                 # Standalone TorchScript/ONNX detector weights
│   │   └── README.md
│   └── ocr/                      # Standalone Pure PyTorch Vision-Attention weights
│       ├── thai_plate_attn_ocr.pt # Converted standalone checkpoint (55.25 MB)
│       └── README.md
│
├── src/
│   ├── models/                   # PURE PYTORCH ARCHITECTURES (Zero EasyOCR / Zero YOLO packages)
│   │   ├── vision_attention_ocr.py # Standalone GRCNN + BiLSTM + Attention Decoder
│   │   ├── plate_detector.py     # Standalone TorchScript / torchvision.ops.nms detector
│   │   └── vocab.py              # Thai alphanumeric + 80 discrete province token decoder
│   │
│   ├── core/
│   │   ├── detector.py           # Adaptive vehicle ROI extractor & coordinate normalizer
│   │   ├── design_classifier.py  # CIE LAB plate palette classifier (White/Yellow/Orange/Green)
│   │   ├── vehicle_mapper.py     # Maps plate design + letter prefix to vehicle type
│   │   └── ocr_reader.py         # Standardized OCR reader interface
│   │
│   ├── schema/
│   │   └── types.py              # NormalizedCoordinates (s_x,s_y,s_w,s_h), DetectionResult
│   │
│   ├── utils/
│   │   ├── color_extractor.py    # Color clustering in LAB/HSV space
│   │   └── dlt_rules.py          # DLT legal rules (e.g., ฒ -> pickup, น -> van, ท -> taxi)
│   │
│   └── pipeline/
│       └── adaptive_pipeline.py  # Master end-to-end execution pipeline
│
├── tools/                        # SEPARATED: Reusable Code Connectors & Weight Converters
│   ├── transfer_ocr_weights.py   # Extracts & migrates EasyOCR weights into pure PyTorch
│   └── export_detector.py        # Converts YOLO weights to TorchScript/ONNX
│
├── interfaces/
│   ├── io_spec.json              # Standardized API contract
│   └── service.py                # Standalone callable service / CLI entrypoint
│
├── samples/                      # Verified test samples for all 8 categories
│   ├── 01_private_car/sample.png
│   ├── 02_private_pickup/sample.png
│   ├── 03_private_van/sample.png
│   ├── 04_public_taxi/sample.png
│   ├── 05_tuktuk/sample.png
│   ├── 06_trailer_tractor/sample.png
│   ├── 07_service_vehicle/sample.png
│   └── 08_motorcycle/sample.png
│
├── tests/
│   ├── test_detection.py         # Coordinate normalization bounds [0.0, 1.0]
│   ├── test_classification.py    # DLT legal rules verification
│   └── test_ocr_model.py         # Pure PyTorch OCR forward pass & token decoding
│
├── requirements.txt              # Strictly torch, torchvision, opencv-python, numpy, pyyaml
├── run.bat                       # Windows execution launcher
└── README.md                     # This documentation
```

---

## 1. Zero 3rd-Party Library Lock-In

`selen-lpr` is built with **zero dependency on `easyocr`** and **zero dependency on `ultralytics`**:
- **Detector**: Runs native TorchScript / ONNX with `torchvision.ops.nms`.
- **OCR Engine**: Pure PyTorch implementation of ClovaAI Deep Text Recognition (`RCNN + BiLSTM + Attention Decoder`).
- **Weight Transfer**: Dedicated migration script in `tools/transfer_ocr_weights.py` extracts weights from previous models with 100% parameter compatibility, completely separating past freelance code from this new commercial deployment.

---

## 2. Quick Execution

```powershell
cd selen-lpr
# Run service on private car sample
python interfaces/service.py --image samples/01_private_car/sample.png

# Run service on pickup truck sample
python interfaces/service.py --image samples/02_private_pickup/sample.png

# Run service on public taxi sample
python interfaces/service.py --image samples/04_public_taxi/sample.png
```

---

## 3. Standardized Output Payload (`interfaces/io_spec.json`)

```json
{
  "source": "samples/02_private_pickup/sample.png",
  "detections_count": 1,
  "detections": [
    {
      "s_x": 0.0,
      "s_y": 0.0,
      "s_w": 1.0,
      "s_h": 1.0,
      "d3": "1กข",
      "d4": "8920",
      "province": "กรุงเทพมหานคร",
      "vehicle_category": "private_pickup",
      "vehicle_description": "Private Pickups (2-door light commercial trucks)",
      "background_color": "white",
      "text_color": "green",
      "confidence": 0.95
    }
  ]
}
```
