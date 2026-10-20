# Model Weights & Runtime Registry

This directory holds local inference weights and optimized runtime models (`.pt`, `.onnx`, `.engine`, OpenVINO IR `.xml`/`.bin`).

> **Note**: Raw weights are ignored from git version control via `.gitignore` to maintain a lightweight repository.

---

## 1. Supported Model Backends

| Model ID | Backend | Target Hardware | Recommended Role | Default Path |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO-World (`yolov8s-worldv2.pt`)** | PyTorch / ONNX | CUDA / CPU Edge | High-speed zero-shot plate localization | `models/yolov8s-worldv2.pt` |
| **YOLOv8 Nano/Small (`yolov8n.pt`, `yolov8s.pt`)** | PyTorch / ONNX | CPU Edge | Ultra-low latency edge vehicle & plate anchor | `models/yolov8s.pt` |
| **OWLv2 (`google/owlv2-base-patch16-ensemble`)** | HuggingFace | CUDA / Workstation | High-accuracy open-vocabulary detection | Auto-cached via HuggingFace |
| **Qwen2-VL (`Qwen/Qwen2-VL-2B-Instruct`)** | Transformers | RTX 4090 / CUDA | VLM classification & real vs. fake plate OCR | Auto-cached via HuggingFace |
| **Thai LPR ONNX Detector (`thai_plate_detector.onnx`)** | ONNX Runtime | CPU Edge / OpenVINO | CPU appliance inference ($<25\text{ms}$) | `models/thai_plate_detector.onnx` |

---

## 2. Exporting YOLO Models to ONNX for CPU Edge Inference

To export an Ultralytics model to an optimized ONNX runtime graph for CPU deployment:

```bash
python -c "from ultralytics import YOLO; model = YOLO('models/yolov8s-worldv2.pt'); model.export(format='onnx', imgsz=640, simplify=True, opset=17)"
```

This generates `models/yolov8s-worldv2.onnx` ready for CPU-only edge inference with ONNX Runtime.

---

## 3. Automated Model Initialization

The pipeline automatically detects weights placed in this directory or downloads pre-trained open weights on first execution when an internet connection is available.
