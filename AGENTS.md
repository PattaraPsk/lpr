# Thai LPR Edge Project Context & Rules

## Project Overview
- Objective: Low-latency, edge-optimized Thai License Plate Recognition (ALPR/ANPR).
- Target Runtime: CPU Edge / Workstation inference (Python 3.10).
- Core Pipeline:
  1. Vehicle / Plate Detection (YOLO / MobileNet detector)
  2. Perspective Rectification / Preprocessing
  3. Character Recognition & Thai OCR (Character detection + Province classification)
  4. Post-processing & Regex Validation (Thai format: หมวดอักษร + ตัวเลข + จังหวัด)

## Engineering Guardrails & Non-Negotiables
- Edge Performance: Maintain low latency and minimal memory overhead. Do not introduce heavy dependencies or cloud API calls without approval.
- Concurrency & Streams: Ensure thread safety on RTSP/video stream ingestion and frame-dropping logic.
- Model Artifacts: Never commit or alter raw model weights (`.onnx`, `.engine`, `.pt`, `.xml`).
- Environment: Target Python 3.10.x compatibility; do not break Windows `.bat` execution scripts or Linux service daemons.
- Testing: Validate inference accuracy and throughput on local sample test frames before marking tasks complete.