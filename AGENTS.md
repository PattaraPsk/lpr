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
- ISO/IEC 42001:2023 AI Management & Change Logging Mandate:
  - **All Changes Must Be Logged**: Any modification to models, datasets, inference pipelines, post-processing logic, or configuration must be recorded in `CHANGELOG.md` conforming to ISO/IEC 42001:2023 (Entry ID, Date UTC, Component, ISO 42001 Clause, Objective, Risk Assessment, Verification Gate).
  - **Operational Event Logging (Annex A.9.3)**: Runtime inference and streaming operations must output ISO 8601 UTC timestamps, active model version keys, confidence thresholds, and sanitized paths/credentials.
  - **Auditability & Traceability**: No undocumented changes or silent model replacements permitted. Every active model checkpoint must be documented in `models/MODEL_CATALOG.md`.

## Thai License Plate Domain Knowledge & DLT Specifications
- **Valid Character Set (41 Consonants):**
  - Standard DLT vehicle plates utilize 41 out of the 44 Thai consonants:
    `กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ`
  - Letters such as **`ซ`**, **`ห`**, and **`ป` are actively used** across passenger cars, commercial/pickup trucks, and motorcycles (e.g., `1กซ`, `1กป`, `1กห`, `ปข`, `หม`).
- **Strictly Excluded Consonants (Only 3):**
  - `ฃ` (Kho Khuat) — Obsolete character; not used in modern Thai orthography.
  - `ฅ` (Kho Khon) — Obsolete character; not used in modern Thai orthography.
  - `ฏ` (To Patak) — Excluded by DLT to eliminate optical confusion with `ฎ` (Do Chada).
- **Post-Processing & OCR Remapping:**
  - Standard OCR correction mappings: `{"ฃ": "ข", "ฅ": "ค", "ฏ": "ฎ"}`.
  - Standard plates **never** contain vowels, tone marks, or diacritics (except 2022+ vanity auction plates).
  - Excluded negative/offensive word combinations: `ศพ`, `สพ`, `ศบ`, `ษพ`, `ชน`, `จน`, `ตด`, `งง`.
