# Thai LPR Documentation Index

Welcome to the technical documentation library for the Thai License Plate Recognition (ALPR/ANPR) Edge System.

---

## 📚 Core Documents

1. **[Development Process with Antigravity (Step-by-Step Guide)](ANTIGRAVITY_DEV_PROCESS.md)**
   - Practical engineering workflow, 6-phase development lifecycle, subagent orchestration (`research`, `self`), slash commands (`/plan`, `/boost`, `/teamwork`), and edge verification playbooks.

2. **[Labeling Process, Synthetic Data Generation & Standards Compliance](LABELING_AND_SYNTHETIC_DATA.md)**
   - End-to-end data engineering architecture, auto-crop detection, AI pre-annotation (VLM / SelenOCR), and human-in-the-loop review (`DataManipulator`).
   - Physics-calibrated parametric synthetic data engine, DLT font suites, 3D perspective distortion, and the 70:30 Golden Ratio curriculum (170k corpus).
   - Complete International & Enterprise Standards Compliance Matrix (ISO/IEC 42001, 5259, 25012, 3166-2:TH, 10646 NFC, 8601, 27001, 14496-14, 29119, and DLT specifications).

3. **[Repository Standards Manual](../STANDARDS.md)**
   - Enterprise compliance manual and specifications across ISO standards.

4. **[Model Catalog](../models/MODEL_CATALOG.md)**
   - Neural network model cards, input/output tensors, inference latencies, accuracy benchmarks, and SHA-256 weight hashes.

5. **[Engineering Guardrails & Rules](../AGENTS.md)**
   - Non-negotiables for low-latency edge inference, thread safety, and change governance.

6. **[AI Management System Change Log](../CHANGELOG.md)**
   - Auditable change log conforming to ISO/IEC 42001:2023.
