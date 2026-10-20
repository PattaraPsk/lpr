# International & Enterprise Standards Compliance Manual

This repository conforms to international engineering standards, ISO/IEC specifications, and AI governance frameworks to ensure production-grade edge reliability, forensic data integrity, and cross-system interoperability.

---

## Standards Compliance Matrix

| Standard | Domain / Scope | Implementation Location | Compliance Mechanism |
| :--- | :--- | :--- | :--- |
| **ISO 3166-2:TH** | Geographic Subdivision Codes (Thailand) | `tools/DataManipulator/app/provinces.py`<br>`tools/DataManipulator/app/session_manager.py` | Full mapping of all 77 Thai provinces to official `TH-XX` codes (e.g. `TH-10` Bangkok, `TH-50` Chiang Mai). Serialized in session files and export schemas. |
| **ISO/IEC 10646 / Unicode 15.0** | Universal Coded Character Set (NFC) | `tools/DataManipulator/app/provinces.py`<br>`tools/DataManipulator/app/session_manager.py`<br>`tools/owod/core/owod_engine.py` | Canonical composition (`unicodedata.normalize('NFC', text)`) applied at all ingestion, matching, and serialization points to eliminate vowel/tone mark decomposition anomalies. |
| **ISO 8601:2019** | Date, Time & Duration Representation | `tools/detect_cam_videos.py`<br>`tools/rtsp_collector/`<br>`tools/owod/core/quality_auditor.py`<br>`tools/DataManipulator/app/session_manager.py` | UTC timestamps in ISO 8601 extended format (`YYYY-MM-DDTHH:MM:SSZ`) and durations formatted according to ISO 8601 duration notation (e.g., `PT1M30S`). |
| **ISO/IEC 25012:2008** | Data Quality Model (Data Integrity) | `tools/detect_cam_videos.py`<br>`tools/rtsp_collector/manifest.py`<br>`tools/owod/core/quality_auditor.py`<br>`tools/DataManipulator/app/session_manager.py` | Chunked SHA-256 cryptographic hashing of source videos, bounding box label sets, and configuration files. Atomic write-and-replace (`.tmp`) pattern for state files. |
| **ISO/IEC 5259-2:2023** | AI Data Quality Measures for ML | `tools/owod/core/quality_auditor.py`<br>`tools/owod/build_dlt_lpr_dataset.py`<br>`tools/owod/core/dataset_generator.py` | Automated quality auditing producing `data_quality_report.json`: bounding box syntactic containment ($0.0 \le x_c, y_c, w, h \le 1.0$), zero-area detection, split ratios, and class imbalance metrics. |
| **ISO/IEC 27001:2022** | Information Security & Privacy Protection | `tools/rtsp_collector/security.py`<br>`tools/detect_cam_videos.py` | Automatic masking of RTSP streaming credentials (`admin:***@host`) and sanitization of user directory paths (`.../Users/*** /...`) across logs and trackers. |
| **ISO/IEC 14496-14 / ITU-T H.264** | MP4 Video File Format Integrity | `tools/rtsp_collector/validator.py`<br>`tools/detect_cam_videos.py` | Automated container atom validation, moov-atom integrity checks, frame sampling, and edge stream ingestion dropping logic. |
| **ISO/IEC 42001:2023** | AI Management System (Model Governance) | `models/MODEL_CATALOG.md`<br>`.gitignore`<br>`AGENTS.md`<br>`tests/test_iso_compliance.py` | Strict repository governance prohibiting raw model binary commits (`.pt`, `.onnx`, `.engine`, `.xml`). Standardized model directory structure, verifiable model architecture cards (`MODEL_CATALOG.md`), and deterministic random seeding (`RANDOM_SEED = 42`). |
| **ISO/IEC 25010:2011** | Software Quality & Maintainability | All modules in `tools/` | Structured logging with standard timestamps and log levels, decoupled architectural layers (`app/`, `core/`, `ui/`), and thread-safe ingestion workers. |
| **ISO/IEC/IEEE 29119** | Software Testing & Quality Assurance | `tests/test_iso_compliance.py`<br>`tools/DataManipulator/test_app.py`<br>`run_tests.bat` | Automated test suite verifying ISO 3166-2:TH lookups, Unicode NFC normalization, ISO 8601 durations, ISO 25012 SHA-256 integrity, ISO 27001 masking, and ISO 5259 dataset auditing. |
| **ISO/IEC 19770 & PEP 517/621** | Software Packaging & Identification | `pyproject.toml`<br>`requirements.txt` | Standardized Python build specification and reproducible edge runtime dependency definitions. |

---

## Detailed Specifications

### 1. ISO 3166-2:TH (Thai Province Standard)
- All 76 provinces and the Bangkok Special Administrative Area are indexed with their ISO 3166-2 code (`TH-10` to `TH-96`).
- Standardized lookup table in `tools/DataManipulator/app/provinces.py`:
  - `get_iso_code(province_name: str) -> Optional[str]`
  - `get_province_by_iso(code: str) -> Optional[Tuple[str, str]]`
  - `match_province_value(text: str) -> str`
- Annotation exports output both the canonical Thai name and the ISO 3166-2 code for cross-database interoperability.

### 2. ISO/IEC 10646 & Unicode 15.0 NFC Normalization
Thai script can be represented in multiple byte combinations:
- Precomposed vowel combinations vs separate vowels (e.g. `เ` + `เ` vs `แ`).
- Differing ordering of upper tone marks and floating vowels.
All text strings are normalized via:
```python
import unicodedata
text_normalized = unicodedata.normalize('NFC', raw_text.strip())
```

### 3. ISO/IEC 5259-2:2023 AI Data Quality Auditing
Datasets built via `tools/owod/build_dlt_lpr_dataset.py` or `tools/owod/core/dataset_generator.py` generate an automated `data_quality_report.json` with the following schema:
```json
{
  "standard_compliance": {
    "data_quality_standard": "ISO/IEC 5259-2:2023",
    "quality_model_standard": "ISO/IEC 25012:2008",
    "timestamp_standard": "ISO 8601",
    "integrity_standard": "FIPS 180-4 / ISO 27001"
  },
  "audit_metadata": {
    "dataset_root": "dataset/yolo_dlt_lpr_dataset",
    "audit_timestamp_utc": "2026-10-20T03:40:00Z",
    "config_file": "data.yaml",
    "config_sha256": "<hash>"
  },
  "volume_and_split": {
    "total_images": 1250,
    "total_annotations": 1820,
    "splits": { ... },
    "train_val_image_ratio": 0.15
  },
  "quality_measures": {
    "syntactic_validity_rate": 1.0,
    "valid_annotations": 1820,
    "invalid_annotations": 0,
    "out_of_bounds_boxes": 0,
    "degraded_zero_area_boxes": 0,
    "quality_grade": "A (PASSED)"
  },
  "representation_and_bias": {
    "class_counts": { ... },
    "class_proportions": { ... },
    "imbalance_ratio": 1.45
  }
}
```

### 4. Running Compliance Verification
To verify compliance across all modules on Windows or Linux:
```cmd
run_tests.bat
```
Or directly via Python:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
