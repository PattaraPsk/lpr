# ISO-Standardized RTSP Stream Ingestion Pipeline

A production-grade, systematic Data Engineering pipeline for high-resolution CCTV/RTSP camera ingestion, purpose-built for the **Thai License Plate Reader (LPR)** and **Open-World Object Detection (OWOD)** ecosystem.

---

## 🏛️ International Standards Compliance Architecture

| Standard | Purpose & Implementation |
| :--- | :--- |
| **ISO 8601:2019** | **Date, Time & Duration Representation**<br>• Partitioned storage layout: `CAM01/YYYY/MM/DD/`<br>• Timestamp formatting: `CAM01_CH101_YYYY-MM-DDTHH-MM-SS+0700.mp4`<br>• Strict UTC/Local ISO timestamps in manifests and logs (`PT15M`, `PT6.010S`). |
| **ISO/IEC 14496-14** | **MPEG-4 Part 14 (MP4 File Format)**<br>• Remuxes H.264 video into standard ISO MP4 containers.<br>• Moves `moov` atom to file start via `+faststart` for zero-latency random access and instant streaming compatibility with PyTorch/OpenCV/YOLO readers.<br>• Zero transcode loss (`-c:v copy`). |
| **ISO/IEC 25012:2008** | **Data Quality Requirements and Evaluation**<br>• **Integrity**: Computes SHA-256 cryptographic hashes for every segment immediately upon closure.<br>• **Completeness & Accuracy**: Deep inspection via FFprobe for valid dimensions, framerates, PTS continuity, and container health. |
| **ISO/IEC 27001:2022** | **Information Security Management**<br>• Decouples camera passwords and endpoint URLs into `.env` (strictly `.gitignore` excluded).<br>• Enforces credential masking in all console outputs, logs, and metadata records (`rtsp://user:***@host:port/path`). |
| **ISO/IEC 20547-3** | **Big Data Reference Architecture (BDRA)**<br>• Staging buffer isolation (`.staging/`) prevents downstream processes from reading incomplete writes.<br>• Atomic promotion to Bronze Landing Zone.<br>• Immutable Lineage tracking via `manifest.jsonl` and sidecar `.json`. |
| **ISO 9001 / ISO/IEC 12207** | **Software Life Cycle & Operational Reliability**<br>• Real-time disk capacity protection (halts gracefully if free disk < safety limit).<br>• Automatic fault-tolerant reconnect with exponential backoff on stream disconnection. |

---

## 📁 Storage Hierarchy

```text
dataset/Raw Video Record/
├── manifest.jsonl                                      # Central append-only audit catalog
├── .staging/                                           # Temporary write-isolation buffer
├── .logs/
│   └── collector.log                                   # Structured operational log
└── CAM01/
    └── 2026/
        └── 10/
            └── 19/
                ├── CAM01_CH101_2026-10-19T13-47-42+0700.mp4   # Lossless H.264 MP4
                └── CAM01_CH101_2026-10-19T13-47-42+0700.json  # Metadata sidecar
```

---

## 🚀 Quick Start

### 1. Interactive Runner
From the repository root or `tools/rtsp_collector`:
```cmd
run.bat
```
Menu options:
- `[1]` Probe RTSP Stream Health (ISO 25012 Diagnostics)
- `[2]` Start Continuous Ingestion (15-min Segments, Lossless MP4)
- `[3]` Start Short Test Ingestion (60-second Test Capture)
- `[4]` Verify Storage Data Quality & SHA-256 Hashes
- `[5]` View Collector Storage Status & Statistics

---

### 2. Command-Line Interface (CLI)

#### A. Probe & Validate Stream Connection
```cmd
python tools/rtsp_collector/cli.py probe
```
*Outputs resolution, framerate, server description, pixel format, and ISO compliance verification.*

#### B. Continuous 24/7 Ingestion (Segmented)
```cmd
python tools/rtsp_collector/cli.py record
```
*Captures continuous 15-minute segments (customizable in `config.yaml` or `--segment 900`) with auto-reconnection and SHA-256 validation.*

#### C. Custom Bounded Capture
```cmd
# Record 1 hour in 10-minute segments:
python tools/rtsp_collector/cli.py record --duration 3600 --segment 600

# Quick 30-second test:
python tools/rtsp_collector/cli.py record --duration 30 --segment 30
```

#### D. Data Quality & Checksum Audit (ISO 25012)
```cmd
python tools/rtsp_collector/cli.py verify
```
*Verifies all files listed in `manifest.jsonl` against disk, recalculating cryptographic SHA-256 hashes to guarantee data integrity.*

#### E. System & Storage Status
```cmd
python tools/rtsp_collector/cli.py status
```
*Displays free disk space, total recorded chunks, volume in GB, and cumulative duration.*

---

## ⚙️ Configuration (`config.yaml`)

```yaml
stream:
  stream_url: "${RTSP_URL}"
  camera_id: "CAM01"
  channel_id: "101"
  location_tag: "Main_Entrance"
  transport: "tcp"              # Lossless TCP delivery
  socket_timeout_seconds: 10   # Disconnect detection

ingestion:
  segment_duration_seconds: 900 # 15 minutes per segment
  copy_codecs: true             # Lossless copy (-c:v copy)
  enable_faststart: true        # Places moov atom at beginning

storage:
  base_dir: "dataset/Raw Video Record"
  staging_dir: "dataset/Raw Video Record/.staging"
  partition_pattern: "{camera_id}/%Y/%m/%d"
  min_disk_free_gb: 5.0         # Safety disk exhaustion guard

quality_assurance:
  compute_sha256: true          # ISO 25012 integrity checksum
  verify_media_headers: true    # FFprobe stream header validation
  generate_sidecar_json: true   # Companion JSON metadata
  manifest_filename: "manifest.jsonl"
```
