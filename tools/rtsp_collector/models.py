"""
Data Models and Schemas (ISO/IEC 25012 & ISO 8601 Compliant)
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


def format_iso8601_duration(seconds: float) -> str:
    """Formats duration seconds into ISO 8601 duration representation (e.g. PT15M30S)."""
    if seconds <= 0:
        return "PT0S"
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60
    millis = int(round((seconds - total_sec) * 1000))
    
    parts = ["P"]
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}H")
    if minutes > 0:
        time_parts.append(f"{minutes}M")
    if sec > 0 or millis > 0 or not time_parts:
        if millis > 0:
            time_parts.append(f"{sec}.{millis:03d}S")
        else:
            time_parts.append(f"{sec}S")
    
    return "PT" + "".join(time_parts)


@dataclass
class StreamProfile:
    """Probed metadata for an active RTSP video stream."""
    codec_name: str = "unknown"
    profile: str = "unknown"
    width: int = 0
    height: int = 0
    pix_fmt: str = "unknown"
    fps: float = 0.0
    color_space: str = "unknown"
    server_description: str = "unknown"
    raw_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QualityMetrics:
    """ISO/IEC 25012 Data Quality & Verification Metrics."""
    sha256: str = ""
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    frame_count: int = 0
    codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    is_valid: bool = False
    validation_status: str = "PENDING"  # PASSED, DEGRADED, CORRUPTED
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SegmentManifest:
    """
    Standardized Ingestion Metadata & Lineage Record.
    Adheres strictly to ISO 8601 and ISO/IEC 25012:2008.
    """
    file_name: str
    relative_path: str
    camera_id: str
    channel_id: str
    location_tag: str
    
    start_time_iso_utc: str
    end_time_iso_utc: str
    start_time_iso_local: str
    end_time_iso_local: str
    
    duration_seconds: float
    duration_iso8601: str
    file_size_bytes: int
    sha256_checksum: str
    
    video_codec: str
    resolution: str
    fps: float
    total_frames: int
    
    source_url_masked: str
    quality_status: str  # PASSED, DEGRADED, CORRUPTED
    validation_notes: Optional[str] = None
    
    collector_version: str = "1.0.0"
    created_at_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    standards_compliance: List[str] = field(
        default_factory=lambda: [
            "ISO 8601:2019 (Date and Time Representation)",
            "ISO/IEC 14496-14 (MP4 File Format)",
            "ISO/IEC 25012:2008 (Data Quality Model - Integrity & Completeness)",
            "ISO/IEC 27001:2022 (Information Security - Credential Decoupling)",
            "ISO/IEC 20547-3 (Big Data Reference Architecture - Ingestion Layer)"
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
