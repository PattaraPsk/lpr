"""
src/pipeline/types.py
Data structures and typed models for the Thai LPR pipeline.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BoundingBox:
    """Represents a 2D bounding box with coordinates [xmin, ymin, xmax, ymax]."""
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float = 1.0
    label: str = "plate"

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1.0, float(self.height))

    @property
    def center(self) -> Tuple[int, int]:
        return (self.xmin + self.width // 2, self.ymin + self.height // 2)

    def to_xyxy(self) -> List[int]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    def to_xywh(self) -> List[int]:
        return [self.xmin, self.ymin, self.width, self.height]


@dataclass
class PlateReading:
    """Represents structured, formatted Thai license plate OCR output."""
    raw_text: str = ""
    consonants: str = ""
    digits: str = ""
    province: str = ""
    formatted_string: str = ""
    confidence: float = 0.0
    is_valid_dlt: bool = False
    plate_category: str = "private_car"  # "private_car", "pickup", "van", "taxi", "auction"

    def __str__(self) -> str:
        if self.formatted_string:
            return self.formatted_string
        parts = [p for p in [self.consonants, self.digits, self.province] if p]
        return " ".join(parts) if parts else "UNREADABLE"


@dataclass
class LPRResult:
    """Complete inference result for a detected license plate within a frame."""
    frame_id: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    bbox: BoundingBox = field(default_factory=lambda: BoundingBox(0, 0, 0, 0))
    reading: PlateReading = field(default_factory=PlateReading)
    track_id: Optional[int] = None
    plate_crop: Optional[np.ndarray] = None
    deskew_angle: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetrics:
    """Telemetry metrics for streaming and video pipelines."""
    total_frames: int = 0
    dropped_frames: int = 0
    plates_detected: int = 0
    valid_reads: int = 0
    avg_latency_ms: float = 0.0
    fps: float = 0.0
