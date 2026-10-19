"""
Data Quality & Integrity Validator (ISO/IEC 25012:2008 & ISO/IEC 25024:2015)
Performs cryptographic checksum verification, container validation, and frame consistency checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .models import QualityMetrics

logger = logging.getLogger("RTSPCollector.Validator")


def calculate_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash of a file for ISO 25012 data integrity."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class MediaValidator:
    """
    Validates recorded video media against ISO/IEC 14496 and ISO/IEC 25012 standards.
    """

    def __init__(self, ffprobe_path: str = "ffprobe"):
        self.ffprobe_path = ffprobe_path

    def validate_file(self, file_path: Path | str, expected_duration: Optional[float] = None) -> QualityMetrics:
        path = Path(file_path)
        metrics = QualityMetrics()

        if not path.is_file():
            metrics.validation_status = "CORRUPTED"
            metrics.error_message = f"File does not exist: {path}"
            return metrics

        metrics.file_size_bytes = path.stat().st_size
        if metrics.file_size_bytes == 0:
            metrics.validation_status = "CORRUPTED"
            metrics.error_message = "File is empty (0 bytes)"
            return metrics

        # 1. Compute Cryptographic SHA-256 Checksum (ISO 25012 Inherent Quality)
        try:
            metrics.sha256 = calculate_sha256(path)
        except Exception as e:
            metrics.validation_status = "CORRUPTED"
            metrics.error_message = f"Failed calculating SHA-256: {e}"
            return metrics

        # 2. Deep Media Probe using FFprobe
        cmd = [
            self.ffprobe_path,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path.resolve())
        ]

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            if result.returncode != 0:
                metrics.validation_status = "CORRUPTED"
                metrics.error_message = f"FFprobe failed to parse file: {result.stderr.strip()}"
                return metrics

            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            format_info = data.get("format", {})

            # Locate video stream
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            if not video_stream:
                metrics.validation_status = "CORRUPTED"
                metrics.error_message = "No valid video stream found in container"
                return metrics

            metrics.codec = video_stream.get("codec_name", "unknown")
            metrics.width = int(video_stream.get("width", 0))
            metrics.height = int(video_stream.get("height", 0))

            # Parse framerate (e.g. "20/1" or "25/1")
            fps_str = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = map(float, fps_str.split("/"))
                metrics.fps = round(num / den, 2) if den > 0 else 0.0
            except Exception:
                metrics.fps = 0.0

            # Parse duration from container format or video stream
            duration_val = format_info.get("duration") or video_stream.get("duration")
            if duration_val:
                try:
                    metrics.duration_seconds = round(float(duration_val), 3)
                except ValueError:
                    metrics.duration_seconds = 0.0

            # Parse total frames if reported
            nb_frames = video_stream.get("nb_frames")
            if nb_frames and nb_frames.isdigit():
                metrics.frame_count = int(nb_frames)
            elif metrics.duration_seconds > 0 and metrics.fps > 0:
                metrics.frame_count = int(round(metrics.duration_seconds * metrics.fps))

            # Quality Completeness and Integrity Evaluation
            if metrics.width > 0 and metrics.height > 0 and metrics.duration_seconds > 0:
                # Check duration completeness if expected duration was passed
                if expected_duration and metrics.duration_seconds < (expected_duration * 0.80):
                    metrics.validation_status = "DEGRADED"
                    metrics.error_message = (
                        f"Short segment: expected ~{expected_duration}s, got {metrics.duration_seconds}s "
                        "(likely interrupted by network disconnect)"
                    )
                else:
                    metrics.validation_status = "PASSED"
                    metrics.is_valid = True
            else:
                metrics.validation_status = "CORRUPTED"
                metrics.error_message = "Incomplete video headers or zero duration"

        except subprocess.TimeoutExpired:
            metrics.validation_status = "CORRUPTED"
            metrics.error_message = "FFprobe verification timed out (>30s)"
        except Exception as e:
            metrics.validation_status = "CORRUPTED"
            metrics.error_message = f"Validation exception: {e}"

        return metrics
