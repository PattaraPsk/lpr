"""
RTSP Stream Ingestion Engine (ISO/IEC 20547-3 & ISO/IEC 14496 Compliant)
Manages robust network stream capture, keyframe-aligned MP4 storage, staging isolation,
watchdog supervision, and automated recovery with exponential backoff.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .manifest import ManifestManager
from .models import SegmentManifest, StreamProfile, format_iso8601_duration
from .security import mask_rtsp_url
from .validator import MediaValidator

logger = logging.getLogger("RTSPCollector.Engine")


class StreamIngestionEngine:
    """
    Production-grade RTSP Ingestion Engine conforming to ISO standards:
    - ISO 8601: Timestamping and partitioned storage hierarchy
    - ISO/IEC 14496-14: MP4 container with FastStart moov placement
    - ISO/IEC 25012: Data quality, frame count validation, and SHA-256 integrity
    - ISO/IEC 27001: Credential isolation and log masking
    - ISO/IEC 20547-3: Big data staging-to-landing pipeline architecture
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Stream configuration
        stream_cfg = config.get("stream", {})
        self.raw_stream_url = os.path.expandvars(stream_cfg.get("stream_url", ""))
        self.masked_stream_url = mask_rtsp_url(self.raw_stream_url)
        self.camera_id = stream_cfg.get("camera_id", "CAM01")
        self.channel_id = stream_cfg.get("channel_id", "101")
        self.location_tag = stream_cfg.get("location_tag", "Default_Location")
        self.transport = stream_cfg.get("transport", "tcp")
        self.socket_timeout = stream_cfg.get("socket_timeout_seconds", 10)

        # Ingestion configuration
        ingest_cfg = config.get("ingestion", {})
        self.segment_duration = int(ingest_cfg.get("segment_duration_seconds", 900))
        self.ffmpeg_path = ingest_cfg.get("ffmpeg_path", "ffmpeg")
        self.ffprobe_path = ingest_cfg.get("ffprobe_path", "ffprobe")
        self.copy_codecs = ingest_cfg.get("copy_codecs", True)
        self.enable_faststart = ingest_cfg.get("enable_faststart", True)

        # Storage configuration
        storage_cfg = config.get("storage", {})
        self.base_storage_dir = Path(storage_cfg.get("base_dir", "dataset/Raw Video Record")).resolve()
        self.staging_dir = Path(storage_cfg.get("staging_dir", "dataset/Raw Video Record/.staging")).resolve()
        self.min_disk_free_gb = float(storage_cfg.get("min_disk_free_gb", 5.0))
        self.partition_pattern = storage_cfg.get("partition_pattern", "{camera_id}/%Y/%m/%d")

        # Quality Assurance
        qa_cfg = config.get("quality_assurance", {})
        self.compute_sha256 = qa_cfg.get("compute_sha256", True)
        self.generate_sidecar = qa_cfg.get("generate_sidecar_json", True)
        manifest_filename = qa_cfg.get("manifest_filename", "manifest.jsonl")

        # Resilience configuration
        res_cfg = config.get("resilience", {})
        self.max_retries = int(res_cfg.get("max_reconnect_attempts", -1))
        self.initial_retry_delay = float(res_cfg.get("initial_retry_delay_sec", 2.0))
        self.max_retry_delay = float(res_cfg.get("max_retry_delay_sec", 60.0))
        self.backoff_multiplier = float(res_cfg.get("backoff_multiplier", 1.5))

        # Operational components
        self.validator = MediaValidator(ffprobe_path=self.ffprobe_path)
        self.manifest_mgr = ManifestManager(
            base_storage_dir=self.base_storage_dir,
            manifest_filename=manifest_filename
        )

        # Runtime state
        self._is_running = False
        self._current_process: Optional[subprocess.Popen] = None
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Initializes storage and isolated staging directories."""
        self.base_storage_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def check_disk_space(self) -> float:
        """Returns free disk space in Gigabytes for the storage partition."""
        usage = shutil.disk_usage(self.base_storage_dir)
        free_gb = usage.free / (1024 ** 3)
        return free_gb

    def probe_stream(self) -> StreamProfile:
        """
        Actively inspects the RTSP video stream to retrieve codec profile,
        resolution, and transport characteristics.
        """
        logger.info(f"Probing stream: {self.masked_stream_url}")
        cmd = [
            self.ffprobe_path,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-rtsp_transport", self.transport,
            "-timeout", str(self.socket_timeout * 1000000),
            self.raw_stream_url
        ]

        profile = StreamProfile()
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            if res.returncode != 0:
                logger.error(f"Failed to probe RTSP stream: returncode {res.returncode}")
                return profile

            data = json.loads(res.stdout)
            streams = data.get("streams", [])
            format_info = data.get("format", {})
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

            if video_stream:
                profile.codec_name = video_stream.get("codec_name", "unknown")
                profile.profile = video_stream.get("profile", "unknown")
                profile.width = int(video_stream.get("width", 0))
                profile.height = int(video_stream.get("height", 0))
                profile.pix_fmt = video_stream.get("pix_fmt", "unknown")
                profile.color_space = video_stream.get("color_space", "unknown")
                
                fps_str = video_stream.get("r_frame_rate", "0/1")
                try:
                    num, den = map(float, fps_str.split("/"))
                    profile.fps = round(num / den, 2) if den > 0 else 0.0
                except Exception:
                    profile.fps = 0.0

            tags = format_info.get("tags", {})
            profile.server_description = tags.get("title", tags.get("comment", "Generic RTSP Server"))
            profile.raw_details = data

        except Exception as e:
            logger.error(f"Exception during stream probe: {e}")

        return profile

    def _record_segment_to_staging(
        self,
        staging_file: Path,
        duration_sec: float
    ) -> bool:
        """
        Executes FFmpeg with TCP transport and FastStart atom placement.
        Captures stream directly into staging area without re-encoding.
        """
        cmd = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", self.transport,
            "-timeout", str(self.socket_timeout * 1000000),
            "-i", self.raw_stream_url,
            "-t", str(duration_sec),
        ]

        if self.copy_codecs:
            cmd.extend(["-c:v", "copy"])
        if self.enable_faststart:
            cmd.extend(["-movflags", "+faststart"])

        cmd.extend(["-y", str(staging_file.resolve())])

        # Mask command for logging (ISO 27001)
        masked_cmd = [arg if arg != self.raw_stream_url else self.masked_stream_url for arg in cmd]
        logger.debug(f"FFmpeg command: {' '.join(masked_cmd)}")

        try:
            self._current_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = self._current_process.communicate()
            return_code = self._current_process.returncode
            self._current_process = None

            if return_code != 0 and stderr:
                logger.warning(f"FFmpeg returned non-zero code {return_code}: {stderr.strip()}")
                # If file exists and has size > 0, we can still attempt partial salvage
                if staging_file.is_file() and staging_file.stat().st_size > 1024:
                    return True
                return False
            return True

        except Exception as e:
            logger.error(f"Error executing FFmpeg capture: {e}")
            self._current_process = None
            return False

    def capture_single_segment(
        self,
        duration_sec: Optional[float] = None
    ) -> Optional[SegmentManifest]:
        """
        Captures a single video segment, validates it, and promotes it to ISO partitioned storage.
        """
        seg_dur = duration_sec or self.segment_duration

        # 1. Disk space safety check
        free_gb = self.check_disk_space()
        if free_gb < self.min_disk_free_gb:
            logger.critical(
                f"Disk space low: {free_gb:.2f} GB free (Safety threshold: {self.min_disk_free_gb} GB). Ingestion aborted."
            )
            return None

        # 2. Timing timestamps (ISO 8601 UTC and Local Offset)
        start_dt_utc = datetime.datetime.now(datetime.timezone.utc)
        start_dt_local = start_dt_utc.astimezone()

        # Temporary staging path
        temp_name = f"staging_{self.camera_id}_CH{self.channel_id}_{start_dt_utc.strftime('%Y%m%dT%H%M%SZ')}.mp4"
        staging_file = self.staging_dir / temp_name

        logger.info(
            f"Starting capture: {self.camera_id} (Channel {self.channel_id}) for {seg_dur}s -> {temp_name}"
        )

        success = self._record_segment_to_staging(staging_file, seg_dur)

        end_dt_utc = datetime.datetime.now(datetime.timezone.utc)
        end_dt_local = end_dt_utc.astimezone()

        if not staging_file.is_file() or staging_file.stat().st_size == 0:
            logger.error("Capture failed: No file generated in staging directory.")
            if staging_file.exists():
                staging_file.unlink(missing_ok=True)
            return None

        # 3. Quality Assurance & Validation (ISO 25012)
        metrics = self.validator.validate_file(staging_file, expected_duration=seg_dur)
        logger.info(
            f"Segment validation: {metrics.validation_status} | "
            f"Duration: {metrics.duration_seconds}s | Frames: {metrics.frame_count} | "
            f"Size: {metrics.file_size_bytes / (1024*1024):.2f} MB"
        )

        # 4. Resolve Target ISO Partition Path
        # Format: {camera_id}/YYYY/MM/DD/
        partition_subfolder = start_dt_local.strftime(
            self.partition_pattern.format(camera_id=self.camera_id)
        )
        target_dir = self.base_storage_dir / partition_subfolder
        target_dir.mkdir(parents=True, exist_ok=True)

        # Filename adheres to ISO 8601 formatting: CAM01_CH101_2026-10-19T13-45-00+07-00.mp4
        iso_time_str = start_dt_local.strftime("%Y-%m-%dT%H-%M-%S%z")
        final_filename = f"{self.camera_id}_CH{self.channel_id}_{iso_time_str}.mp4"
        final_destination = target_dir / final_filename

        # 5. Atomic Promotion from Staging to Partitioned Landing Zone
        shutil.move(str(staging_file), str(final_destination))
        rel_path = final_destination.relative_to(self.base_storage_dir).as_posix()

        # 6. Build Manifest Record
        manifest_record = SegmentManifest(
            file_name=final_filename,
            relative_path=rel_path,
            camera_id=self.camera_id,
            channel_id=str(self.channel_id),
            location_tag=self.location_tag,
            start_time_iso_utc=start_dt_utc.isoformat(),
            end_time_iso_utc=end_dt_utc.isoformat(),
            start_time_iso_local=start_dt_local.isoformat(),
            end_time_iso_local=end_dt_local.isoformat(),
            duration_seconds=metrics.duration_seconds,
            duration_iso8601=format_iso8601_duration(metrics.duration_seconds),
            file_size_bytes=metrics.file_size_bytes,
            sha256_checksum=metrics.sha256,
            video_codec=metrics.codec,
            resolution=f"{metrics.width}x{metrics.height}",
            fps=metrics.fps,
            total_frames=metrics.frame_count,
            source_url_masked=self.masked_stream_url,
            quality_status=metrics.validation_status,
            validation_notes=metrics.error_message
        )

        # 7. Append to Central Manifest and Sidecar
        self.manifest_mgr.append_segment(manifest_record, write_sidecar=self.generate_sidecar)

        return manifest_record

    def run_continuous_ingestion(self, total_duration_sec: Optional[float] = None) -> None:
        """
        Runs continuous 24/7 stream ingestion with automated fault tolerance,
        exponential backoff on network drop, and graceful SIGINT/SIGTERM handling.
        """
        self._is_running = True
        logger.info(
            f"Initiating continuous ISO stream collection for {self.camera_id} "
            f"[Segment Duration: {self.segment_duration}s, Storage: {self.base_storage_dir}]"
        )

        # Register graceful termination handlers
        def _signal_handler(sig, frame):
            logger.info("Termination signal received. Gracefully concluding stream capture...")
            self._is_running = False
            if self._current_process:
                try:
                    self._current_process.terminate()
                except Exception:
                    pass

        try:
            signal.signal(signal.SIGINT, _signal_handler)
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            pass  # On non-main thread or Windows limitations

        retry_count = 0
        current_delay = self.initial_retry_delay
        total_collected_sec = 0.0

        while self._is_running:
            # Check duration ceiling if specified
            if total_duration_sec and total_collected_sec >= total_duration_sec:
                logger.info(f"Target duration of {total_duration_sec}s reached. Concluding capture.")
                break

            remaining_time = None
            if total_duration_sec:
                remaining_time = min(self.segment_duration, total_duration_sec - total_collected_sec)

            record = self.capture_single_segment(duration_sec=remaining_time)

            if record and record.quality_status in ("PASSED", "DEGRADED"):
                # Successful chunk captured
                retry_count = 0
                current_delay = self.initial_retry_delay
                total_collected_sec += record.duration_seconds
                logger.info(
                    f"Chunk complete: {record.file_name} ({record.duration_seconds}s). "
                    f"Cumulative: {total_collected_sec:.1f}s"
                )
            else:
                # Connection or capture failure -> apply exponential backoff
                retry_count += 1
                logger.warning(
                    f"Stream dropped or segment capture failed (Attempt {retry_count}). "
                    f"Retrying in {current_delay:.1f}s..."
                )
                
                if 0 <= self.max_retries < retry_count:
                    logger.error(f"Camera offline: Maximum retry limit ({self.max_retries}) reached without stream recovery. Concluding ingestion.")
                    break

                time.sleep(current_delay)
                current_delay = min(current_delay * self.backoff_multiplier, self.max_retry_delay)

        self._is_running = False
        logger.info(f"Stream ingestion concluded. Total captured duration: {total_collected_sec:.1f}s.")

    def stop(self) -> None:
        """Halts the ingestion process safely."""
        self._is_running = False
        if self._current_process:
            try:
                self._current_process.terminate()
            except Exception:
                pass
