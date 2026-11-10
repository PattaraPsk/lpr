#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SelenCORE LPR Engine — Isolated Integration Adapter.

Wraps the full Thai LPR detection + OCR reader pipeline into a single,
integration-ready class with no dependency on the interactive demo UI.

Designed for embedding into SelenCORE or any external system.
Compliant with ISO/IEC 42001:2023 (Annex A.9.3), ISO 8601, ISO 3166-2:TH.

Usage Example (see Developer Manual: docs/SELENCORE_LPR_MANUAL.md)
-------------------------------------------------------------------
    from selencore import LPREngine

    engine = LPREngine.from_default_models()
    results = engine.process_frame(bgr_frame)
    for det in results.detections:
        print(det.plate_text, det.province, det.confidence)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Optional torch import guard
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


# ---------------------------------------------------------------------------
# Project root resolution (works from any working directory)
# ---------------------------------------------------------------------------
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent


def _add_project_root_to_path() -> None:
    """Ensures the LPR project root is on sys.path for internal imports."""
    root_str = str(_PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_add_project_root_to_path()


# ---------------------------------------------------------------------------
# Data-class definitions — public contract with SelenCORE
# ---------------------------------------------------------------------------

@dataclass
class PlateReading:
    """
    Structured OCR result for a single detected license plate.

    Fields
    ------
    d3 : Thai consonant prefix / leading digit group  (e.g. ``"1กข"``)
    d4 : Registration number digits                   (e.g. ``"1234"``)
    province : Thai province name (NFC-normalised)    (e.g. ``"ชลบุรี"``)
    province_iso : ISO 3166-2:TH code                 (e.g. ``"TH-20"``)
    full_plate : Human-readable concatenation         (e.g. ``"1กข 1234 ชลบุรี"``)
    confidence : Mean token-level softmax probability (0.0 – 1.0)
    tokens : Raw argmax token indices from the model  (may be ``[]`` for non-attention arches)
    error : Non-None if OCR inference failed
    """
    d3: str = ""
    d4: str = ""
    province: str = ""
    province_iso: str = ""
    full_plate: str = ""
    confidence: float = 0.0
    tokens: List[int] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class Detection:
    """
    Single license plate detection result for one frame.

    Fields
    ------
    box : Bounding box in full-frame pixel coords ``[x1, y1, x2, y2]``
    det_confidence : YOLO plate detection confidence (0.0 – 1.0)
    vehicle_type : Matched vehicle class — ``"Car"``, ``"Motorcycle"``, ``"Truck"``, ``"Bus"``, ``"Unknown"``
    orientation : Plate orientation — ``"Front"``, ``"Rear"``, ``"Unknown"``
    reading : Structured OCR result (see :class:`PlateReading`)
    crop_bgr : NumPy BGR uint8 array of the plate crop (may be ``None``)
    track_id : Assigned tracker ID across frames (``-1`` if tracking disabled)
    """
    box: List[float]
    det_confidence: float
    vehicle_type: str
    orientation: str
    reading: PlateReading
    crop_bgr: Optional[np.ndarray] = None
    track_id: int = -1

    @property
    def plate_text(self) -> str:
        """Convenience alias for ``reading.full_plate``."""
        return self.reading.full_plate

    @property
    def confidence(self) -> float:
        """Convenience alias for ``reading.confidence``."""
        return self.reading.confidence


@dataclass
class LPRResult:
    """
    Complete per-frame output of :class:`LPREngine.process_frame`.

    Fields
    ------
    detections : All plate detections found in the frame
    frame_index : Monotonic integer frame counter
    timestamp_iso : ISO 8601 UTC timestamp of inference (Annex A.9.3)
    det_latency_ms : Detector inference wall-clock time (ms)
    ocr_latency_ms : OCR inference wall-clock time (ms)
    total_latency_ms : Full pipeline latency (ms)
    source_shape : ``(height, width)`` of the input frame
    """
    detections: List[Detection]
    frame_index: int
    timestamp_iso: str
    det_latency_ms: float
    ocr_latency_ms: float
    total_latency_ms: float
    source_shape: Tuple[int, int]

    @property
    def has_plates(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable dict suitable for JSON encoding or SelenCORE event bus."""
        return {
            "frame_index": self.frame_index,
            "timestamp_iso": self.timestamp_iso,
            "det_latency_ms": round(self.det_latency_ms, 2),
            "ocr_latency_ms": round(self.ocr_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "source_shape": list(self.source_shape),
            "detections": [
                {
                    "track_id": d.track_id,
                    "box": [round(v, 1) for v in d.box],
                    "det_confidence": round(d.det_confidence, 4),
                    "vehicle_type": d.vehicle_type,
                    "orientation": d.orientation,
                    "plate_text": d.plate_text,
                    "d3": d.reading.d3,
                    "d4": d.reading.d4,
                    "province": d.reading.province,
                    "province_iso": d.reading.province_iso,
                    "ocr_confidence": round(d.confidence, 4),
                    "ocr_error": d.reading.error,
                }
                for d in self.detections
            ],
        }


# ---------------------------------------------------------------------------
# Default model candidate paths
# ---------------------------------------------------------------------------

_DEFAULT_DETECTOR_CANDIDATES: List[Path] = [
    _PROJECT_ROOT / "models" / "detector" / "vehicle_plate_yolov11n.pt",
    _PROJECT_ROOT / "models" / "detector" / "vehicle_plate_detector.pt",
    _PROJECT_ROOT / "vehicle_plate_detector.pt",
]

_DEFAULT_VEHICLE_CANDIDATES: List[Path] = [
    _PROJECT_ROOT / "models" / "detector" / "yolov8s.pt",
    _PROJECT_ROOT / "yolov8s.pt",
]

_DEFAULT_READER_CANDIDATES: List[Path] = [
    _PROJECT_ROOT / "models" / "reader" / "thai_plate_reader.pth",
    _PROJECT_ROOT / "models" / "reader" / "selen-lpr-0.0.1" / "best_reader.pth",
    _PROJECT_ROOT / "thai_plate_reader.pth",
]


def _resolve_first_existing(candidates: List[Path], label: str) -> Path:
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"[SelenCORE LPR] No {label} checkpoint found. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


# ---------------------------------------------------------------------------
# LPREngine — main integration class
# ---------------------------------------------------------------------------

class LPREngine:
    """
    Single-call Thai License Plate Recognition engine.

    Wraps:
    - YOLOv11n plate + orientation detector
    - YOLOv8s vehicle context detector (optional)
    - VisionAttentionOCR / SelenLPR reader

    Parameters
    ----------
    detector_path : Path to YOLOv11n ``.pt`` plate detector weights.
    reader_path : Path to ``.pth`` OCR reader weights.
    vehicle_detector_path : Path to YOLOv8s vehicle weights. ``None`` disables.
    conf_thresh : YOLO plate detection confidence threshold (default 0.25).
    veh_conf_thresh : YOLO vehicle detection confidence threshold (default 0.35).
    device : PyTorch device string — ``"cpu"``, ``"cuda:0"``, etc.
    min_plate_width_px : Minimum plate pixel width to run OCR on (default 14).
    min_plate_height_px : Minimum plate pixel height to run OCR on (default 8).
    enable_motion_filter : Use MOG2 motion pre-filter to skip stationary scenes.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        detector_path: Path,
        reader_path: Path,
        vehicle_detector_path: Optional[Path] = None,
        conf_thresh: float = 0.25,
        veh_conf_thresh: float = 0.35,
        device: Optional[str] = None,
        min_plate_width_px: int = 14,
        min_plate_height_px: int = 8,
        enable_motion_filter: bool = True,
    ):
        if not _HAS_YOLO:
            raise ImportError("ultralytics is required: pip install ultralytics")
        if not _HAS_PIL:
            raise ImportError("Pillow is required: pip install Pillow")
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required: pip install torch")

        self.conf_thresh = conf_thresh
        self.veh_conf_thresh = veh_conf_thresh
        self.min_plate_width_px = min_plate_width_px
        self.min_plate_height_px = min_plate_height_px
        self.enable_motion_filter = enable_motion_filter
        self._frame_index: int = 0

        # Resolve compute device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load detector
        self._detector = YOLO(str(detector_path)).to(self.device)

        # Load vehicle detector (optional)
        self._veh_detector = None
        if vehicle_detector_path and vehicle_detector_path.is_file():
            self._veh_detector = YOLO(str(vehicle_detector_path)).to(self.device)

        # Load OCR reader via project's PlateReader singleton
        from tools.DataManipulator.app.plate_reader import PlateReader
        self._reader = PlateReader(weights_path=str(reader_path), device=self.device)
        if self._reader.error_msg:
            raise RuntimeError(f"[SelenCORE LPR] Reader load error: {self._reader.error_msg}")

        # Province ISO lookup
        from tools.DataManipulator.app.provinces import get_iso_code
        self._get_iso_code = get_iso_code

        # Import detector helpers (lazy — avoids circular imports)
        from tools.alpr.detector import (
            MotionDetector,
            detect_moving_vehicles,
            detect_vehicles,
            detect_plates_coarse_to_fine,
            detect_plates_direct,
            match_vehicle_and_orientation,
        )
        self._MotionDetector = MotionDetector
        self._detect_moving_vehicles = detect_moving_vehicles
        self._detect_vehicles = detect_vehicles
        self._detect_plates_coarse_to_fine = detect_plates_coarse_to_fine
        self._detect_plates_direct = detect_plates_direct
        self._match_vehicle_and_orientation = match_vehicle_and_orientation

        # Motion detector instance
        self._motion_detector = MotionDetector() if enable_motion_filter else None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_default_models(
        cls,
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        veh_conf_thresh: float = 0.35,
        enable_motion_filter: bool = True,
    ) -> "LPREngine":
        """
        Constructs ``LPREngine`` by auto-discovering model weights from the
        standard ``models/`` directory. Raises ``FileNotFoundError`` if any
        required model is missing.
        """
        det = _resolve_first_existing(_DEFAULT_DETECTOR_CANDIDATES, "plate detector")
        rdr = _resolve_first_existing(_DEFAULT_READER_CANDIDATES, "OCR reader")
        veh: Optional[Path] = None
        for c in _DEFAULT_VEHICLE_CANDIDATES:
            if c.is_file():
                veh = c.resolve()
                break
        return cls(
            detector_path=det,
            reader_path=rdr,
            vehicle_detector_path=veh,
            device=device,
            conf_thresh=conf_thresh,
            veh_conf_thresh=veh_conf_thresh,
            enable_motion_filter=enable_motion_filter,
        )

    @classmethod
    def from_paths(
        cls,
        detector_path: str | Path,
        reader_path: str | Path,
        vehicle_detector_path: Optional[str | Path] = None,
        **kwargs,
    ) -> "LPREngine":
        """
        Constructs ``LPREngine`` from explicit model paths.

        Parameters
        ----------
        detector_path : str | Path
            Path to YOLOv11n plate/orientation detector (``.pt``).
        reader_path : str | Path
            Path to OCR reader checkpoint (``.pth``).
        vehicle_detector_path : str | Path, optional
            Path to YOLOv8s vehicle detector. Pass ``None`` to disable.
        **kwargs :
            Forwarded to :class:`LPREngine.__init__`.
        """
        veh = Path(vehicle_detector_path).resolve() if vehicle_detector_path else None
        return cls(
            detector_path=Path(detector_path).resolve(),
            reader_path=Path(reader_path).resolve(),
            vehicle_detector_path=veh,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[List[int]] = None,
        run_ocr: bool = True,
        coarse_to_fine: bool = True,
    ) -> LPRResult:
        """
        Run the full LPR pipeline on one BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input frame in OpenCV BGR format, shape ``(H, W, 3)`` dtype ``uint8``.
        roi : list[int] | None
            Optional Region of Interest ``[x1, y1, x2, y2]`` in pixel coords.
            When given, detection runs only inside this region.
        run_ocr : bool
            If ``False``, skips OCR — returns bounding boxes and vehicle context only.
        coarse_to_fine : bool
            If ``True`` and vehicle detector is loaded, uses vehicle-crop cascade.
            If ``False`` or no vehicle detector, runs plate detection on full frame.

        Returns
        -------
        LPRResult
            Fully populated result for the frame. Always returns a valid object —
            check ``result.has_plates`` and individual ``det.reading.error``.
        """
        t0 = time.perf_counter()
        ts_iso = datetime.now(timezone.utc).isoformat()
        self._frame_index += 1
        frame_idx = self._frame_index

        orig_h, orig_w = frame_bgr.shape[:2]

        # --- ROI clipping ---
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            rx1 = max(0, min(rx1, orig_w - 20))
            ry1 = max(0, min(ry1, orig_h - 20))
            rx2 = max(rx1 + 20, min(rx2, orig_w))
            ry2 = max(ry1 + 20, min(ry2, orig_h))
            det_frame = frame_bgr[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            det_frame = frame_bgr
            offset_x, offset_y = 0, 0

        # --- Stage 1: Vehicle detection (coarse) ---
        vehicles: List[Tuple[List[float], str, float]] = []
        if self._veh_detector is not None and coarse_to_fine:
            if self.enable_motion_filter and self._motion_detector is not None:
                _, motion_boxes, _ = self._motion_detector.update(frame_bgr)
                vehicles = self._detect_moving_vehicles(
                    veh_detector=self._veh_detector,
                    det_frame=det_frame,
                    motion_boxes=motion_boxes,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    conf_thresh=self.veh_conf_thresh,
                    device=self.device,
                    motion_detector=self._motion_detector,
                )
            else:
                vehicles = self._detect_vehicles(
                    veh_detector=self._veh_detector,
                    det_frame=det_frame,
                    offset_x=offset_x,
                    offset_y=offset_y,
                    conf_thresh=self.veh_conf_thresh,
                    device=self.device,
                )

        # --- Stage 2: Plate detection (fine) ---
        t_det0 = time.perf_counter()
        if coarse_to_fine and self._veh_detector is not None and vehicles:
            plates, orientations = self._detect_plates_coarse_to_fine(
                detector=self._detector,
                frame=frame_bgr,
                vehicles=vehicles,
                orig_w=orig_w,
                orig_h=orig_h,
                conf_thresh=self.conf_thresh,
                device=self.device,
            )
        else:
            plates, orientations = self._detect_plates_direct(
                detector=self._detector,
                det_frame=det_frame,
                offset_x=offset_x,
                offset_y=offset_y,
                conf_thresh=self.conf_thresh,
                device=self.device,
            )
        det_latency_ms = (time.perf_counter() - t_det0) * 1000.0

        # --- Stage 3: OCR per plate ---
        t_ocr0 = time.perf_counter()
        detections: List[Detection] = []

        for p_box, p_conf in plates:
            px1, py1, px2, py2 = p_box
            pw = px2 - px1
            ph = py2 - py1

            # Size guard
            if pw < self.min_plate_width_px or ph < self.min_plate_height_px:
                continue
            # Reject implausibly large (>45% of frame width — probably non-plate)
            if (pw / orig_w) > 0.45:
                continue

            cx1 = max(0, int(px1))
            cy1 = max(0, int(py1))
            cx2 = min(orig_w, int(px2))
            cy2 = min(orig_h, int(py2))
            crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
            if crop_bgr.size == 0:
                continue

            # Vehicle + orientation spatial matching
            veh_type, orientation = self._match_vehicle_and_orientation(
                p_box, orientations, vehicles
            )

            # OCR
            reading = PlateReading()
            if run_ocr:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                raw = self._reader.predict(pil_crop, use_cache=False)
                iso = self._get_iso_code(raw.get("province", "")) or ""
                reading = PlateReading(
                    d3=raw.get("d3", ""),
                    d4=raw.get("d4", ""),
                    province=raw.get("province", ""),
                    province_iso=iso,
                    full_plate=raw.get("full_plate", ""),
                    confidence=float(raw.get("confidence", 0.0)),
                    tokens=raw.get("tokens", []),
                    error=raw.get("error"),
                )

            detections.append(Detection(
                box=p_box,
                det_confidence=p_conf,
                vehicle_type=veh_type,
                orientation=orientation,
                reading=reading,
                crop_bgr=crop_bgr,
                track_id=-1,
            ))

        ocr_latency_ms = (time.perf_counter() - t_ocr0) * 1000.0
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return LPRResult(
            detections=detections,
            frame_index=frame_idx,
            timestamp_iso=ts_iso,
            det_latency_ms=det_latency_ms,
            ocr_latency_ms=ocr_latency_ms,
            total_latency_ms=total_latency_ms,
            source_shape=(orig_h, orig_w),
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def process_image_file(self, image_path: str | Path) -> LPRResult:
        """
        Process a single image file (JPEG/PNG/BMP).

        Parameters
        ----------
        image_path : str | Path
            Absolute or relative path to the image file.

        Returns
        -------
        LPRResult
        """
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        return self.process_frame(frame)

    def process_pil_image(self, pil_image: "Image.Image") -> LPRResult:
        """
        Process a PIL Image object directly.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            Image in any PIL-supported mode. Internally converted to BGR NumPy.

        Returns
        -------
        LPRResult
        """
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self.process_frame(bgr)

    def reset_motion_model(self) -> None:
        """
        Resets the MOG2 background model.
        Call this when switching camera sources or scenes.
        """
        if self._motion_detector is not None:
            from tools.alpr.detector import MotionDetector
            self._motion_detector = MotionDetector()

    @property
    def frame_count(self) -> int:
        """Total number of frames processed since instantiation."""
        return self._frame_index

    def __repr__(self) -> str:
        return (
            f"LPREngine(v{self.VERSION}, device={self.device!r}, "
            f"conf={self.conf_thresh}, frames_processed={self._frame_index})"
        )
