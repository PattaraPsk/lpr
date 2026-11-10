#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial Multi-Object Plate Tracker and Consensus Resolution.
Compliant with ISO 3166-2:TH and ISO 8601.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tools.DataManipulator.app.provinces import get_iso_code


def compute_sharpness(crop_bgr: np.ndarray) -> float:
    """Computes Laplacian variance sharpness score on an image crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def box_iou(b1: List[float], b2: List[float]) -> float:
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def box_center_dist(b1: List[float], b2: List[float]) -> float:
    """Euclidean distance between bounding box center points."""
    cx1 = (b1[0] + b1[2]) * 0.5
    cy1 = (b1[1] + b1[3]) * 0.5
    cx2 = (b2[0] + b2[2]) * 0.5
    cy2 = (b2[1] + b2[3]) * 0.5
    return float(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)


class TrackedPlate:
    """
    Tracks a license plate across consecutive video frames, accumulating OCR hypotheses
    and computing consensus readings.
    """

    def __init__(
        self,
        track_id: int,
        box: List[float],
        conf: float,
        frame_idx: int,
        crop_bgr: np.ndarray,
        ocr_result: Dict[str, Any],
        vehicle_type: str = "Unknown",
        orientation: str = "Unknown"
    ):
        self.track_id = track_id
        self.current_box = box
        self.last_frame = frame_idx
        self.vehicle_type = vehicle_type
        self.orientation = orientation
        self.hits = 1
        self.misses = 0

        # OCR reading history
        self.ocr_history: List[Dict[str, Any]] = [ocr_result]
        self.best_crop = crop_bgr.copy()
        self.best_conf = ocr_result.get("confidence", 0.0)
        self.best_ocr = ocr_result

        # Optical sharpness history
        self.best_sharpness = compute_sharpness(crop_bgr)
        self.best_sharp_frame = frame_idx
        self.best_sharp_crop = crop_bgr.copy()
        self.best_sharp_car: Optional[np.ndarray] = None

        # Consensus reading
        self.consensus_plate = ocr_result.get("full_plate", "")
        self.consensus_d3 = ocr_result.get("d3", "")
        self.consensus_d4 = ocr_result.get("d4", "")
        self.consensus_province = ocr_result.get("province", "")
        self.iso_province = get_iso_code(self.consensus_province) or ""

        self.first_seen_iso = datetime.now(timezone.utc).isoformat()
        self.last_seen_iso = self.first_seen_iso

    def update(
        self,
        box: List[float],
        conf: float,
        frame_idx: int,
        crop_bgr: np.ndarray,
        ocr_result: Dict[str, Any],
        vehicle_type: str = "Unknown",
        orientation: str = "Unknown",
        car_crop: Optional[np.ndarray] = None
    ):
        self.current_box = box
        self.last_frame = frame_idx
        self.hits += 1
        self.misses = 0
        self.last_seen_iso = datetime.now(timezone.utc).isoformat()

        if vehicle_type != "Unknown":
            self.vehicle_type = vehicle_type
        if orientation != "Unknown":
            self.orientation = orientation

        cur_sharpness = compute_sharpness(crop_bgr)
        if cur_sharpness > self.best_sharpness or self.best_sharp_crop is None:
            self.best_sharpness = cur_sharpness
            self.best_sharp_frame = frame_idx
            self.best_sharp_crop = crop_bgr.copy()
            if car_crop is not None:
                self.best_sharp_car = car_crop.copy()

        ocr_conf = ocr_result.get("confidence", 0.0)
        if ocr_result.get("full_plate"):
            self.ocr_history.append(ocr_result)
            if ocr_conf > self.best_conf:
                self.best_conf = ocr_conf
                self.best_ocr = ocr_result
                self.best_crop = crop_bgr.copy()

        # Update consensus from history
        self._recompute_consensus()

    def _recompute_consensus(self):
        d3_candidates = [o["d3"] for o in self.ocr_history if o.get("d3")]
        d4_candidates = [o["d4"] for o in self.ocr_history if o.get("d4")]
        prov_candidates = [o["province"] for o in self.ocr_history if o.get("province")]

        if d3_candidates:
            self.consensus_d3 = collections.Counter(d3_candidates).most_common(1)[0][0]
        if d4_candidates:
            self.consensus_d4 = collections.Counter(d4_candidates).most_common(1)[0][0]
        if prov_candidates:
            self.consensus_province = collections.Counter(prov_candidates).most_common(1)[0][0]

        self.iso_province = get_iso_code(self.consensus_province) or ""
        self.consensus_plate = f"{self.consensus_d3} {self.consensus_d4} {self.consensus_province}".strip()
