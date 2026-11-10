#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Detection & Vehicle Cascade Strategy.
Compliant with ISO/IEC 42001 and ISO/IEC 25012.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from tools.alpr.tracker import box_iou


def match_vehicle_and_orientation(
    plate_box: List[float],
    orientations: List[Tuple[List[float], str, float]],
    vehicles: List[Tuple[List[float], str, float]]
) -> Tuple[str, str]:
    """
    Spatial containment matching for vehicle class (including Motorcycle) and orientation.
    Enforces Thailand DLT rule: Motorcycles strictly only have rear plates.
    """
    px1, py1, px2, py2 = plate_box
    pcx = (px1 + px2) * 0.5
    pcy = (py1 + py2) * 0.5

    matched_veh = "Unknown"
    best_veh_score = 1e9

    for (vx1, vy1, vx2, vy2), v_type, v_conf in vehicles:
        vw = vx2 - vx1
        vh = vy2 - vy1
        # Expand vehicle bounding box slightly (especially downward where motorcycle plates sit)
        exp_x = vw * 0.12
        exp_y_top = vh * 0.10
        exp_y_bot = vh * 0.35 if v_type == "Motorcycle" else vh * 0.15

        if (vx1 - exp_x <= pcx <= vx2 + exp_x) and (vy1 - exp_y_top <= pcy <= vy2 + exp_y_bot):
            v_area = vw * vh
            if v_area < best_veh_score:
                best_veh_score = v_area
                matched_veh = v_type

    # Proximity fallback for motorcycles when plate is slightly outside box
    if matched_veh == "Unknown":
        for (vx1, vy1, vx2, vy2), v_type, _ in vehicles:
            if v_type == "Motorcycle":
                dist = ((pcx - (vx1 + vx2) * 0.5) ** 2 + (pcy - (vy1 + vy2) * 0.5) ** 2) ** 0.5
                diag = ((vx2 - vx1) ** 2 + (vy2 - vy1) ** 2) ** 0.5
                if dist < diag * 0.75:
                    matched_veh = "Motorcycle"
                    break

    matched_orient = "Unknown"
    for (ox1, oy1, ox2, oy2), o_name, _ in orientations:
        if ox1 <= pcx <= ox2 and oy1 <= pcy <= oy2:
            matched_orient = o_name
            break

    # In Thailand, motorcycles strictly only have rear plates
    if matched_veh == "Motorcycle" and matched_orient == "Unknown":
        matched_orient = "Rear"

    return matched_veh, matched_orient


def compute_crop_sharpness(crop_bgr: np.ndarray) -> float:
    """
    Computes Laplacian variance sharpness score of an image crop.
    Higher values indicate sharper edges and better optical focus.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


class MotionDetector:
    """
    Edge-optimized Background/Foreground Motion Detector.
    Uses downscaled MOG2 background modeling and morphological filtering
    to extract moving foreground areas before running heavy neural vehicle detectors.
    Eliminates computational overhead from static backgrounds and parked vehicles.
    """

    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 25.0,
        detect_shadows: bool = False,
        downsample_size: Tuple[int, int] = (640, 360),
        min_motion_area: float = 400.0,
        learning_rate: float = -1.0,
    ):
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.downsample_size = downsample_size
        self.min_motion_area = min_motion_area
        self.learning_rate = learning_rate
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.frame_count = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, List[List[int]], np.ndarray]:
        """
        Processes frame, updates background model, and returns:
        (has_motion, motion_boxes, fg_mask).
        Motion boxes are scaled back to frame resolution: [x1, y1, x2, y2].
        """
        self.frame_count += 1
        orig_h, orig_w = frame.shape[:2]
        small = cv2.resize(frame, self.downsample_size, interpolation=cv2.INTER_LINEAR)
        fg_mask = self.subtractor.apply(small, learningRate=self.learning_rate)
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_DILATE, self.kernel_dilate)

        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = orig_w / float(self.downsample_size[0])
        scale_y = orig_h / float(self.downsample_size[1])

        motion_boxes: List[List[int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self.min_motion_area:
                x, y, w, h = cv2.boundingRect(c)
                gx1 = max(0, int(x * scale_x))
                gy1 = max(0, int(y * scale_y))
                gx2 = min(orig_w, int((x + w) * scale_x))
                gy2 = min(orig_h, int((y + h) * scale_y))
                motion_boxes.append([gx1, gy1, gx2, gy2])

        has_motion = len(motion_boxes) > 0
        return has_motion, motion_boxes, fg_clean

    def is_moving(self, box: List[float], motion_boxes: List[List[int]], min_overlap: float = 0.08) -> bool:
        """
        Determines whether candidate bounding box overlaps with any active moving foreground region.
        """
        if not motion_boxes:
            return False
        bx1, by1, bx2, by2 = box
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        b_area = bw * bh
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5

        for mx1, my1, mx2, my2 in motion_boxes:
            # Check center point containment
            if mx1 <= bcx <= mx2 and my1 <= bcy <= my2:
                return True
            # Check spatial intersection
            ix1 = max(bx1, float(mx1))
            iy1 = max(by1, float(my1))
            ix2 = min(bx2, float(mx2))
            iy2 = min(by2, float(my2))
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                m_area = max(1.0, float((mx2 - mx1) * (my2 - my1)))
                if (inter_area / b_area) >= min_overlap or (inter_area / m_area) >= min_overlap:
                    return True
        return False


def detect_moving_vehicles(
    veh_detector: YOLO,
    det_frame: np.ndarray,
    motion_boxes: List[List[int]],
    offset_x: int,
    offset_y: int,
    conf_thresh: float,
    device: str,
    motion_detector: Optional[MotionDetector] = None
) -> List[Tuple[List[float], str, float]]:
    """
    Detects vehicles only when motion is present and strictly filters out stationary/parked vehicles.
    """
    if not motion_boxes:
        return []

    raw_vehicles = detect_vehicles(
        veh_detector=veh_detector,
        det_frame=det_frame,
        offset_x=offset_x,
        offset_y=offset_y,
        conf_thresh=conf_thresh,
        device=device
    )

    moving_vehicles: List[Tuple[List[float], str, float]] = []
    for veh in raw_vehicles:
        box = veh[0]
        if motion_detector is not None:
            if motion_detector.is_moving(box, motion_boxes):
                moving_vehicles.append(veh)
        else:
            # Fallback inline overlap check
            bx1, by1, bx2, by2 = box
            b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            is_mv = False
            for mx1, my1, mx2, my2 in motion_boxes:
                if mx1 <= bcx <= mx2 and my1 <= bcy <= my2:
                    is_mv = True
                    break
                ix1 = max(bx1, float(mx1))
                iy1 = max(by1, float(my1))
                ix2 = min(bx2, float(mx2))
                iy2 = min(by2, float(my2))
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if (inter_area / b_area) >= 0.08:
                        is_mv = True
                        break
            if is_mv:
                moving_vehicles.append(veh)

    return moving_vehicles


def detect_vehicles(
    veh_detector: YOLO,
    det_frame: np.ndarray,
    offset_x: int,
    offset_y: int,
    conf_thresh: float,
    device: str
) -> List[Tuple[List[float], str, float]]:
    """Runs coarse vehicle detection on ROI or frame."""
    vehicles: List[Tuple[List[float], str, float]] = []
    veh_map = {2: "Car", 3: "Motorcycle", 7: "Truck", 5: "Bus"}
    veh_results = veh_detector(
        det_frame,
        classes=[2, 3, 5, 7],
        conf=conf_thresh,
        verbose=False,
        device=device
    )[0]
    for b in veh_results.boxes:
        cid = int(b.cls[0])
        if cid in veh_map:
            bx = b.xyxy[0].tolist()
            gbx = [bx[0] + offset_x, bx[1] + offset_y, bx[2] + offset_x, bx[3] + offset_y]
            vehicles.append((gbx, veh_map[cid], float(b.conf[0])))
    return vehicles


def detect_plates_coarse_to_fine(
    detector: YOLO,
    frame: np.ndarray,
    vehicles: List[Tuple[List[float], str, float]],
    orig_w: int,
    orig_h: int,
    conf_thresh: float,
    device: str
) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], str, float]]]:
    """
    Batched inference pass across candidate vehicle crops.
    Returns (plates, orientations).
    """
    plates: List[Tuple[List[float], float]] = []
    orientations: List[Tuple[List[float], str, float]] = []

    valid_crops = []
    crop_meta = []
    for v_box, v_cls, v_conf in vehicles:
        vx1, vy1, vx2, vy2 = [int(v) for v in v_box]
        vw = max(1, vx2 - vx1)
        vh = max(1, vy2 - vy1)
        if vw < 45 or vh < 45:
            continue
        pad_x = int(vw * 0.10)
        pad_y1 = int(vh * 0.10)
        pad_y2 = int(vh * 0.35) if v_cls == "Motorcycle" else int(vh * 0.15)
        cx1 = max(0, vx1 - pad_x)
        cy1 = max(0, vy1 - pad_y1)
        cx2 = min(orig_w, vx2 + pad_x)
        cy2 = min(orig_h, vy2 + pad_y2)
        v_crop = frame[cy1:cy2, cx1:cx2]
        if v_crop.size == 0 or (cx2 - cx1) < 20 or (cy2 - cy1) < 20:
            continue
        valid_crops.append(v_crop)
        crop_meta.append((cx1, cy1, v_cls))

    if valid_crops:
        batch_res = detector(
            valid_crops,
            conf=conf_thresh,
            imgsz=640,
            verbose=False,
            device=device
        )
        for idx, c_res in enumerate(batch_res):
            cx1, cy1, _ = crop_meta[idx]
            for b in c_res.boxes:
                cid = int(b.cls[0])
                c_conf = float(b.conf[0])
                bx = b.xyxy[0].tolist()
                gbx = [bx[0] + cx1, bx[1] + cy1, bx[2] + cx1, bx[3] + cy1]
                if cid == 1:
                    plates.append((gbx, c_conf))
                elif cid == 0:
                    orientations.append((gbx, "Front", c_conf))
                elif cid == 2:
                    orientations.append((gbx, "Rear", c_conf))

    # Deduplicate overlapping plate detections
    if len(plates) > 1:
        keep_plates = []
        for p_box, p_conf in sorted(plates, key=lambda x: x[1], reverse=True):
            if not any(box_iou(p_box, k_box) > 0.50 for k_box, _ in keep_plates):
                keep_plates.append((p_box, p_conf))
        plates = keep_plates

    return plates, orientations


def detect_plates_direct(
    detector: YOLO,
    det_frame: np.ndarray,
    offset_x: int,
    offset_y: int,
    conf_thresh: float,
    device: str
) -> Tuple[List[Tuple[List[float], float]], List[Tuple[List[float], str, float]]]:
    """
    Direct full-frame or user-drawn ROI plate detection.
    Returns (plates, orientations).
    """
    plates: List[Tuple[List[float], float]] = []
    orientations: List[Tuple[List[float], str, float]] = []

    det_results = detector(
        det_frame,
        conf=conf_thresh,
        imgsz=960,
        verbose=False,
        device=device
    )[0]
    for b in det_results.boxes:
        cls_id = int(b.cls[0])
        conf_val = float(b.conf[0])
        bx = b.xyxy[0].tolist()
        gbx = [bx[0] + offset_x, bx[1] + offset_y, bx[2] + offset_x, bx[3] + offset_y]
        if cls_id == 1:
            plates.append((gbx, conf_val))
        elif cls_id == 0:
            orientations.append((gbx, "Front", conf_val))
        elif cls_id == 2:
            orientations.append((gbx, "Rear", conf_val))

    return plates, orientations
