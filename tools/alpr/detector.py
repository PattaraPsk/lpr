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
