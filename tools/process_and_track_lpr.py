#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai License Plate Video Pipeline:
1. Video Rotation (180 degrees) with safe backup & hardware acceleration.
2. Plate & Vehicle Detection + Front/Rear Orientation.
3. Multi-Object Spatial Tracking across video frames.
4. Thai OCR Reading (Edge Thai OCR Reader).
5. Optimal Snapshot Extraction (Crop plate & scene at best reading position).
6. Comprehensive ISO 8601 / ISO 25012 / ISO 3166-2:TH Compliant Reporting.
"""

import os
import sys
import glob
import json
import csv
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import PlateReader
from tools.DataManipulator.app.provinces import match_province_value, get_iso_code, normalize_thai_script
from tools.DataManipulator.app.plate_types import match_plate_type

# =============================================================================
# Helper Utilities & Standards Compliance
# =============================================================================

def calculate_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def format_iso_duration(seconds: float) -> str:
    """Formats seconds into ISO 8601 duration string (e.g. 'PT1M30S')."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    parts = ["PT"]
    if hours > 0:
        parts.append(f"{hours}H")
    if minutes > 0:
        parts.append(f"{minutes}M")
    parts.append(f"{secs}S")
    return "".join(parts)

def box_iou(box1: List[float], box2: List[float]) -> float:
    """Computes Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0.0 else 0.0

def box_center_dist(box1: List[float], box2: List[float]) -> float:
    """Computes Euclidean distance between centers of two boxes."""
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

# Load Thai font for annotations
THAI_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\leelawad.ttf",
    r"C:\Windows\Fonts\arial.ttf"
]
_THAI_FONT_PATH = None
for _fp in THAI_FONT_CANDIDATES:
    if os.path.exists(_fp):
        _THAI_FONT_PATH = _fp
        break

def draw_thai_text(
    cv_img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 24,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    bg_color: Optional[Tuple[int, int, int]] = (0, 255, 0),
    padding: int = 4
) -> np.ndarray:
    """
    Renders text (including Thai Unicode) onto an OpenCV BGR image using PIL.
    Colors are given in BGR.
    """
    if not text:
        return cv_img
    
    # Convert BGR to RGB PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype(_THAI_FONT_PATH, font_size) if _THAI_FONT_PATH else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        
    # Measure text bounding box
    bbox = draw.textbbox((pos[0], pos[1]), text, font=font)
    x1, y1, x2, y2 = bbox
    
    # Draw background box if specified
    if bg_color is not None:
        # bg_color is BGR, convert to RGB for PIL
        rgb_bg = (bg_color[2], bg_color[1], bg_color[0])
        draw.rectangle((x1 - padding, y1 - padding, x2 + padding, y2 + padding), fill=rgb_bg)
        
    # Draw text
    rgb_text = (text_color[2], text_color[1], text_color[0])
    draw.text((pos[0], pos[1]), text, fill=rgb_text, font=font)
    
    # Convert back to BGR OpenCV image
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# =============================================================================
# Step 1: Video Rotation (180 Degrees)
# =============================================================================

def rotate_video_file(src_video: Path, backup_dir: Path) -> bool:
    """
    Safely backs up original video and rotates 180 degrees using FFmpeg (h264_nvenc / libx264).
    """
    backup_path = backup_dir / src_video.name
    temp_rot_path = src_video.with_suffix(".rot.tmp.mp4")

    # Step 1: Backup if not already backed up
    if not backup_path.exists():
        shutil.copy2(src_video, backup_path)
        print(f"  [BACKUP] Copied original to: {backup_path.name}")
    else:
        print(f"  [BACKUP] Existing backup verified for: {backup_path.name}")

    # Use backup as the input source to ensure clean rotation
    input_file = backup_path

    # Try FFmpeg with hardware acceleration (nvenc)
    cmd_nvenc = [
        "ffmpeg", "-y", "-hwaccel", "cuda",
        "-i", str(input_file),
        "-vf", "hflip,vflip",
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "20",
        "-c:a", "copy",
        str(temp_rot_path)
    ]

    # Fallback to software encoding (libx264)
    cmd_cpu = [
        "ffmpeg", "-y",
        "-i", str(input_file),
        "-vf", "hflip,vflip",
        "-c:v", "libx264", "-crf", "19", "-preset", "fast",
        "-c:a", "copy",
        str(temp_rot_path)
    ]

    success = False
    try:
        res = subprocess.run(cmd_nvenc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        if res.returncode == 0 and temp_rot_path.exists() and temp_rot_path.stat().st_size > 1024:
            success = True
    except Exception as e:
        print(f"  [WARN] NVENC rotation error: {e}, attempting CPU fallback...")

    if not success:
        try:
            res = subprocess.run(cmd_cpu, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if res.returncode == 0 and temp_rot_path.exists() and temp_rot_path.stat().st_size > 1024:
                success = True
        except Exception as e:
            print(f"  [ERROR] CPU rotation also failed: {e}")

    if success:
        # Atomic replace
        if src_video.exists():
            src_video.unlink()
        temp_rot_path.replace(src_video)
        print(f"  [ROTATED 180°] Successfully rotated and saved: {src_video.name}")
        return True
    else:
        if temp_rot_path.exists():
            temp_rot_path.unlink()
        print(f"  [FAILED] Could not rotate {src_video.name}")
        return False

# =============================================================================
# Step 2: Spatial Multi-Object Plate Tracker
# =============================================================================

class TrackedPlateObservation:
    def __init__(
        self,
        frame_idx: int,
        timestamp_sec: float,
        plate_box: List[float],
        plate_conf: float,
        vehicle_type: str,
        orientation: str,
        vehicle_box: Optional[List[float]],
        plate_crop_bgr: np.ndarray,
        ocr_result: Dict[str, Any],
        reading_score: float
    ):
        self.frame_idx = frame_idx
        self.timestamp_sec = timestamp_sec
        self.plate_box = plate_box
        self.plate_conf = plate_conf
        self.vehicle_type = vehicle_type
        self.orientation = orientation
        self.vehicle_box = vehicle_box
        self.plate_crop_bgr = plate_crop_bgr
        self.ocr_result = ocr_result
        self.reading_score = reading_score

class PlateTrack:
    def __init__(self, track_id: int, initial_obs: TrackedPlateObservation):
        self.track_id = track_id
        self.observations: List[TrackedPlateObservation] = [initial_obs]
        self.last_frame = initial_obs.frame_idx
        self.last_box = initial_obs.plate_box
        self.best_obs: TrackedPlateObservation = initial_obs

    def update(self, obs: TrackedPlateObservation):
        self.observations.append(obs)
        self.last_frame = obs.frame_idx
        self.last_box = obs.plate_box
        if obs.reading_score > self.best_obs.reading_score:
            self.best_obs = obs

    def get_consensus_text(self) -> Dict[str, Any]:
        """Calculates consensus / best reading over all track observations."""
        d3_list = [o.ocr_result.get("d3", "") for o in self.observations if o.ocr_result.get("d3")]
        d4_list = [o.ocr_result.get("d4", "") for o in self.observations if o.ocr_result.get("d4")]
        prov_list = [o.ocr_result.get("province", "") for o in self.observations if o.ocr_result.get("province")]

        best_d3 = Counter(d3_list).most_common(1)[0][0] if d3_list else self.best_obs.ocr_result.get("d3", "")
        best_d4 = Counter(d4_list).most_common(1)[0][0] if d4_list else self.best_obs.ocr_result.get("d4", "")
        best_prov = Counter(prov_list).most_common(1)[0][0] if prov_list else self.best_obs.ocr_result.get("province", "")

        full_text = f"{best_d3} {best_d4} {best_prov}".strip()
        return {
            "d3": best_d3,
            "d4": best_d4,
            "province": best_prov,
            "full_plate": full_text
        }

# =============================================================================
# Step 3: Best Reading Score Calculation
# =============================================================================

def compute_reading_score(
    crop_bgr: np.ndarray,
    ocr_res: Dict[str, Any],
    box: List[float],
    frame_w: int,
    frame_h: int
) -> float:
    """
    Evaluates how suitable a frame/crop is as the 'best reading position':
    1. OCR Confidence: higher recognition probability.
    2. Format Validity: valid Thai prefix (d3) and digits (d4) and province.
    3. Resolution/Area: plate has sufficient pixel size.
    4. Boundary Margin: plate is not truncated by image edges.
    5. Sharpness: Laplacian variance of grayscale crop.
    """
    conf = float(ocr_res.get("confidence", 0.0))
    d3 = ocr_res.get("d3", "")
    d4 = ocr_res.get("d4", "")
    prov = ocr_res.get("province", "")

    # Format validity bonus
    valid_score = 0.0
    if d3 and d4:
        valid_score += 0.5
        if len(d3) in [2, 3] and 1 <= len(d4) <= 4:
            valid_score += 0.3
    if prov:
        valid_score += 0.2

    # Area score (ideal area ~ 15,000 to 50,000 px)
    pw = max(1.0, box[2] - box[0])
    ph = max(1.0, box[3] - box[1])
    area = pw * ph
    area_score = min(1.0, area / 25000.0)

    # Edge proximity penalty
    dist_edge = min(box[0], box[1], frame_w - box[2], frame_h - box[3])
    edge_score = min(1.0, max(0.0, dist_edge / 40.0))

    # Sharpness score
    sharp_score = 0.5
    if crop_bgr.size > 0:
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharp_score = min(1.0, variance / 400.0)
        except Exception:
            pass

    # Weighted composite score
    total_score = (
        conf * 0.40 +
        valid_score * 0.30 +
        area_score * 0.15 +
        edge_score * 0.10 +
        sharp_score * 0.05
    )
    return float(total_score)

# =============================================================================
# Step 4: Video Processing & Tracking Runner
# =============================================================================

def process_video_tracks(
    video_path: Path,
    output_dir: Path,
    detector_model: YOLO,
    veh_model: YOLO,
    ocr_reader: PlateReader,
    sample_interval: int = 2,
    device: str = "cuda:0"
) -> List[Dict[str, Any]]:
    """
    Processes a rotated video, tracks plates, evaluates reading position,
    and extracts optimal crops and scene snapshots.
    """
    video_stem = video_path.stem
    plates_out_dir = output_dir / "plates"
    scenes_out_dir = output_dir / "scenes"
    plates_out_dir.mkdir(parents=True, exist_ok=True)
    scenes_out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    active_tracks: List[PlateTrack] = []
    finished_tracks: List[PlateTrack] = []
    next_track_id = 1

    # Keep a cache of best scene frames in memory or reload on finish
    # To save memory, store (best_frame_idx, track_id) and extract in second pass or cache
    best_scenes_cache: Dict[int, np.ndarray] = {}

    pbar = tqdm(total=total_frames, desc=f"Track & Read {video_path.name[:25]}", unit="frames")
    f_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % sample_interval == 0:
            h, w = frame.shape[:2]
            frame_sec = round(f_idx / max(fps, 1.0), 2)

            # 1. Run Detector (vehicle_plate_yolov11n)
            det_res = detector_model(frame, conf=0.25, imgsz=1280, verbose=False, device=device)[0]

            # 2. Run Vehicle Classifier (yolov8s)
            veh_res = veh_model(frame, classes=[2, 3, 5, 7], conf=0.25, verbose=False, device=device)[0]

            # Parse detections
            plates: List[Tuple[List[float], float]] = []
            orientations: List[Tuple[List[float], str, float]] = []
            for b in det_res.boxes:
                cls_id = int(b.cls[0])
                c_val = float(b.conf[0])
                box = b.xyxy[0].tolist()
                if cls_id == 1:  # LicensePlate
                    plates.append((box, c_val))
                elif cls_id == 0:
                    orientations.append((box, "Front", c_val))
                elif cls_id == 2:
                    orientations.append((box, "Rear", c_val))

            # Vehicles (COCO map: 2: Car, 3: Motorcycle, 7: Truck, 5: Bus)
            veh_map = {2: "Car", 3: "Motorcycle", 7: "Truck", 5: "Bus"}
            vehicles: List[Tuple[List[float], str, float]] = []
            for b in veh_res.boxes:
                cls_id = int(b.cls[0])
                if cls_id in veh_map:
                    vehicles.append((b.xyxy[0].tolist(), veh_map[cls_id], float(b.conf[0])))

            current_frame_obs: List[TrackedPlateObservation] = []

            for p_box, p_conf in plates:
                px1, py1, px2, py2 = p_box
                pw = px2 - px1
                ph = py2 - py1

                # Filter out microscopic or degenerate detections
                if pw < 14 or ph < 8 or (pw / w) > 0.4:
                    continue

                # Match Vehicle
                matched_veh = "Unknown"
                matched_veh_box = None
                p_cx = (px1 + px2) / 2.0
                p_cy = (py1 + py2) / 2.0
                for v_box, v_name, v_conf in vehicles:
                    vx1, vy1, vx2, vy2 = v_box
                    if (vx1 <= p_cx <= vx2) and (vy1 <= p_cy <= vy2):
                        matched_veh = v_name
                        matched_veh_box = v_box
                        break

                # Match Orientation (Front / Rear)
                matched_orient = "unknown"
                for o_box, o_name, o_conf in orientations:
                    ox1, oy1, ox2, oy2 = o_box
                    if (ox1 <= p_cx <= ox2) and (oy1 <= p_cy <= oy2):
                        matched_orient = o_name
                        break

                # Crop plate with 8% padding for contextual clarity
                pad_x = int(pw * 0.08)
                pad_y = int(ph * 0.08)
                crop_x1 = max(0, int(px1) - pad_x)
                crop_y1 = max(0, int(py1) - pad_y)
                crop_x2 = min(w, int(px2) + pad_x)
                crop_y2 = min(h, int(py2) + pad_y)
                crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop_bgr.size == 0:
                    continue

                # OCR Reading
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                ocr_res = ocr_reader.predict(pil_crop)

                # Reading score
                reading_score = compute_reading_score(crop_bgr, ocr_res, p_box, w, h)

                obs = TrackedPlateObservation(
                    frame_idx=f_idx,
                    timestamp_sec=frame_sec,
                    plate_box=p_box,
                    plate_conf=p_conf,
                    vehicle_type=matched_veh,
                    orientation=matched_orient,
                    vehicle_box=matched_veh_box,
                    plate_crop_bgr=crop_bgr.copy(),
                    ocr_result=ocr_res,
                    reading_score=reading_score
                )
                current_frame_obs.append(obs)

            # Spatial association to active tracks
            unmatched_obs = []
            for obs in current_frame_obs:
                best_track: Optional[PlateTrack] = None
                best_match_score = -1.0

                for track in active_tracks:
                    iou = box_iou(obs.plate_box, track.last_box)
                    dist = box_center_dist(obs.plate_box, track.last_box)
                    # Frames elapsed
                    frame_diff = obs.frame_idx - track.last_frame
                    if frame_diff <= 30:  # within 1.5 seconds
                        # If boxes overlap or centers are close relative to width
                        ref_w = max(1.0, obs.plate_box[2] - obs.plate_box[0])
                        normalized_dist = dist / ref_w
                        if iou > 0.20 or normalized_dist < 2.5:
                            match_score = iou * 2.0 + max(0.0, 3.0 - normalized_dist)
                            if match_score > best_match_score:
                                best_match_score = match_score
                                best_track = track

                if best_track is not None:
                    best_track.update(obs)
                else:
                    unmatched_obs.append(obs)

            # Create new tracks for unmatched observations
            for obs in unmatched_obs:
                new_track = PlateTrack(track_id=next_track_id, initial_obs=obs)
                next_track_id += 1
                active_tracks.append(new_track)

            # Check for active tracks that have expired (> 25 frames inactive)
            still_active = []
            for track in active_tracks:
                if f_idx - track.last_frame > 25:
                    finished_tracks.append(track)
                else:
                    still_active.append(track)
            active_tracks = still_active

        pbar.update(1)
        f_idx += 1

    cap.release()
    pbar.close()

    # Move any remaining active tracks to finished
    finished_tracks.extend(active_tracks)

    # Filter out spurious tracks (e.g. tracks with only 1 observation and low confidence)
    valid_tracks = [t for t in finished_tracks if len(t.observations) >= 2 or t.best_obs.plate_conf > 0.70]
    print(f"[{video_path.name}] Found {len(valid_tracks)} valid tracked plate passage(s).")

    # Second pass: Extract annotated scene snapshots for the best reading frame of each track
    # Group tracks by their best frame index
    frame_to_tracks: Dict[int, List[PlateTrack]] = {}
    for track in valid_tracks:
        bf = track.best_obs.frame_idx
        frame_to_tracks.setdefault(bf, []).append(track)

    if frame_to_tracks:
        cap_scenes = cv2.VideoCapture(str(video_path))
        sorted_frame_indices = sorted(frame_to_tracks.keys())
        current_req_idx = 0
        cur_frame = 0

        while cap_scenes.isOpened() and current_req_idx < len(sorted_frame_indices):
            ret, scene_frame = cap_scenes.read()
            if not ret:
                break
            if cur_frame == sorted_frame_indices[current_req_idx]:
                for trk in frame_to_tracks[cur_frame]:
                    best_scenes_cache[trk.track_id] = scene_frame.copy()
                current_req_idx += 1
            cur_frame += 1
        cap_scenes.release()

    results_list: List[Dict[str, Any]] = []

    for trk in valid_tracks:
        best = trk.best_obs
        consensus = trk.get_consensus_text()
        d3 = consensus["d3"] or best.ocr_result.get("d3", "")
        d4 = consensus["d4"] or best.ocr_result.get("d4", "")
        prov = consensus["province"] or best.ocr_result.get("province", "")
        full_plate = consensus["full_plate"] or f"{d3} {d4} {prov}".strip() or "UNKNOWN_PLATE"

        # Sanitize for filename
        clean_text = f"{d3}_{d4}_{prov}".strip("_")
        clean_text = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in clean_text)
        if not clean_text:
            clean_text = "plate"

        # 1. Save Best Plate Crop Snapshot
        crop_filename = f"{video_stem}_trk{trk.track_id:02d}_{clean_text}.jpg"
        crop_path = plates_out_dir / crop_filename
        cv2.imwrite(str(crop_path), best.plate_crop_bgr)

        # 2. Save Best Scene Snapshot with Annotations
        scene_filename = f"{video_stem}_trk{trk.track_id:02d}_{clean_text}_scene.jpg"
        scene_path = scenes_out_dir / scene_filename
        
        scene_img = best_scenes_cache.get(trk.track_id)
        if scene_img is not None:
            vis_scene = scene_img.copy()
            x1, y1, x2, y2 = map(int, best.plate_box)

            # Draw vehicle box if available
            if best.vehicle_box is not None:
                vx1, vy1, vx2, vy2 = map(int, best.vehicle_box)
                cv2.rectangle(vis_scene, (vx1, vy1), (vx2, vy2), (255, 180, 0), 2)
                veh_lbl = f"{best.vehicle_type} ({best.orientation})"
                cv2.putText(vis_scene, veh_lbl, (vx1, max(20, vy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

            # Draw license plate box
            cv2.rectangle(vis_scene, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plate_tag = f"{full_plate} [OCR: {best.ocr_result.get('confidence', 0.0):.2f}]"
            # Render Thai label with PIL
            vis_scene = draw_thai_text(
                vis_scene,
                text=plate_tag,
                pos=(x1, max(0, y1 - 32)),
                font_size=26,
                text_color=(0, 0, 0),
                bg_color=(0, 255, 0),
                padding=4
            )

            # Add watermark header with ISO timestamp & track metadata
            header_text = f"Track #{trk.track_id} | Best Frame: {best.frame_idx} ({best.timestamp_sec:.2f}s) | Score: {best.reading_score:.3f}"
            vis_scene = draw_thai_text(
                vis_scene,
                text=header_text,
                pos=(20, 15),
                font_size=22,
                text_color=(0, 240, 255),
                bg_color=(20, 20, 20),
                padding=6
            )

            cv2.imwrite(str(scene_path), vis_scene)

        # Build structured result item
        record = {
            "video_file": video_path.name,
            "track_id": trk.track_id,
            "total_frames_tracked": len(trk.observations),
            "best_frame_idx": best.frame_idx,
            "best_timestamp_sec": best.timestamp_sec,
            "best_timestamp_iso": format_iso_duration(best.timestamp_sec),
            "reading_score": round(best.reading_score, 4),
            "d3": d3,
            "d4": d4,
            "province": prov,
            "full_plate": full_plate,
            "ocr_confidence": round(best.ocr_result.get("confidence", 0.0), 4),
            "vehicle_type": best.vehicle_type,
            "orientation": best.orientation,
            "plate_bbox_xyxy": [round(c, 1) for c in best.plate_box],
            "crop_snapshot": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
            "scene_snapshot": str(scene_path.relative_to(output_dir)).replace("\\", "/") if scene_img is not None else None
        }
        results_list.append(record)

    return results_list

# =============================================================================
# Main Orchestration CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Rotate videos 180°, run detection, tracking, OCR reading, and extract optimal snapshots."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\Projects-anti\lpr\dataset\raw\19-Oct-2026",
        help="Directory containing target MP4 videos"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=r"D:\Projects-anti\lpr\dataset\raw\19-Oct-2026_raw_backup",
        help="Directory for safe original video backup"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Projects-anti\lpr\dataset\snapshot\19-Oct-2026_snapshots",
        help="Directory for snapshots and report artifacts"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=2,
        help="Process every N frames (default: 2, ~10 fps for 20fps camera)"
    )
    parser.add_argument(
        "--skip-rotation",
        action="store_true",
        help="Skip video rotation if already rotated"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    backup_dir = Path(args.backup_dir)
    output_dir = Path(args.output_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted([f for f in input_dir.glob("*.mp4") if not f.name.startswith(".") and ".tmp" not in f.name])
    if not video_files:
        print(f"[ERROR] No MP4 files found in {input_dir}")
        sys.exit(1)

    print("\n================ THAI LPR VIDEO PROCESSOR & TRACKER ================")
    print(f"Target Directory  : {input_dir}")
    print(f"Backup Directory  : {backup_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Total Videos      : {len(video_files)}")
    print(f"Sample Interval   : every {args.sample_interval} frames")
    print("=====================================================================\n")

    # Step 1: Rotate Videos 180°
    if not args.skip_rotation:
        print(">>> STEP 1: Rotating Videos 180° with Safe Backup...")
        for vid in video_files:
            print(f"Processing rotation for: {vid.name}")
            rotate_video_file(vid, backup_dir)
        print("[SUCCESS] All videos successfully backed up and rotated 180°.\n")
    else:
        print("[INFO] Skipping video rotation per flag.\n")

    # Step 2: Load Neural Models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f">>> STEP 2: Loading Neural Models on {device}...")

    det_model_path = ROOT_DIR / "models" / "detector" / "vehicle_plate_detector_best.pt"
    if not det_model_path.exists():
        det_model_path = ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt"
    if not det_model_path.exists():
        det_model_path = ROOT_DIR / "vehicle_plate_detector.pt"
    print(f"  Detector  : {det_model_path.name}")
    detector_model = YOLO(str(det_model_path)).to(device)

    veh_model_path = ROOT_DIR / "models" / "detector" / "yolov8s.pt"
    if not veh_model_path.exists():
        veh_model_path = ROOT_DIR / "yolov8s.pt"
    print(f"  Classifier: {veh_model_path.name}")
    veh_model = YOLO(str(veh_model_path)).to(device)

    ocr_model_path = ROOT_DIR / "models" / "reader" / "edge_reader" / "best_norm_ED.pth"
    if not ocr_model_path.exists():
        ocr_model_path = ROOT_DIR / "best_norm_ED.pth"
    print(f"  OCR Reader: {ocr_model_path.name}")
    ocr_reader = PlateReader.get_instance(str(ocr_model_path))

    print("[SUCCESS] Neural models ready.\n")

    # Step 3: Run Video Tracking & Snapshot Extraction
    print(">>> STEP 3: Tracking & Extracting Best Plate Snapshots across all videos...")
    all_video_results: List[Dict[str, Any]] = []

    for vid in video_files:
        print(f"\nProcessing Video: {vid.name}")
        t0 = datetime.now(timezone.utc)
        vid_results = process_video_tracks(
            video_path=vid,
            output_dir=output_dir,
            detector_model=detector_model,
            veh_model=veh_model,
            ocr_reader=ocr_reader,
            sample_interval=args.sample_interval,
            device=device
        )
        t_elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        print(f"Completed {vid.name} in {t_elapsed:.1f}s: {len(vid_results)} plate track snapshot(s) generated.")
        all_video_results.extend(vid_results)

    # Step 4: Save Summary Catalog & Metadata (JSON & CSV)
    print("\n>>> STEP 4: Exporting Summary Catalog (JSON & CSV)...")
    json_path = output_dir / "results_summary.json"
    csv_path = output_dir / "results_summary.csv"

    summary_payload = {
        "version": "1.0",
        "standards": ["ISO 8601:2019", "ISO/IEC 25012:2008", "ISO 3166-2:TH", "ISO/IEC 10646"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(input_dir.resolve()).replace("\\", "/"),
        "total_videos_processed": len(video_files),
        "total_plates_captured": len(all_video_results),
        "tracks": all_video_results
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # Write CSV
    csv_fields = [
        "video_file", "track_id", "total_frames_tracked", "best_frame_idx",
        "best_timestamp_sec", "best_timestamp_iso", "reading_score",
        "d3", "d4", "province", "full_plate", "ocr_confidence",
        "vehicle_type", "orientation", "plate_bbox_xyxy", "crop_snapshot", "scene_snapshot"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_video_results:
            row_dict = r.copy()
            row_dict["plate_bbox_xyxy"] = json.dumps(row_dict["plate_bbox_xyxy"])
            writer.writerow(row_dict)

    print(f"  [SAVED] JSON Catalog : {json_path}")
    print(f"  [SAVED] CSV Catalog  : {csv_path}")
    print(f"\n[DONE] Pipeline execution completed successfully! Total snapshots: {len(all_video_results)}.\n")

if __name__ == "__main__":
    main()
