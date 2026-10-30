#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Length Multi-Video Vehicle & License Plate Collector with Tracking & Deduplication.
Complies with:
- ISO 8601:2019 (Date/Time & Duration Representations)
- ISO/IEC 42001:2023 (Artificial Intelligence Management System — Annex A.9.3 Operational Logging)
- ISO/IEC 25012:2008 (Data Quality & SHA-256 Cryptographic Lineage)
- ISO 3166-2:TH (Thailand Administrative Subdivisions)
- Thailand DLT Vehicle & License Plate Standards
"""

import os
import sys
import json
import csv
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Add repository root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# UTF-8 stdout configuration for Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tools.DataManipulator.app.plate_reader import PlateReader
from tools.DataManipulator.app.provinces import normalize_thai_script, match_province_value


def imwrite_safe(file_path: Path, img: np.ndarray) -> bool:
    """Saves image with complete UTF-8 path support on Windows."""
    try:
        is_success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if is_success:
            with open(file_path, "wb") as f:
                f.write(buf)
            return True
    except Exception as e:
        print(f"[WARN] Error saving image {file_path.name}: {e}")
    return False


def calculate_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 cryptographic hash (ISO/IEC 25012)."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_iso_duration(seconds: float) -> str:
    """Formats seconds into ISO 8601 duration string (e.g. 'PT1H20M15S')."""
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


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Intersection-over-Union between [x1, y1, x2, y2]."""
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


def compute_reading_score(crop_bgr: np.ndarray, ocr_res: Dict[str, Any], plate_box: List[float]) -> float:
    """
    Computes an objective reading score for selecting the optimal snapshot:
    - OCR confidence (40%)
    - Thai format validity (30%)
    - Bounding box area / resolution (20%)
    - Laplacian sharpness (10%)
    """
    conf = float(ocr_res.get("confidence", 0.0))
    d3 = ocr_res.get("d3", "")
    d4 = ocr_res.get("d4", "")
    prov = ocr_res.get("province", "")

    valid_score = 0.0
    if d3 and d4:
        valid_score += 0.5
        if len(d3) in [2, 3] and 1 <= len(d4) <= 4:
            valid_score += 0.3
    if prov:
        valid_score += 0.2

    pw = max(1.0, plate_box[2] - plate_box[0])
    ph = max(1.0, plate_box[3] - plate_box[1])
    area = pw * ph
    area_score = min(1.0, area / 20000.0)

    sharp_score = 0.5
    if crop_bgr.size > 0:
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharp_score = min(1.0, variance / 350.0)
        except Exception:
            pass

    return float(conf * 0.40 + valid_score * 0.30 + area_score * 0.20 + sharp_score * 0.10)


class VehicleTrackRecord:
    def __init__(self, track_id: int, initial_frame: int, timestamp_sec: float, veh_box: List[float], veh_class: str):
        self.track_id = track_id
        self.initial_frame = initial_frame
        self.last_frame = initial_frame
        self.first_timestamp = timestamp_sec
        self.last_timestamp = timestamp_sec
        self.veh_class = veh_class
        self.boxes: List[List[float]] = [veh_box]
        
        # Best vehicle observation
        self.best_veh_frame = initial_frame
        self.best_veh_sec = timestamp_sec
        self.best_veh_box = veh_box
        self.best_car_crop: Optional[np.ndarray] = None
        self.max_car_area = (veh_box[2] - veh_box[0]) * (veh_box[3] - veh_box[1])
        
        # License plate observation
        self.has_plate = False
        self.best_plate_crop: Optional[np.ndarray] = None
        self.best_plate_box: Optional[List[float]] = None
        self.best_plate_conf = 0.0
        self.best_reading_score = 0.0
        self.best_ocr_res: Dict[str, Any] = {}
        self.best_orientation = "unknown"
        self.plate_eval_attempts = 0

    def update_box(self, frame_idx: int, timestamp_sec: float, veh_box: List[float], car_crop: np.ndarray):
        self.last_frame = frame_idx
        self.last_timestamp = timestamp_sec
        self.boxes.append(veh_box)
        area = (veh_box[2] - veh_box[0]) * (veh_box[3] - veh_box[1])
        
        # If no plate yet, keep the largest clear car crop
        if not self.has_plate and area > self.max_car_area:
            self.max_car_area = area
            self.best_veh_frame = frame_idx
            self.best_veh_sec = timestamp_sec
            self.best_veh_box = veh_box
            self.best_car_crop = car_crop.copy()

    def update_plate(
        self,
        frame_idx: int,
        timestamp_sec: float,
        veh_box: List[float],
        car_crop: np.ndarray,
        plate_box: List[float],
        plate_crop: np.ndarray,
        plate_conf: float,
        ocr_res: Dict[str, Any],
        orientation: str,
        score: float
    ):
        self.plate_eval_attempts += 1
        if score > self.best_reading_score or not self.has_plate:
            self.has_plate = True
            self.best_reading_score = score
            self.best_plate_conf = plate_conf
            self.best_plate_crop = plate_crop.copy()
            self.best_plate_box = plate_box
            self.best_ocr_res = ocr_res
            self.best_orientation = orientation
            self.best_veh_frame = frame_idx
            self.best_veh_sec = timestamp_sec
            self.best_veh_box = veh_box
            self.best_car_crop = car_crop.copy()

    def is_stationary(self, threshold_px: float = 60.0) -> bool:
        """Determines if the vehicle was parked/stationary during tracking."""
        if len(self.boxes) < 2:
            return True
        c_x0 = (self.boxes[0][0] + self.boxes[0][2]) / 2.0
        c_y0 = (self.boxes[0][1] + self.boxes[0][3]) / 2.0
        c_x1 = (self.boxes[-1][0] + self.boxes[-1][2]) / 2.0
        c_y1 = (self.boxes[-1][1] + self.boxes[-1][3]) / 2.0
        dist = ((c_x1 - c_x0) ** 2 + (c_y1 - c_y0) ** 2) ** 0.5
        return dist < threshold_px


def process_single_video(
    video_path: Path,
    output_base_dir: Path,
    veh_model: YOLO,
    plate_model: YOLO,
    ocr_reader: PlateReader,
    sample_interval: int = 10,
    max_frames: Optional[int] = None,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Runs full length of a single video, tracking vehicles and capturing
    exactly 1 car crop + 1 license plate crop per vehicle.
    """
    video_stem = video_path.stem
    video_out_dir = output_base_dir / video_stem
    cars_out_dir = video_out_dir / "cars"
    plates_out_dir = video_out_dir / "plates"
    cars_out_dir.mkdir(parents=True, exist_ok=True)
    plates_out_dir.mkdir(parents=True, exist_ok=True)

    source_sha256 = calculate_sha256(video_path)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(fps, 1.0)

    print(f"\n{'='*75}")
    print(f"PROCESSING VIDEO: {video_path.name}")
    print(f"Resolution : {w}x{h} | FPS: {fps:.2f} | Frames: {total_frames:,}")
    print(f"Duration   : {format_iso_duration(duration_sec)} ({duration_sec/60:.2f} minutes)")
    print(f"Interval   : Every {sample_interval} frames (~{fps/sample_interval:.2f} samples/sec)")
    print(f"Output Dir : {video_out_dir}")
    print(f"{'='*75}\n")

    active_tracks: Dict[int, VehicleTrackRecord] = {}
    completed_tracks: List[VehicleTrackRecord] = []
    stationary_registry: List[VehicleTrackRecord] = []

    f_idx = 0
    sampled_count = 0
    t_start = time.time()
    pbar = tqdm(total=min(total_frames, max_frames) if max_frames else total_frames, desc=f"Track {video_stem[:20]}", unit="frames", dynamic_ncols=True)

    while cap.isOpened():
        if max_frames and f_idx >= max_frames:
            break
        if f_idx % sample_interval != 0:
            ret = cap.grab()
            if not ret:
                break
            f_idx += 1
            pbar.update(1)
            continue

        ret, frame = cap.read()
        if not ret:
            break
        sampled_count += 1
        frame_sec = round(f_idx / max(fps, 1.0), 2)

        # Vehicle detection with ByteTrack persistence
        res = veh_model.track(
            frame,
            classes=[2, 3, 5, 7],
            conf=0.35,
            imgsz=1280,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            device=device
        )[0]

        current_frame_track_ids = set()

        if res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().numpy()
            clss = res.boxes.cls.int().cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for box, trk_id, cls_id, conf in zip(boxes, ids, clss, confs):
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                # Filter out tiny distant noise
                if bw < 100 or bh < 60:
                    continue

                current_frame_track_ids.add(trk_id)
                veh_name = veh_model.names[cls_id].capitalize()

                # Add 4% contextual padding for car crop
                pad_x = int(bw * 0.04)
                pad_y = int(bh * 0.04)
                cx1 = max(0, int(box[0]) - pad_x)
                cy1 = max(0, int(box[1]) - pad_y)
                cx2 = min(w, int(box[2]) + pad_x)
                cy2 = min(h, int(box[3]) + pad_y)
                car_crop = frame[cy1:cy2, cx1:cx2]
                if car_crop.size == 0:
                    continue

                if trk_id not in active_tracks:
                    record = VehicleTrackRecord(
                        track_id=trk_id,
                        initial_frame=f_idx,
                        timestamp_sec=frame_sec,
                        veh_box=[float(c) for c in box],
                        veh_class=veh_name
                    )
                    record.best_car_crop = car_crop.copy()
                    active_tracks[trk_id] = record
                else:
                    record = active_tracks[trk_id]
                    record.update_box(f_idx, frame_sec, [float(c) for c in box], car_crop)

                # Plate & orientation detection on vehicle crop (rate-limited to 6 evaluations per track)
                if record.plate_eval_attempts < 6:
                    p_res = plate_model(car_crop, conf=0.20, verbose=False, device=device)[0]
                    best_pb, best_pc, orient_name = None, 0.0, "unknown"
                    for pb in p_res.boxes:
                        p_cls = int(pb.cls[0])
                        p_lbl = plate_model.names[p_cls]
                        if p_lbl == "LicensePlate":
                            p_conf = float(pb.conf[0])
                            if p_conf > best_pc:
                                best_pc = p_conf
                                best_pb = pb.xyxy[0].cpu().numpy()
                        elif p_lbl in ["Front", "Rear"]:
                            orient_name = p_lbl

                    if best_pb is not None:
                        # 6% padding for plate crop
                        pbw = best_pb[2] - best_pb[0]
                        pbh = best_pb[3] - best_pb[1]
                        px1 = max(0, int(best_pb[0]) - int(pbw * 0.06))
                        py1 = max(0, int(best_pb[1]) - int(pbh * 0.06))
                        px2 = min(car_crop.shape[1], int(best_pb[2]) + int(pbw * 0.06))
                        py2 = min(car_crop.shape[0], int(best_pb[3]) + int(pbh * 0.06))
                        plate_crop = car_crop[py1:py2, px1:px2]

                        if plate_crop.size > 0:
                            # Map coordinates back to full frame
                            abs_p_box = [
                                float(cx1 + px1),
                                float(cy1 + py1),
                                float(cx1 + px2),
                                float(cy1 + py2)
                            ]
                            # Run Thai OCR Reader
                            crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                            ocr_out = ocr_reader.predict(Image.fromarray(crop_rgb))
                            score = compute_reading_score(plate_crop, ocr_out, abs_p_box)
                            record.update_plate(
                                frame_idx=f_idx,
                                timestamp_sec=frame_sec,
                                veh_box=[float(c) for c in box],
                                car_crop=car_crop,
                                plate_box=abs_p_box,
                                plate_crop=plate_crop,
                                plate_conf=best_pc,
                                ocr_res=ocr_out,
                                orientation=orient_name,
                                score=score
                            )
                    else:
                        record.plate_eval_attempts += 1

        # Check for inactive tracks (> 25 frames elapsed without detection)
        expired_ids = []
        for tid, trk in active_tracks.items():
            if f_idx - trk.last_frame > 25:
                expired_ids.append(tid)

        for tid in expired_ids:
            finished = active_tracks.pop(tid)
            # Spatial deduplication for stationary/parked vehicles
            if finished.is_stationary():
                is_duplicate = False
                for stat in stationary_registry:
                    iou = compute_iou(finished.best_veh_box, stat.best_veh_box)
                    if iou > 0.50:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    stationary_registry.append(finished)
                    completed_tracks.append(finished)
            else:
                completed_tracks.append(finished)

        f_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Finalize remaining active tracks
    for tid, finished in active_tracks.items():
        if finished.is_stationary():
            is_dup = False
            for stat in stationary_registry:
                if compute_iou(finished.best_veh_box, stat.best_veh_box) > 0.50:
                    is_dup = True
                    break
            if not is_dup:
                stationary_registry.append(finished)
                completed_tracks.append(finished)
        else:
            completed_tracks.append(finished)

    # Sort tracks chronologically by initial frame
    completed_tracks.sort(key=lambda t: t.initial_frame)
    print(f"\nFinalizing {len(completed_tracks)} unique vehicle tracks for {video_path.name}...")

    # Save cropped images and format structured records
    vehicle_records: List[Dict[str, Any]] = []
    class_counts: Dict[str, int] = Counter()
    vehicles_with_plates = 0

    for trk in completed_tracks:
        class_counts[trk.veh_class] += 1
        d3 = trk.best_ocr_res.get("d3", "")
        d4 = trk.best_ocr_res.get("d4", "")
        prov = trk.best_ocr_res.get("province", "")
        full_plate = trk.best_ocr_res.get("full_plate", "")
        if full_plate:
            vehicles_with_plates += 1

        clean_plate = f"{d3}_{d4}_{prov}".strip("_")
        clean_plate = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in clean_plate)
        if not clean_plate:
            clean_plate = "no_plate"

        # 1. Save Cropped Car (Exactly Once)
        car_filename = f"{video_stem}_car_trk{trk.track_id:04d}_{trk.veh_class.lower()}_{clean_plate}.jpg"
        car_path = cars_out_dir / car_filename
        if trk.best_car_crop is not None and trk.best_car_crop.size > 0:
            imwrite_safe(car_path, trk.best_car_crop)
            rel_car_path = str(car_path.relative_to(output_base_dir)).replace("\\", "/")
        else:
            rel_car_path = None

        # 2. Save Cropped Plate (If present)
        rel_plate_path = None
        if trk.has_plate and trk.best_plate_crop is not None and trk.best_plate_crop.size > 0:
            plate_filename = f"{video_stem}_plate_trk{trk.track_id:04d}_{clean_plate}.jpg"
            plate_path = plates_out_dir / plate_filename
            imwrite_safe(plate_path, trk.best_plate_crop)
            rel_plate_path = str(plate_path.relative_to(output_base_dir)).replace("\\", "/")

        record = {
            "video_file": str(video_path.name),
            "track_id": int(trk.track_id),
            "vehicle_class": str(trk.veh_class),
            "orientation": str(trk.best_orientation),
            "first_frame": int(trk.initial_frame),
            "last_frame": int(trk.last_frame),
            "best_frame": int(trk.best_veh_frame),
            "first_timestamp_sec": round(float(trk.first_timestamp), 2),
            "best_timestamp_sec": round(float(trk.best_veh_sec), 2),
            "best_timestamp_iso": format_iso_duration(trk.best_veh_sec),
            "is_stationary": bool(trk.is_stationary()),
            "has_plate": bool(trk.has_plate),
            "plate_text": str(full_plate) if trk.has_plate and full_plate else None,
            "d3": str(d3) if trk.has_plate and d3 else None,
            "d4": str(d4) if trk.has_plate and d4 else None,
            "province": str(prov) if trk.has_plate and prov else None,
            "plate_confidence": round(float(trk.best_plate_conf), 4) if trk.has_plate else 0.0,
            "ocr_confidence": round(float(trk.best_ocr_res.get("confidence", 0.0)), 4) if trk.has_plate else 0.0,
            "reading_score": round(float(trk.best_reading_score), 4) if trk.has_plate else 0.0,
            "car_bbox_xyxy": [round(float(c), 1) for c in trk.best_veh_box],
            "plate_bbox_xyxy": [round(float(c), 1) for c in trk.best_plate_box] if trk.best_plate_box else None,
            "car_crop_image": rel_car_path,
            "plate_crop_image": rel_plate_path
        }
        vehicle_records.append(record)

    # Save video-level JSON & CSV logs
    json_path = video_out_dir / "vehicles_log.json"
    csv_path = video_out_dir / "vehicles_log.csv"

    serializable_class_counts = {str(k): int(v) for k, v in class_counts.items()}

    video_summary = {
        "version": "1.0",
        "standard": "ISO/IEC 42001:2023 / ISO 8601:2019 / ISO/IEC 25012:2008",
        "video_file": str(video_path.name),
        "source_sha256": source_sha256,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(float(time.time() - t_start), 2),
        "total_video_frames": int(total_frames),
        "sampled_frames": int(sampled_count),
        "sample_interval": int(sample_interval),
        "fps": round(float(fps), 2),
        "video_duration_iso": format_iso_duration(duration_sec),
        "total_vehicles_detected": int(len(vehicle_records)),
        "vehicles_with_plates": int(vehicles_with_plates),
        "class_breakdown": serializable_class_counts,
        "vehicles": vehicle_records
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(video_summary, f, indent=2, ensure_ascii=False)

    if vehicle_records:
        csv_cols = list(vehicle_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writeheader()
            for r in vehicle_records:
                row_copy = r.copy()
                row_copy["car_bbox_xyxy"] = json.dumps(row_copy["car_bbox_xyxy"])
                row_copy["plate_bbox_xyxy"] = json.dumps(row_copy["plate_bbox_xyxy"])
                writer.writerow(row_copy)

    print(f"[DONE] {video_path.name} processed in {(time.time() - t_start)/60:.1f} minutes.")
    print(f"       Total Unique Vehicles : {len(vehicle_records)}")
    print(f"       Vehicles with Plates  : {vehicles_with_plates}")
    print(f"       Class Breakdown       : {dict(class_counts)}")
    print(f"       Log saved to          : {json_path}")

    return video_summary


def main():
    parser = argparse.ArgumentParser(
        description="Run full length of videos, collect cropped car and license plate (each car once), create logs & summary."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"D:\Projects-anti\lpr\dataset\raw\27-Oct-2026",
        help="Directory containing the target video files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"D:\Projects-anti\lpr\dataset\snapshot\27-Oct-2026",
        help="Directory to save car and plate crops and structured logs"
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=10,
        help="Process every N frames (default: 10, ~2.5 fps for 25 fps video)"
    )
    parser.add_argument(
        "--video-pattern",
        type=str,
        default="*.mp4",
        help="Glob pattern for video files"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process per video (default: None for full length)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(list(input_dir.glob(args.video_pattern)))
    if not video_files:
        print(f"[ERROR] No video files found matching '{args.video_pattern}' in {input_dir}")
        sys.exit(1)

    print("\n" + "="*80)
    print("  THAI LPR FULL-LENGTH VIDEO VEHICLE & LICENSE PLATE COLLECTOR")
    print("  ISO/IEC 42001:2023 & ISO 8601:2019 COMPLIANT EXECUTION")
    print("="*80)
    print(f"Input Directory : {input_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Total Videos    : {len(video_files)}")
    for i, vf in enumerate(video_files, 1):
        print(f"  [{i}] {vf.name} ({vf.stat().st_size / 1e9:.2f} GB)")
    print("="*80 + "\n")

    # Load neural models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Neural Models on {device}...")

    veh_model_path = ROOT_DIR / "models" / "detector" / "yolov8s.pt"
    if not veh_model_path.exists():
        veh_model_path = ROOT_DIR / "yolov8s.pt"
    print(f"  Vehicle Detector : {veh_model_path.name}")
    veh_model = YOLO(str(veh_model_path)).to(device)

    plate_model_path = ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt"
    if not plate_model_path.exists():
        plate_model_path = ROOT_DIR / "vehicle_plate_detector.pt"
    print(f"  Plate Detector   : {plate_model_path.name}")
    plate_model = YOLO(str(plate_model_path)).to(device)

    ocr_model_path = ROOT_DIR / "models" / "reader" / "thai_plate_reader" / "best_norm_ED.pth"
    if not ocr_model_path.exists():
        ocr_model_path = ROOT_DIR / "best_norm_ED.pth"
    print(f"  Thai OCR Reader  : {ocr_model_path.name}")
    ocr_reader = PlateReader.get_instance(str(ocr_model_path))

    print("[INFO] Neural models loaded successfully.\n")

    total_pipeline_start = time.time()
    consolidated_summaries = []
    all_vehicles: List[Dict[str, Any]] = []

    for vid in video_files:
        summary = process_single_video(
            video_path=vid,
            output_base_dir=output_dir,
            veh_model=veh_model,
            plate_model=plate_model,
            ocr_reader=ocr_reader,
            sample_interval=args.sample_interval,
            max_frames=args.max_frames,
            device=device
        )
        consolidated_summaries.append(summary)
        all_vehicles.extend(summary["vehicles"])

    # Export Consolidated Summary (JSON & CSV)
    print("\n" + "="*80)
    print("GENERATING CONSOLIDATED MULTI-VIDEO AUDIT CATALOG...")
    print("="*80)

    consolidated_json_path = output_dir / "summary_report.json"
    consolidated_csv_path = output_dir / "summary_report.csv"
    markdown_report_path = output_dir / "SUMMARY_REPORT.md"

    overall_class_counts: Dict[str, int] = Counter()
    total_plates_recognized = 0
    total_stationary = 0
    for v in all_vehicles:
        overall_class_counts[v["vehicle_class"]] += 1
        if v["has_plate"] and v["plate_text"]:
            total_plates_recognized += 1
        if v["is_stationary"]:
            total_stationary += 1

    overall_summary_payload = {
        "version": "1.0",
        "governance_standard": "ISO/IEC 42001:2023 (Annex A.9.3 & A.7.2)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_pipeline_elapsed_sec": round(time.time() - total_pipeline_start, 2),
        "total_videos_processed": len(video_files),
        "total_unique_vehicles_collected": len(all_vehicles),
        "total_vehicles_with_plates": total_plates_recognized,
        "total_stationary_vehicles": total_stationary,
        "overall_class_breakdown": dict(overall_class_counts),
        "video_summaries": consolidated_summaries
    }

    with open(consolidated_json_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary_payload, f, indent=2, ensure_ascii=False)

    if all_vehicles:
        cols = list(all_vehicles[0].keys())
        with open(consolidated_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in all_vehicles:
                r_c = row.copy()
                r_c["car_bbox_xyxy"] = json.dumps(r_c["car_bbox_xyxy"])
                r_c["plate_bbox_xyxy"] = json.dumps(r_c["plate_bbox_xyxy"])
                writer.writerow(r_c)

    # Generate Executive Markdown Report
    with open(markdown_report_path, "w", encoding="utf-8") as f:
        f.write("# Thai ALPR Multi-Video Vehicle & License Plate Catalog\n\n")
        f.write(f"- **Generated At (UTC)**: `{datetime.now(timezone.utc).isoformat()}`\n")
        f.write(f"- **Execution Standard**: ISO/IEC 42001:2023 (Annex A.9.3 Operational Event Logging)\n")
        f.write(f"- **Total Videos**: `{len(video_files)}`\n")
        f.write(f"- **Total Unique Vehicles**: `{len(all_vehicles)}`\n")
        f.write(f"- **Vehicles with Thai Plate Recognized**: `{total_plates_recognized}`\n")
        f.write(f"- **Stationary / Parked Deduplicated**: `{total_stationary}`\n\n")

        f.write("## Overall Vehicle Class Breakdown\n\n")
        f.write("| Vehicle Class | Unique Count | Percentage |\n")
        f.write("| :--- | :--- | :--- |\n")
        for cls_name, cnt in overall_class_counts.most_common():
            pct = (cnt / max(1, len(all_vehicles))) * 100
            f.write(f"| **{cls_name}** | {cnt} | {pct:.1f}% |\n")

        f.write("\n## Per-Video Summary Breakdown\n\n")
        f.write("| Video File | Duration | Total Frames | Unique Vehicles | Plates Captured | Classes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for s in consolidated_summaries:
            cls_str = ", ".join(f"{k}: {v}" for k, v in s["class_breakdown"].items())
            f.write(f"| `{s['video_file']}` | {s['video_duration_iso']} | {s['total_video_frames']:,} | **{s['total_vehicles_detected']}** | {s['vehicles_with_plates']} | {cls_str} |\n")

        f.write("\n## Verification & Data Integrity (ISO/IEC 25012)\n\n")
        f.write("| Video File | SHA-256 Checksum |\n")
        f.write("| :--- | :--- |\n")
        for s in consolidated_summaries:
            f.write(f"| `{s['video_file']}` | `{s['source_sha256']}` |\n")

    print(f"\n[SUCCESS] Pipeline execution finished!")
    print(f"  Total Unique Vehicles  : {len(all_vehicles)}")
    print(f"  Plates Recognized      : {total_plates_recognized}")
    print(f"  JSON Catalog           : {consolidated_json_path}")
    print(f"  CSV Catalog            : {consolidated_csv_path}")
    print(f"  Markdown Report        : {markdown_report_path}\n")


if __name__ == "__main__":
    main()
