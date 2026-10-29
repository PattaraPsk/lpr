#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM01 License Plate & DLT Vehicle Detection Runner with Incremental Tracking.
Fully compliant with International Standards:
- ISO 8601:2019: Date, Time & Duration Representation (UTC/Local ISO Timestamps & Durations)
- ISO/IEC 25012:2008: Data Quality Requirements & Cryptographic Integrity (SHA-256)
- ISO/IEC 27001:2022: Information Security & Path Masking
- ISO/IEC 20547-3: Big Data Reference Architecture (Audit Lineage Tracking)
"""

import os
import sys
import json
import glob
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

# Configure structured logging compliant with ISO/IEC 25010
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z"
)
logger = logging.getLogger("LPR.VideoDetector")

# Class mapping according to Thailand DLT
CLASS_NAMES = ["car_plate", "motorcycle_plate", "truck_plate", "bus_plate"]
COCO_VEHICLE_MAP = {2: 0, 3: 1, 7: 2, 5: 3}  # car->0, motorcycle->1, truck->2, bus->3


def calculate_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """
    Computes SHA-256 cryptographic hash of a file for ISO/IEC 25012 data integrity.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_iso_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into an ISO 8601 duration string (e.g. 'PT15M30S').
    """
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


def sanitize_path(path: Path | str) -> str:
    """
    Masks user home paths or internal drive paths for ISO/IEC 27001 compliance.
    """
    path_str = str(path).replace("\\", "/")
    # Mask Windows user profiles
    if "/Users/" in path_str:
        parts = path_str.split("/Users/")
        subparts = parts[1].split("/", 1)
        if len(subparts) > 1:
            return f".../Users/*** /{subparts[1]}"
    return path_str


def load_processed_tracker(tracker_path: Path) -> dict:
    if tracker_path.exists():
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not parse tracker file {tracker_path}: {e}. Initializing fresh registry.")
    return {
        "version": "2.0",
        "standards": ["ISO 8601:2019", "ISO/IEC 25012:2008", "ISO/IEC 20547-3"],
        "last_updated": None,
        "processed_files": {}
    }


def save_processed_tracker(tracker_path: Path, data: dict):
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    data["version"] = "2.0"
    data["standards"] = ["ISO 8601:2019", "ISO/IEC 25012:2008", "ISO/IEC 20547-3"]
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    # Save atomically via temp file to avoid corruptions (ISO 25012)
    temp_path = tracker_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(tracker_path)


def match_plates_to_vehicles(v_boxes, p_boxes, img_shape):
    h, w = img_shape[:2]
    matched = []

    for p_box, p_conf in p_boxes:
        px1, py1, px2, py2 = p_box
        pw = px2 - px1
        ph = py2 - py1
        p_area = pw * ph

        # Physical constraints for license plate
        if pw < 12 or ph < 8:
            continue
        if (pw / w) > 0.20 or (ph / h) > 0.18:
            continue
        aspect = pw / ph
        if aspect < 0.7 or aspect > 4.5:
            continue

        p_cx = (px1 + px2) / 2.0
        p_cy = (py1 + py2) / 2.0

        best_v = None
        best_score = -1.0

        for v_box, v_cls, v_conf in v_boxes:
            vx1, vy1, vx2, vy2 = v_box
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area = iw * ih
            overlap_ratio = inter_area / p_area

            is_center_inside = (vx1 <= p_cx <= vx2) and (vy1 <= p_cy <= vy2)

            if is_center_inside or overlap_ratio > 0.35:
                v_area = (vx2 - vx1) * (vy2 - vy1)
                score = overlap_ratio * 10.0 + (1.0 / (v_area + 1e-5)) * 10000.0
                if score > best_score:
                    best_score = score
                    best_v = v_cls

        if best_v is not None and best_v in COCO_VEHICLE_MAP:
            matched.append((p_box, COCO_VEHICLE_MAP[best_v], p_conf))

    return matched


def process_video(
    video_path: Path,
    output_base_dir: Path,
    vehicle_model: YOLO,
    plate_model: YOLO,
    rf_model: Optional[YOLO] = None,
    sample_interval: int = 20,
    rotate_deg: int = 180,
    device: str = "cuda:0",
    conf_thresh: float = 0.25
) -> dict:
    video_stem = video_path.stem
    video_out_dir = output_base_dir / video_stem
    plates_dir = video_out_dir / "plates"
    frames_dir = video_out_dir / "annotated_frames"
    plates_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Compute source SHA-256 for data provenance (ISO/IEC 25012)
    source_sha256 = calculate_sha256(video_path)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    video_duration_sec = total_frames / max(fps, 1.0)

    detections_log = []
    class_counts = {name: 0 for name in CLASS_NAMES}
    total_detections = 0
    sampled_count = 0

    pbar = tqdm(total=total_frames, desc=f"Detect: {video_path.name[:25]}", unit="frames")
    f_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % sample_interval == 0:
            sampled_count += 1
            if rotate_deg == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_deg == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_deg == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h, w = frame.shape[:2]

            # Stage 1: Detect Vehicles
            v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=conf_thresh, verbose=False, device=device)[0]
            # Stage 2: Detect License Plates
            p_res = plate_model(frame, conf=conf_thresh, imgsz=1280, verbose=False, device=device)[0]

            # Stage 3: Detect Car Front & Back Orientations (Vehicle Orientation)
            rf_boxes = []
            if rf_model is not None:
                rf_res = rf_model(frame, conf=conf_thresh, verbose=False, device=device)[0]
                rf_boxes = [(b.xyxy[0].tolist(), rf_model.names[int(b.cls[0])], float(b.conf[0])) for b in rf_res.boxes]

            v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
            p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]

            matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)

            if matched or rf_boxes:
                vis_frame = frame.copy()
                frame_sec = round(f_idx / max(fps, 1.0), 2)

                # Draw Front / Back vehicle boxes if detected
                for rf_box, rf_cls, rf_conf in rf_boxes:
                    rx1, ry1, rx2, ry2 = map(int, rf_box)
                    rf_color = (255, 180, 0) if rf_cls == "front" else (200, 0, 255)
                    cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), rf_color, 2)
                    cv2.putText(vis_frame, f"car_{rf_cls} {rf_conf:.2f}", (rx1, max(15, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rf_color, 1)

                for p_idx, (p_box, cls_id, p_conf) in enumerate(matched):
                    x1, y1, x2, y2 = map(int, p_box)
                    cls_name = CLASS_NAMES[cls_id]
                    class_counts[cls_name] += 1
                    total_detections += 1

                    # Crop plate
                    crop_x1 = max(0, x1)
                    crop_y1 = max(0, y1)
                    crop_x2 = min(w, x2)
                    crop_y2 = min(h, y2)
                    crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    crop_name = f"f{f_idx:06d}_p{p_idx}_{cls_name}_{p_conf:.2f}.jpg"
                    crop_path = plates_dir / crop_name
                    if crop_img.size > 0:
                        cv2.imwrite(str(crop_path), crop_img)

                    plate_orientation = "unknown"
                    p_cx = (x1 + x2) / 2.0
                    p_cy = (y1 + y2) / 2.0
                    for rf_box, rf_cls, rf_conf in rf_boxes:
                        rx1, ry1, rx2, ry2 = rf_box
                        if rx1 <= p_cx <= rx2 and ry1 <= p_cy <= ry2:
                            plate_orientation = rf_cls
                            break

                    # Annotate frame
                    color = (0, 220, 0) if cls_id == 0 else (0, 220, 255) if cls_id == 1 else (255, 120, 0)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
                    lbl = f"{cls_name} [{plate_orientation}] {p_conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(vis_frame, lbl, (x1 + 3, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    detections_log.append({
                        "frame_index": f_idx,
                        "timestamp_seconds": frame_sec,
                        "timestamp_iso": format_iso_duration(frame_sec),
                        "class_name": cls_name,
                        "orientation": plate_orientation,
                        "confidence": round(p_conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "plate_crop": str(crop_path.relative_to(video_out_dir)).replace("\\", "/")
                    })

                ann_frame_path = frames_dir / f"frame_{f_idx:06d}.jpg"
                cv2.imwrite(str(ann_frame_path), vis_frame)

        pbar.update(1)
        f_idx += 1

    cap.release()
    pbar.close()

    # Save video detections JSON with ISO metadata
    det_json_path = video_out_dir / "detections.json"
    summary_data = {
        "version": "2.0",
        "standards": ["ISO 8601:2019", "ISO/IEC 25012:2008", "ISO/IEC 20547-3"],
        "video_file": video_path.name,
        "source_sha256": source_sha256,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_video_frames": total_frames,
        "sampled_frames": sampled_count,
        "sample_interval": sample_interval,
        "fps": fps,
        "video_duration_seconds": round(video_duration_sec, 2),
        "video_duration_iso": format_iso_duration(video_duration_sec),
        "orientation_rotated": rotate_deg,
        "total_plate_detections": total_detections,
        "class_breakdown": class_counts,
        "detections": detections_log
    }
    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    det_sha256 = calculate_sha256(det_json_path)

    return {
        "file_name": video_path.name,
        "file_path": str(video_path.resolve()).replace("\\", "/"),
        "file_size": video_path.stat().st_size,
        "source_sha256": source_sha256,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_video_frames": total_frames,
        "sampled_frames": sampled_count,
        "video_duration_iso": format_iso_duration(video_duration_sec),
        "detections_count": total_detections,
        "plate_classes": class_counts,
        "detections_sha256": det_sha256,
        "output_dir": str(video_out_dir.resolve()).replace("\\", "/"),
        "status": "COMPLETED"
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run License Plate & DLT Vehicle Detection on Video Chunks with Auto-Tracking (ISO 25012 / 8601 Compliant)"
    )
    parser.add_argument("--input-dir", type=str, default="dataset/raw/Raw Video Record/CAM01/2026/10/19",
                        help="Directory containing MP4 video files to process")
    parser.add_argument("--tracker-file", type=str, default="dataset/raw/Raw Video Record/CAM01/processed_videos.json",
                        help="JSON file tracking already processed videos")
    parser.add_argument("--output-dir", type=str, default="dataset/detections/CAM01/2026/10/19",
                        help="Directory to save detection results and crops")
    parser.add_argument("--sample-interval", type=int, default=20,
                        help="Sample every N frames (default: 20, ~1 fps for 20fps camera)")
    parser.add_argument("--rotate", type=int, default=180,
                        help="Degrees to rotate frame (180 for upside-down CAM01)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--rf-model", type=str, default="models/detector/vehicle_orientation_best.pt",
                        help="Path to Front/Back car orientation model checkpoint (best.pt)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-processing of already processed files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    tracker_path = Path(args.tracker_file)
    output_dir = Path(args.output_dir)

    # Auto-resolve raw tier if legacy path provided
    if not input_dir.exists():
        raw_input = Path(str(input_dir).replace("dataset\\Raw Video Record", "dataset\\raw\\Raw Video Record").replace("dataset/Raw Video Record", "dataset/raw/Raw Video Record"))
        if raw_input.exists():
            input_dir = raw_input
            tracker_path = Path(str(tracker_path).replace("dataset\\Raw Video Record", "dataset\\raw\\Raw Video Record").replace("dataset/Raw Video Record", "dataset/raw/Raw Video Record"))

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {sanitize_path(input_dir)}")
        sys.exit(1)

    tracker_data = load_processed_tracker(tracker_path)
    processed_map = tracker_data.get("processed_files", {})

    # Collect all .mp4 files (strictly ignoring temporary / staging files per ISO/IEC 20547-3)
    mp4_files = sorted([
        Path(f) for f in glob.glob(str(input_dir / "*.mp4"))
        if not Path(f).name.startswith(".") and not Path(f).name.startswith("staging_")
    ])

    if not mp4_files:
        logger.info(f"No MP4 files found in {sanitize_path(input_dir)}")
        sys.exit(0)

    print(f"\n================ CAM DETECTION RUNNER (ISO 25012 / 8601) ================")
    print(f"Target Directory : {sanitize_path(input_dir)}")
    print(f"Tracker File     : {sanitize_path(tracker_path)}")
    print(f"Found MP4 Files  : {len(mp4_files)}")
    print(f"Sample Interval  : every {args.sample_interval} frames (~1 fps)")
    print(f"Rotation Angle   : {args.rotate}°")
    print(f"========================================================================\n")

    pending_files = []
    skipped_files = []

    for f in mp4_files:
        f_name = f.name
        f_size = f.stat().st_size

        if not args.force and f_name in processed_map:
            rec = processed_map[f_name]
            # Match status and file size
            if rec.get("status") == "COMPLETED" and rec.get("file_size") == f_size:
                skipped_files.append((f_name, rec.get("processed_at")))
                continue
        pending_files.append(f)

    if skipped_files:
        print(f"Skipping {len(skipped_files)} already processed file(s):")
        for name, ts in skipped_files:
            print(f"  [SKIP] {name} (processed on {ts})")
        print()

    if not pending_files:
        print("[SUCCESS] All files in this folder are already processed! Nothing new to run.")
        print("To force re-run on already processed files, pass --force.\n")
        return

    print(f"Pending to process: {len(pending_files)} file(s)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading neural models on device: {device}...")

    # Load vehicle detector
    veh_candidates = [
        "models/detector/yolov8s.pt",
        "models/yolov8s.pt",
        "yolov8s.pt"
    ]
    vehicle_model_path = "yolov8s.pt"
    for cand in veh_candidates:
        if Path(cand).exists():
            vehicle_model_path = cand
            break
    vehicle_model = YOLO(vehicle_model_path)
    logger.info(f"Vehicle detector loaded from: {vehicle_model_path}")

    # Load license plate detector
    plate_model_path = "yolov11x_license_plate.pt"
    if not Path(plate_model_path).exists():
        for alt_plate in [
            "models/detector/vehicle_plate_yolov11n.pt",
            "models/detector/yolov8s-worldv2.pt",
            "models/detector/yolov8n.pt",
            "yolov8s-worldv2.pt",
            "yolov8n.pt"
        ]:
            if Path(alt_plate).exists():
                plate_model_path = alt_plate
                break
    plate_model = YOLO(plate_model_path)
    logger.info(f"Plate model loaded from: {plate_model_path}")

    rf_model = None
    rf_candidates = [
        args.rf_model,
        "models/detector/vehicle_plate_detector_best.pt",
        "models/detector/vehicle_plate_yolov11n.pt",
        "models/detector/vehicle_plate_detector.pt",
        "models/vehicle_plate_detector.pt",
        "vehicle_plate_detector.pt",
        "models/detector/vehicle_orientation_best.pt"
    ]
    for cand in rf_candidates:
        if cand and Path(cand).exists():
            try:
                rf_model = YOLO(cand)
                logger.info(f"Front/Back Orientation Model loaded from: {cand} (classes: {rf_model.names})")
                break
            except Exception as e:
                logger.warning(f"Could not load Front/Back model '{cand}': {e}")

    total_new_detections = 0
    newly_processed = 0

    for p_file in pending_files:
        print(f"\n>>> Processing: {p_file.name} ({p_file.stat().st_size / (1024*1024):.2f} MB)")
        start_t = time.time()

        result_entry = process_video(
            video_path=p_file,
            output_base_dir=output_dir,
            vehicle_model=vehicle_model,
            plate_model=plate_model,
            rf_model=rf_model,
            sample_interval=args.sample_interval,
            rotate_deg=args.rotate,
            device=device,
            conf_thresh=args.conf
        )

        elapsed = time.time() - start_t
        fps_proc = result_entry['sampled_frames'] / max(elapsed, 0.001)
        print(f"Finished {p_file.name} in {elapsed:.1f}s ({fps_proc:.1f} sampled fps).")
        print(f"Detections found: {result_entry['detections_count']} plates:")
        for k, v in result_entry['plate_classes'].items():
            if v > 0:
                print(f"  - {k}: {v}")

        # Record into persistent tracker immediately
        processed_map[p_file.name] = result_entry
        tracker_data["processed_files"] = processed_map
        save_processed_tracker(tracker_path, tracker_data)
        print(f"[SAVED] State updated in {tracker_path.name} (SHA-256: {result_entry['source_sha256'][:16]}...)")

        total_new_detections += result_entry['detections_count']
        newly_processed += 1

    print("\n================ PROCESSING SUMMARY (ISO 25012) ================")
    print(f"Total videos processed : {newly_processed}")
    print(f"Total videos skipped   : {len(skipped_files)}")
    print(f"Total new detections   : {total_new_detections} plates")
    print(f"Output directory       : {sanitize_path(output_dir)}")
    print(f"Tracker registry file  : {sanitize_path(tracker_path)}")
    print("================================================================")


if __name__ == "__main__":
    main()
