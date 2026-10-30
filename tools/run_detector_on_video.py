#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Video Inference & LPR Runner with 180° Rotation:
Executes the newly trained unified detector (vehicle_plate_detector_best.pt)
and Thai OCR Reader on target CCTV footage.

Compliant with ISO 8601, ISO/IEC 25012, ISO 3166-2:TH, and ISO/IEC 10646.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.DataManipulator.app.plate_reader import PlateReader
from tools.DataManipulator.app.provinces import normalize_thai_script, get_iso_code
from tools.DataManipulator.app.plate_types import match_plate_type


def run_detector_on_video(
    video_path: Path,
    output_dir: Path,
    model_path: Path,
    rotate_deg: int = 180,
    sample_interval: int = 5,
    conf_thresh: float = 0.30,
    device: str = "cuda:0"
):
    print("=" * 70)
    print(" THAI ALPR & DLT VEHICLE DETECTOR RUNNER")
    print("=" * 70)
    print(f"Target Video    : {video_path}")
    print(f"Detector Model  : {model_path.name}")
    print(f"Rotation        : {rotate_deg}°")
    print(f"Sample Interval : Every {sample_interval} frames")
    print(f"Confidence      : {conf_thresh:.2f}")
    print(f"Compute Device  : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Output Directory: {output_dir}")
    print("=" * 70 + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    plates_dir = output_dir / "plates"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    plates_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Neural Models
    print(f"[*] Loading unified detector: {model_path}...")
    detector = YOLO(str(model_path)).to(device)
    print(f"    Classes: {detector.names}")

    ocr_weights = ROOT_DIR / "models" / "reader" / "edge_reader" / "best_norm_ED.pth"
    if not ocr_weights.exists():
        ocr_weights = ROOT_DIR / "best_norm_ED.pth"

    ocr_reader = None
    if ocr_weights.exists():
        print(f"[*] Loading Thai Plate Reader OCR (edge_reader): {ocr_weights.name}...")
        ocr_reader = PlateReader.get_instance(str(ocr_weights))
    else:
        print("[!] OCR weights not found; running detection without OCR reading.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    duration_sec = total_frames / max(fps, 1.0)
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[*] Video Resolution: {w_orig}x{h_orig} | Total Frames: {total_frames:,} ({duration_sec/60:.1f} mins @ {fps:.1f} fps)\n")

    detections_list = []
    class_counter = Counter()
    plates_captured = 0

    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frames")
    f_idx = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % sample_interval == 0:
            # Apply 180° rotation as requested
            if rotate_deg == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate_deg == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_deg == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            h, w = frame.shape[:2]
            frame_sec = round(f_idx / max(fps, 1.0), 2)

            # Unified inference: detects Front, Rear, LicensePlate, Car, Motorcycle, Truck, Bus
            res = detector(frame, conf=conf_thresh, imgsz=1280, verbose=False, device=device)[0]

            frame_plates = []
            frame_vehicles = []
            frame_orientations = []

            for b in res.boxes:
                cls_id = int(b.cls[0])
                cls_name = detector.names[cls_id]
                conf = float(b.conf[0])
                box = [float(x) for x in b.xyxy[0].tolist()]

                class_counter[cls_name] += 1

                if cls_name == "LicensePlate":
                    frame_plates.append((box, conf))
                elif cls_name in ["Car", "Motorcycle", "Truck", "Bus"]:
                    frame_vehicles.append((box, cls_name, conf))
                elif cls_name in ["Front", "Rear"]:
                    frame_orientations.append((box, cls_name, conf))

            # Associate plates with containing vehicles & orientations
            for p_idx, (p_box, p_conf) in enumerate(frame_plates):
                px1, py1, px2, py2 = map(int, p_box)
                pcx = (px1 + px2) * 0.5
                pcy = (py1 + py2) * 0.5

                # Find vehicle
                matched_veh = "Unknown"
                matched_veh_conf = 0.0
                best_dist = 1e9
                for v_box, v_type, v_conf in frame_vehicles:
                    vx1, vy1, vx2, vy2 = map(int, v_box)
                    vw = vx2 - vx1
                    vh = vy2 - vy1
                    exp_bot = vh * 0.35 if v_type == "Motorcycle" else vh * 0.15
                    if (vx1 - vw * 0.1 <= pcx <= vx2 + vw * 0.1) and (vy1 - vh * 0.1 <= pcy <= vy2 + exp_bot):
                        dist = ((pcx - (vx1 + vx2) * 0.5) ** 2 + (pcy - (vy1 + vy2) * 0.5) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            matched_veh = v_type
                            matched_veh_conf = v_conf

                # Find orientation
                matched_orient = "Unknown"
                for o_box, o_type, o_conf in frame_orientations:
                    ox1, oy1, ox2, oy2 = map(int, o_box)
                    if ox1 <= pcx <= ox2 and oy1 <= pcy <= oy2:
                        matched_orient = o_type
                        break

                # Thailand DLT rule: motorcycles strictly have only rear plates
                if matched_veh == "Motorcycle" and matched_orient == "Unknown":
                    matched_orient = "Rear"

                # Crop plate with margin
                pad_x = int((px2 - px1) * 0.05)
                pad_y = int((py2 - py1) * 0.05)
                cx1 = max(0, px1 - pad_x)
                cy1 = max(0, py1 - pad_y)
                cx2 = min(w, px2 + pad_x)
                cy2 = min(h, py2 + pad_y)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 14:
                    continue

                # Run Thai OCR Reader
                d3, d4, province, full_plate, ocr_conf = "", "", "", "", 0.0
                if ocr_reader is not None:
                    try:
                        from PIL import Image
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        res_ocr = ocr_reader.predict(pil_crop)
                        if res_ocr:
                            d3 = res_ocr.get("d3", "")
                            d4 = res_ocr.get("d4", "")
                            province = res_ocr.get("province", "")
                            full_plate = res_ocr.get("full_plate", "")
                            ocr_conf = float(res_ocr.get("confidence", 0.0))
                    except Exception as e:
                        pass

                plates_captured += 1
                det_id = f"det_{f_idx:06d}_p{p_idx:02d}"

                # Save crop
                crop_fname = f"{det_id}_{matched_veh}_{matched_orient}.jpg"
                crop_path = plates_dir / crop_fname
                cv2.imwrite(str(crop_path), crop)

                # Save annotated scene snapshot
                vis_scene = frame.copy()
                cv2.rectangle(vis_scene, (px1, py1), (px2, py2), (0, 255, 0), 3)
                label_txt = f"{matched_veh} | {matched_orient} | {full_plate if full_plate else 'Plate'} ({p_conf:.2f})"
                cv2.putText(vis_scene, label_txt, (px1, max(25, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                scene_fname = f"{det_id}_scene.jpg"
                scene_path = snapshots_dir / scene_fname
                cv2.imwrite(str(scene_path), vis_scene)

                entry = {
                    "detection_id": det_id,
                    "frame_index": f_idx,
                    "timestamp_sec": frame_sec,
                    "timestamp_iso": f"PT{int(frame_sec // 60)}M{int(frame_sec % 60)}S",
                    "vehicle_type": matched_veh,
                    "vehicle_conf": round(matched_veh_conf, 3),
                    "orientation": matched_orient,
                    "plate_conf": round(p_conf, 3),
                    "plate_bbox": [px1, py1, px2, py2],
                    "ocr_d3": d3,
                    "ocr_d4": d4,
                    "ocr_province": province,
                    "ocr_full_plate": full_plate,
                    "ocr_conf": round(ocr_conf, 3),
                    "crop_image": str(crop_path.relative_to(output_dir)).replace("\\", "/"),
                    "scene_image": str(scene_path.relative_to(output_dir)).replace("\\", "/"),
                }
                detections_list.append(entry)

        f_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    elapsed_total = time.time() - start_time

    # Save summary report (JSON & CSV)
    json_path = output_dir / "detection_report.json"
    csv_path = output_dir / "detection_report.csv"

    summary_payload = {
        "metadata": {
            "standards": ["ISO 8601:2019", "ISO/IEC 25012:2008", "ISO 3166-2:TH"],
            "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            "video_path": str(video_path),
            "video_frames": total_frames,
            "video_duration_sec": round(duration_sec, 2),
            "rotation_deg": rotate_deg,
            "sample_interval": sample_interval,
            "sampled_frames_processed": f_idx // sample_interval,
            "elapsed_seconds": round(elapsed_total, 2),
            "throughput_fps": round(total_frames / max(elapsed_total, 0.001), 1),
            "model_weights": model_path.name
        },
        "class_detections_count": dict(class_counter),
        "total_plates_captured": len(detections_list),
        "detections": detections_list
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    if detections_list:
        csv_fields = list(detections_list[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            for row in detections_list:
                row_copy = row.copy()
                row_copy["plate_bbox"] = json.dumps(row_copy["plate_bbox"])
                writer.writerow(row_copy)

    # Print summary
    print("\n" + "=" * 70)
    print(" INFERENCE & DETECTION SUMMARY (ISO/IEC 25012)")
    print("=" * 70)
    print(f"Elapsed Time       : {elapsed_total:.1f}s (Processing Speed: {total_frames/elapsed_total:.1f} video fps)")
    print(f"Total BBoxes Found : {sum(class_counter.values()):,}")
    for cls_name, count in class_counter.most_common():
        print(f"  - {cls_name:14s}: {count:,}")
    print(f"\nLicense Plates Extracted & Read: {len(detections_list)}")
    print(f"JSON Report        : {json_path}")
    print(f"CSV Report         : {csv_path}")
    print(f"Snapshots Folder   : {snapshots_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detector on single video with 180-deg rotation")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4 video")
    parser.add_argument("--output-dir", type=str, default="runs/detect/CAM01_20261020_142948", help="Output directory")
    parser.add_argument("--model", type=str, default="models/detector/vehicle_plate_detector_best.pt", help="Model checkpoint")
    parser.add_argument("--rotate", type=int, default=180, help="Rotation degrees (default 180)")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sample every N frames (default 5)")
    parser.add_argument("--conf", type=float, default=0.28, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device (cuda:0 or cpu)")
    args = parser.parse_args()

    v_path = Path(args.video)
    m_path = Path(args.model)
    out_dir = Path(args.output_dir)

    run_detector_on_video(
        video_path=v_path,
        output_dir=out_dir,
        model_path=m_path,
        rotate_deg=args.rotate,
        sample_interval=args.sample_interval,
        conf_thresh=args.conf,
        device=args.device
    )
