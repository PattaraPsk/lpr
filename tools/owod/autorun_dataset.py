#!/usr/bin/env python3
"""
Auto-Run OWOD Labeling on all media in the dataset folder.
Executes zero-shot license plate labeling using YOLO-World with GPU acceleration (CUDA).
"""

import sys
import os
import time
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows console
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add paths
current_dir = Path(__file__).resolve().parent
tools_dir = current_dir.parent
project_root = tools_dir.parent
for p in [str(project_root), str(tools_dir), str(current_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np

from core.media_scanner import MediaScanner
from core.video_processor import VideoProcessor
from core.owod_engine import OWODDetector, load_image_unicode
from core.dataset_generator import DatasetGenerator

def autorun(
    dataset_dir: str = "dataset",
    output_dir: str = "dataset/yolo_owod_labeled_dataset",
    model_name: str = "yolov8s-worldv2.pt",
    prompt: str = "license plate",
    conf_thresh: float = 0.25,
    device: str = "cuda:0",
    train_ratio: float = 0.8,
    video_sample_step: int = 20,
    video_max_frames: int = 100
):
    print("=" * 70)
    print("⚡ OWOD LICENSE PLATE AUTO-RUN BATCH PROCESSOR")
    print("=" * 70)
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU Active: {dev_name} (CUDA {torch.version.cuda})")
            actual_device = device
        else:
            print("⚠️ CUDA not available, falling back to CPU")
            actual_device = "cpu"
    except Exception as e:
        print(f"⚠️ PyTorch check error: {e}, using CPU")
        actual_device = "cpu"
        
    ds_path = Path(dataset_dir).resolve()
    out_path = Path(output_dir).resolve()
    
    print(f"📂 Scanning dataset folder: {ds_path}")
    scanner = MediaScanner()
    scanner.add_directory(ds_path)
    
    stats = scanner.get_stats()
    print(f"📊 Discovered: {stats['total_files']} files "
          f"({stats['total_images']} images, {stats['total_videos']} videos) "
          f"across {stats['subfolder_count']} subfolders")
          
    if stats['total_files'] == 0:
        print(f"❌ No media files found in {ds_path}!")
        return
        
    print(f"🤖 Loading OWOD Model: {model_name}...")
    detector = OWODDetector(
        model_name_or_path=model_name,
        prompt=prompt,
        conf_threshold=conf_thresh,
        device=actual_device
    )
    
    print(f"📁 Initializing Output Dataset: {out_path}")
    generator = DatasetGenerator(
        output_dir=out_path,
        train_ratio=train_ratio,
        save_crops=True,
        save_visualizations=True
    )
    
    print("\n" + "-" * 70)
    print("🚀 Starting Auto-Labeling Pipeline...")
    print("-" * 70)
    
    total_items = len(scanner.items)
    processed_count = 0
    total_plates_found = 0
    start_time = time.time()
    
    try:
        for idx, item in enumerate(scanner.items):
            current_idx = idx + 1
            pct = (current_idx / total_items) * 100
            
            if item.media_type == "image":
                img_rgb = load_image_unicode(item.path)
                if img_rgb is None:
                    continue
                    
                detections = detector.predict(img_rgb, filename_hint=item.filename)
                
                # Check if it's a negative background sample (+_+_+_+ or unreadable)
                is_negative = "+_+_+_+" in item.filename or "-_-_-_-" in item.filename
                
                if detections or is_negative:
                    unique_name = item.unique_identifier
                    generator.save_sample(
                        base_name=unique_name,
                        image_rgb=img_rgb,
                        detections=detections,
                        allow_empty=is_negative
                    )
                    
                total_plates_found += len(detections)
                processed_count += 1
                
                if current_idx % 25 == 0 or current_idx == total_items:
                    elapsed = time.time() - start_time
                    fps = processed_count / elapsed if elapsed > 0 else 0
                    print(
                        f"\r[{current_idx:5d}/{total_items:5d}] {pct:5.1f}% | "
                        f"Plates: {total_plates_found:5d} | "
                        f"Speed: {fps:4.1f} img/s | "
                        f"Current: {item.filename[:35]:35s}",
                        end="", flush=True
                    )
                    
            elif item.media_type == "video":
                v_name = item.filename
                v_plates = 0
                for f_idx, ts, frame_rgb in VideoProcessor.extract_frames(
                    item.path, sample_step=video_sample_step, max_frames=video_max_frames
                ):
                    detections = detector.predict(frame_rgb, filename_hint=v_name)
                    if detections:
                        unique_vid_name = f"{item.unique_identifier}_f{f_idx:06d}"
                        generator.save_sample(
                            base_name=unique_vid_name,
                            image_rgb=frame_rgb,
                            detections=detections
                        )
                        v_plates += len(detections)
                        total_plates_found += len(detections)
                    processed_count += 1
                    
                print(f"\n🎬 Finished Video: {v_name} (Plates found: {v_plates})")
                
    except KeyboardInterrupt:
        print("\n\n⚠️ Execution paused by user (Ctrl+C). Finalizing dataset...")
        
    # Generate data.yaml
    yaml_path = generator.generate_data_yaml()
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎉 OWOD AUTO-LABELING COMPLETED!")
    print("=" * 70)
    print(f"• Total Processed Items: {processed_count:,}")
    print(f"• Total Plates Labeled:  {total_plates_found:,}")
    print(f"• Time Taken:            {total_time:.1f} seconds ({(processed_count / total_time) if total_time > 0 else 0:.1f} fps)")
    print(f"• YOLO Dataset YAML:     {yaml_path}")
    print(f"• Train Images:          {generator.img_train_dir}")
    print(f"• Val Images:            {generator.img_val_dir}")
    print(f"• Plate Crops:           {generator.crops_dir}")
    print(f"• Debug Visualizations:  {generator.vis_dir}")
    print("=" * 70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-Run OWOD Labeling on Dataset")
    parser.add_argument("--dataset", default="dataset", help="Path to input dataset folder")
    parser.add_argument("--output", default="dataset/yolo_owod_labeled_dataset", help="Path to output YOLO dataset")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="OWOD Model checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="cuda:0", help="Inference device (cuda:0 or cpu)")
    parser.add_argument("--split", type=float, default=0.80, help="Train/Val split ratio")
    args = parser.parse_args()
    
    autorun(
        dataset_dir=args.dataset,
        output_dir=args.output,
        model_name=args.model,
        conf_thresh=args.conf,
        device=args.device,
        train_ratio=args.split
    )
