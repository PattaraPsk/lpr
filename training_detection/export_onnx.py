#!/usr/bin/env python3
"""
Export Trained Plate Detector to Optimized ONNX Runtime.
Converts PyTorch (.pt) weights to ONNX format with dynamic batching, opset 17,
and optionally copies the result to ../pipeline/models/.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO Plate Detector to ONNX")
    parser.add_argument("--weights", type=str, default="runs/detect/thai_plate_detector/weights/best.pt", help="Path to .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input dimension")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Enable dynamic shape batching")
    parser.add_argument("--half", action="store_true", help="Export in FP16 half precision")
    parser.add_argument("--deploy-to-pipeline", action="store_true", default=True, help="Copy exported .onnx to pipeline/models/")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path

    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' required. Please install via: pip install ultralytics")
        sys.exit(1)

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset}, dynamic={args.dynamic}, half={args.half})...")
    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half
    )

    onnx_file = Path(exported_path)
    print(f"[SUCCESS] Exported ONNX model: {onnx_file} ({round(onnx_file.stat().st_size / (1024*1024), 2)} MB)")

    if args.deploy_to_pipeline:
        dest_dir = base_dir.parent / "pipeline" / "models"
        if dest_dir.exists():
            dest_file = dest_dir / onnx_file.name
            shutil.copy2(onnx_file, dest_file)
            print(f"[SUCCESS] Deployed to inference pipeline: {dest_file}")
        else:
            print(f"[WARN] Pipeline models directory not found at {dest_dir}")


if __name__ == "__main__":
    main()
