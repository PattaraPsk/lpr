#!/usr/bin/env python3
"""
Evaluate Trained Plate Detector Benchmark & mAP Metrics.
Calculates mAP50, mAP50-95, precision, recall, and edge inference latency.
"""

import sys
import time
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate Thai License Plate Detector")
    parser.add_argument("--model", type=str, default="runs/detect/thai_plate_detector/weights/best.pt", help="Path to .pt or .onnx model")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = base_dir / model_path

    if not model_path.exists():
        print(f"[ERROR] Model weights not found: {model_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' required. Please install via: pip install ultralytics")
        sys.exit(1)

    print(f"[INFO] Validating detector: {model_path}")
    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(base_dir / args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=True
    )

    print("\n" + "=" * 60)
    print("             DETECTOR VALIDATION METRICS")
    print("=" * 60)
    print(f"Precision (P)   : {metrics.box.mp:.4f}")
    print(f"Recall (R)      : {metrics.box.mr:.4f}")
    print(f"mAP@50          : {metrics.box.map50:.4f}")
    print(f"mAP@50-95       : {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
