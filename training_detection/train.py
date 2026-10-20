#!/usr/bin/env python3
"""
License Plate Detection Model Training Script.
Supports Ultralytics YOLOv8 / YOLOv11 plate localization with custom edge hyperparameters.
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Thai License Plate Detector (YOLOv8/v11)")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Base model weights (e.g. yolov8n.pt, yolov8s.pt, yolo11n.pt)")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Path to data.yaml dataset definition")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if OOM on edge GPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (e.g. 640, 1024)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index (e.g. 0, cpu)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker threads")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs")
    parser.add_argument("--name", type=str, default="thai_plate_detector", help="Experiment name")
    parser.add_argument("--save-period", type=int, default=10, help="Checkpoint frequency")
    parser.add_argument("--export-onnx", action="store_true", help="Automatically export best.pt to ONNX upon completion")
    return parser.parse_args()


def train_detector():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' package not found. Please install via: pip install ultralytics")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = base_dir / data_path

    if not data_path.exists():
        print(f"[ERROR] Data config not found at: {data_path}")
        print("Please prepare the dataset using: python prepare_dataset.py --extract-all")
        sys.exit(1)

    print("=" * 70)
    print("      THAI LICENSE PLATE DETECTION - MODEL TRAINING")
    print("=" * 70)
    print(f"Base Model    : {args.model}")
    print(f"Dataset YAML  : {data_path}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch Size    : {args.batch}")
    print(f"Image Size    : {args.imgsz}")
    print(f"Device        : {args.device}")
    print(f"Experiment    : {args.name}")
    print("=" * 70)

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        patience=args.patience,
        name=args.name,
        save=True,
        save_period=args.save_period,
        plots=True,
        exist_ok=True,
    )

    best_weights = Path(f"runs/detect/{args.name}/weights/best.pt")
    print("\n[SUCCESS] Training completed successfully!")
    print(f"[INFO] Best model checkpoint: {best_weights}")

    if args.export_onnx and best_weights.exists():
        print("[INFO] Exporting best model to ONNX...")
        best_model = YOLO(str(best_weights))
        onnx_file = best_model.export(format="onnx", imgsz=args.imgsz, dynamic=True, opset=17)
        print(f"[SUCCESS] Exported ONNX model: {onnx_file}")


if __name__ == "__main__":
    train_detector()
