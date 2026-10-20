#!/usr/bin/env python3
"""
Detector Model Export & Transfer Tool (SEPARATED FROM DEPLOYMENT CODE).
Converts YOLO weights to standalone TorchScript (.torchscript) and ONNX (.onnx) formats.
Allows selen-lpr to run high-speed plate detection with ZERO ultralytics library dependency.
"""

import sys
import argparse
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent


def export_detector(
    weights_path: str,
    output_dir: str = "",
    imgsz: int = 640
):
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        return False

    out_dir = Path(output_dir) if output_dir else ROOT_DIR / "models" / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      SELEN-LPR: DETECTOR EXPORT & WEIGHT TRANSFER TOOL")
    print("=" * 70)
    print(f"Source Weights    : {weights}")
    print(f"Target Directory  : {out_dir}")
    print("=" * 70)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics package required during conversion phase only.")
        return False

    print(f"[INFO] Loading YOLO model from: {weights}")
    model = YOLO(str(weights))

    # Export to TorchScript (Pure PyTorch runtime)
    print("[INFO] Exporting to TorchScript (Pure PyTorch)...")
    ts_path = model.export(format="torchscript", imgsz=imgsz)
    ts_file = Path(ts_path)
    target_ts = out_dir / "plate_detector.torchscript"
    import shutil
    shutil.copy2(ts_file, target_ts)
    print(f"[SUCCESS] Exported TorchScript: {target_ts} ({round(target_ts.stat().st_size / (1024*1024), 2)} MB)")

    # Export to ONNX (Lightweight edge runtime)
    print("[INFO] Exporting to ONNX...")
    onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=True, opset=17)
    onnx_file = Path(onnx_path)
    target_onnx = out_dir / "plate_detector.onnx"
    shutil.copy2(onnx_file, target_onnx)
    print(f"[SUCCESS] Exported ONNX: {target_onnx} ({round(target_onnx.stat().st_size / (1024*1024), 2)} MB)")

    print("=" * 70)
    print("[SUCCESS] Detector exported successfully! selen-lpr can now run with ZERO ultralytics dependency.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export detector to pure PyTorch TorchScript/ONNX")
    parser.add_argument(
        "--weights",
        type=str,
        default=r"d:\Projects-anti\lpr\pipeline\models\yolov8s-worldv2.pt",
        help="Path to YOLO weights file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "models" / "detector"),
        help="Target output directory"
    )
    args = parser.parse_args()

    export_detector(args.weights, args.output_dir)


if __name__ == "__main__":
    main()
