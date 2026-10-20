#!/usr/bin/env python3
"""
Export Trained CRNN OCR Model to ONNX for Edge Inference.
Converts PyTorch checkpoint (.pt) to ONNX Runtime format.
"""

import sys
import shutil
import argparse
from pathlib import Path
import torch

from .char_vocab import get_vocab_size
from .train_ocr import CRNN


def main():
    parser = argparse.ArgumentParser(description="Export CRNN OCR to ONNX")
    parser.add_argument("--weights", type=str, default="thai_plate_crnn.pt", help="Path to .pt weights")
    parser.add_argument("--img-h", type=int, default=32, help="Input height")
    parser.add_argument("--img-w", type=int, default=128, help="Input width")
    parser.add_argument("--output", type=str, default="thai_plate_ocr.onnx", help="Output .onnx filename")
    parser.add_argument("--deploy-to-pipeline", action="store_true", default=True, help="Copy to pipeline/models/")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = base_dir / weights_path

    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)

    model = CRNN(num_classes=get_vocab_size())
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, args.img_h, args.img_w)
    out_path = base_dir / args.output

    print(f"[INFO] Exporting CRNN to ONNX: {out_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size", 3: "width"},
            "logits": {0: "time_step", 1: "batch_size"}
        }
    )

    print(f"[SUCCESS] Exported ONNX model: {out_path} ({round(out_path.stat().st_size / (1024*1024), 2)} MB)")

    if args.deploy_to_pipeline:
        dest_dir = base_dir.parent.parent / "pipeline" / "models"
        if dest_dir.exists():
            dest_file = dest_dir / out_path.name
            shutil.copy2(out_path, dest_file)
            print(f"[SUCCESS] Deployed to inference pipeline: {dest_file}")


if __name__ == "__main__":
    main()
