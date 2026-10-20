#!/usr/bin/env python3
"""
Selen-LPR Standalone Pure PyTorch Service & CLI Interface.
Zero EasyOCR, Zero Ultralytics dependencies.
Processes input images, extracts normalized coordinates (s_x, s_y, s_w, s_h),
classifies vehicle type, runs Vision-Attention OCR, and outputs structured JSON.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add package root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Ensure stdout supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.pipeline.adaptive_pipeline import SelenLPRPipeline


class SelenLPRService:
    """
    Main service implementing the Selen-LPR API Contract.
    """

    def __init__(self, ocr_weights: str = "", detector_weights: str = ""):
        if not ocr_weights:
            ocr_weights = str(BASE_DIR / "models" / "ocr" / "thai_plate_attn_ocr.pt")
        if not detector_weights:
            detector_weights = str(BASE_DIR / "models" / "detector" / "plate_detector.torchscript")

        self.pipeline = SelenLPRPipeline(
            ocr_weights_path=ocr_weights,
            detector_weights_path=detector_weights
        )

    def process_image(self, image_path: str) -> dict:
        img_path = Path(image_path)
        if not img_path.exists():
            return {"error": f"Image file not found: {image_path}"}

        try:
            frame = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            frame = None
        if frame is None:
            return {"error": f"Failed to decode image: {image_path}"}

        results = self.pipeline.process_frame(frame)
        detections = [r.to_dict() for r in results]

        return {
            "source": str(image_path),
            "detections_count": len(detections),
            "detections": detections
        }


def main():
    parser = argparse.ArgumentParser(description="Selen-LPR Pure PyTorch Inference Service")
    parser.add_argument("--image", type=str, default="samples/01_private_car/sample.png", help="Path to input image")
    parser.add_argument("--ocr-weights", type=str, default="", help="Path to OCR .pt weights")
    parser.add_argument("--detector-weights", type=str, default="", help="Path to detector weights")
    parser.add_argument("--output", type=str, default="", help="Path to save output JSON")
    args = parser.parse_args()

    service = SelenLPRService(
        ocr_weights=args.ocr_weights,
        detector_weights=args.detector_weights
    )
    result = service.process_image(args.image)

    formatted_json = json.dumps(result, ensure_ascii=False, indent=2)
    print(formatted_json)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        print(f"[INFO] Saved output JSON to: {args.output}")


if __name__ == "__main__":
    main()
