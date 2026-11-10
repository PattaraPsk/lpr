"""
Inference & Prediction Pipeline.
Applies Thai DLT Post-Processing & Validation Rules.
Complies with ISO/IEC 42001:2023 Operational Event Logging (Annex A.9.3).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import yaml

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.models.model import CRNNEdgeRecognizer, build_model
from src.utils.metrics import sanitize_thai_plate_text
from src.utils.visualization import draw_plate_prediction


class PlatePredictor:
    """
    High-performance, edge-optimized inference predictor for License Plates.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model_config_path: Optional[str | Path] = "configs/model.yaml",
        device: Optional[str] = None,
        img_height: int = 64,
        img_width: int = 192,
    ):
        self.img_height = img_height
        self.img_width = img_width

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load config
        m_cfg = {}
        if model_config_path and Path(model_config_path).exists():
            with open(model_config_path, "r", encoding="utf-8") as f:
                m_cfg = yaml.safe_load(f)

        self.model = build_model(m_cfg)

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)
            self._log_event(f"Loaded checkpoint from: {checkpoint_path}")
        else:
            self._log_event("[WARN] Checkpoint not found; running with initialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def _log_event(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        print(f"{ts} [INFERENCE] {msg}")

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Resizes, normalizes, and converts image to tensor."""
        resized = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (norm - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[str, float]:
        """
        Runs inference on an image path or numpy image array.
        Returns: (predicted_plate_text, confidence_score)
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            img = image_input

        tensor = self.preprocess(img)
        log_probs = self.model(tensor)  # [T, 1, Vocab]

        # Calculate approximate confidence from mean maximum probability across sequence
        probs = log_probs.exp()
        max_probs, _ = probs.max(dim=-1)
        confidence = float(max_probs.mean().item())

        decoded_list = self.model.decode_greedy(log_probs)
        raw_text = decoded_list[0] if decoded_list else ""
        cleaned_text = sanitize_thai_plate_text(raw_text)

        return cleaned_text, confidence

    def predict_batch(self, image_paths: List[str | Path]) -> List[Tuple[str, float]]:
        """Runs prediction across a list of image paths."""
        results = []
        for p in image_paths:
            try:
                res = self.predict(p)
                results.append(res)
            except Exception as e:
                self._log_event(f"Error predicting {p}: {e}")
                results.append(("", 0.0))
        return results


def main():
    parser = argparse.ArgumentParser(description="Predict license plate text on image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/model.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    predictor = PlatePredictor(
        checkpoint_path=args.checkpoint,
        model_config_path=args.config,
    )

    text, conf = predictor.predict(args.image)
    print(f"\n[PREDICTION RESULT]")
    print(f"Image      : {args.image}")
    print(f"Plate Text : {text}")
    print(f"Confidence : {conf:.4f}")

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(args.image)
        if img is not None:
            vis = draw_plate_prediction(img, plate_text=text, confidence=conf)
            out_file = out_dir / f"pred_{Path(args.image).name}"
            cv2.imwrite(str(out_file), vis)
            print(f"Saved visualization to: {out_file}")


if __name__ == "__main__":
    main()
