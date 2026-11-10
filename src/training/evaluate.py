"""
Model Evaluation Pipeline.
Compliant with ISO/IEC 42001:2023 Verification Standards.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.data.dataset import LicensePlateDataset, get_dataloader
from src.models.losses import CTCLossWrapper
from src.models.model import CRNNEdgeRecognizer, build_model
from src.utils.metrics import (
    compute_accuracy,
    compute_character_error_rate,
    compute_normalized_edit_distance,
)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates model on dataloader.
    Returns dictionary with: loss, accuracy, normalized_edit_distance, character_error_rate.
    """
    model.eval()
    if criterion is None:
        criterion = CTCLossWrapper()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            images, targets, input_lengths, target_lengths, raw_labels = batch
            images = images.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            log_probs = model(images)

            if len(targets) > 0 and target_lengths.sum().item() > 0:
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    batches += 1

            # Decode predictions
            if hasattr(model, "decode_greedy"):
                preds = model.decode_greedy(log_probs)
            else:
                preds = [""] * len(raw_labels)

            all_preds.extend(preds)
            all_targets.extend(raw_labels)

    avg_loss = total_loss / max(1, batches)
    acc = compute_accuracy(all_preds, all_targets)
    ned = compute_normalized_edit_distance(all_preds, all_targets)
    cer = compute_character_error_rate(all_preds, all_targets)

    return {
        "val_loss": avg_loss,
        "accuracy": acc,
        "normalized_edit_distance": ned,
        "character_error_rate": cer,
        "total_evaluated": len(all_preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Computer Vision License Plate OCR Model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--config", type=str, default="configs/model.yaml", help="Path to model.yaml")
    parser.add_argument("--split", type=str, default="data/splits/val.txt", help="Path to split.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Load model config
    model_cfg = {}
    if Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)

    model = build_model(model_cfg)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=args.device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    else:
        print(f"[WARN] Checkpoint {ckpt_path} not found. Running with initialized weights.")

    model.to(args.device)

    dataset = LicensePlateDataset(split_file=args.split)
    dataloader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_model(model, dataloader, device=args.device)
    print("\n[EVALUATION RESULTS]")
    print(f"Total Samples : {metrics['total_evaluated']}")
    print(f"Validation Loss: {metrics['val_loss']:.4f}")
    print(f"Exact Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Mean NED       : {metrics['normalized_edit_distance']:.4f}")
    print(f"CER            : {metrics['character_error_rate']:.4f}")


if __name__ == "__main__":
    main()
