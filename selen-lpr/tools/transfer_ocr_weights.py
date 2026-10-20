#!/usr/bin/env python3
"""
Weight Transfer & Model Migration Tool (SEPARATED FROM DEPLOYMENT CODE).
Extracts weights from previous company / freelance models, strips DataParallel wrappers,
validates parameter tensor shapes, and converts into clean standalone PyTorch checkpoints.
"""

import sys
import argparse
from pathlib import Path
import torch

# Add selen-lpr package root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.vision_attention_ocr import VisionAttentionOCR


def transfer_ocr_weights(
    source_checkpoint: str,
    output_checkpoint: str,
    device: str = "cpu"
):
    src_path = Path(source_checkpoint)
    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"[ERROR] Source checkpoint not found at: {src_path}")
        return False

    print("=" * 70)
    print("      SELEN-LPR: WEIGHT TRANSFER & MIGRATION TOOL")
    print("=" * 70)
    print(f"Source Checkpoint : {src_path}")
    print(f"Target Output     : {out_path}")
    print(f"Device            : {device}")
    print("=" * 70)

    # 1. Load source checkpoint
    print("[INFO] Loading source state_dict...")
    raw_state = torch.load(str(src_path), map_location=device)

    # Strip 'module.' prefix from nn.DataParallel
    clean_state = {}
    for k, v in raw_state.items():
        clean_key = k.replace("module.", "")
        clean_state[clean_key] = v

    print(f"[SUCCESS] Cleaned {len(clean_state)} parameter tensors.")

    # 2. Instantiate clean VisionAttentionOCR model
    print("[INFO] Instantiating clean standalone PyTorch VisionAttentionOCR...")
    model = VisionAttentionOCR(num_classes=187, input_channel=1, output_channel=512, hidden_size=512)

    # 3. Load state_dict into clean model
    model_keys = set(model.state_dict().keys())
    state_keys = set(clean_state.keys())

    missing = model_keys - state_keys
    unexpected = state_keys - model_keys

    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {list(missing)[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}")

    model.load_state_dict(clean_state, strict=True)
    print("[SUCCESS] All parameter weights matched with 100% strict compatibility!")

    # 4. Save clean deployment checkpoint
    torch.save(model.state_dict(), str(out_path))
    file_size_mb = round(out_path.stat().st_size / (1024 * 1024), 2)
    print(f"[SUCCESS] Saved standalone deployment checkpoint: {out_path} ({file_size_mb} MB)")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description="Transfer weights from external model to selen-lpr")
    parser.add_argument(
        "--source",
        type=str,
        default=r"D:\Projects-freelance\EasyOCR\saved_models\full_combine_1v21_token_wAugment\best_accuracy.pth",
        help="Path to source .pth checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT_DIR / "models" / "ocr" / "thai_plate_attn_ocr.pt"),
        help="Target output .pt path"
    )
    args = parser.parse_args()

    transfer_ocr_weights(args.source, args.output)


if __name__ == "__main__":
    main()
