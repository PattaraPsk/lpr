#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Training Suite for Thai LPR Pipeline:
1. Plate Detector & Vehicle Classifier (YOLO Detection)
   Trains/fine-tunes YOLO (YOLOv11/YOLOv8) on dataset/dataset-vehicle-orientation (or custom data.yaml)
   Classes: Front, Rear, LicensePlate, Car, Motorcycle, Truck, Bus

2. Thai Plate Reader OCR (VisionAttentionOCR legacy baseline)
   Trains/fine-tunes VisionAttentionOCR on dataset/dataset-auto-label-a1341 (or custom plate crops)
   Alphanumeric & 77 Thai provinces attention sequence model

CLI Usage:
  python tools/train_models.py --task detector --epochs 30 --batch 16 --device cuda:0
  python tools/train_models.py --task reader --epochs 30 --batch 32 --device cuda:0
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure UTF-8 output
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.DataManipulator.app.plate_reader import PlateReader, VisionAttentionOCR, EASYOCR_DECODING_LIST, decode_tokens_to_plate
from tools.DataManipulator.app.provinces import normalize_thai_script

# Build char to index dictionary
CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(EASYOCR_DECODING_LIST) if char}


# =============================================================================
# 1. OCR Dataset Loader for Plate Reader Training
# =============================================================================

def encode_plate_to_tokens(d3: str, d4: str, province: str, max_length: int = 25) -> List[int]:
    """Encodes d3, d4, province into authentic 187-class attention tokens."""
    tokens = []
    # 1. Prefix letters (d3)
    for c in normalize_thai_script(d3):
        if c in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[c])
    tokens.append(60)  # delimiter '_'

    # 2. Digits (d4)
    for c in d4:
        if c in CHAR_TO_IDX:
            tokens.append(CHAR_TO_IDX[c])
    tokens.append(60)  # delimiter '_'

    # 3. Province name
    clean_prov = normalize_thai_script(province).strip()
    if clean_prov in CHAR_TO_IDX:
        tokens.append(CHAR_TO_IDX[clean_prov])

    tokens.append(1)  # EOS token [s]

    # Pad or truncate to max_length + 1
    target_len = max_length + 1
    if len(tokens) < target_len:
        tokens.extend([0] * (target_len - len(tokens)))
    else:
        tokens = tokens[:target_len]

    return tokens


class PlateOCRDataset(Dataset):
    """Loads cropped license plate images and ground-truth text from filenames across one or multiple directories."""
    def __init__(self, plates_dir, max_length: int = 25):
        if isinstance(plates_dir, str):
            raw_dirs = [Path(p.strip()) for p in plates_dir.split(",") if p.strip()]
        elif isinstance(plates_dir, (list, tuple)):
            raw_dirs = [Path(p) for p in plates_dir]
        else:
            raw_dirs = [Path(plates_dir)]

        self.max_length = max_length
        self.samples = []

        # Support folder containing 'plates' subfolder, type_d3_d4_province subfolders, or direct plate images
        for p_dir in raw_dirs:
            if not p_dir.exists():
                # Check relative to dataset/
                alt_dir = ROOT_DIR / p_dir
                if alt_dir.exists():
                    p_dir = alt_dir
                else:
                    continue

            search_dirs = [p_dir / "plates", p_dir]
            for s_dir in search_dirs:
                if s_dir.is_dir():
                    found = (
                        list(s_dir.glob("*.jpg")) + list(s_dir.glob("*.png")) +
                        list(s_dir.glob("*/*.jpg")) + list(s_dir.glob("*/*.png"))
                    )
                    # Filter out non-plate / background placeholder crops
                    valid_found = [
                        f for f in found
                        if "+_+_+_+" not in str(f) and "-_-_-_-" not in str(f)
                    ]
                    if valid_found:
                        self.samples.extend(valid_found)
                        break

        # Deduplicate samples by resolved path
        seen = set()
        deduped = []
        for s in self.samples:
            res = s.resolve()
            if res not in seen:
                seen.add(res)
                deduped.append(s)
        self.samples = deduped

        print(f"[OCR Dataset] Found {len(self.samples):,} valid plate image samples across {len(raw_dirs)} directories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.samples[idx]
        try:
            im = Image.open(p).convert("L")
        except Exception:
            im = Image.new("L", (128, 32), color=128)

        # Authentic NormalizePAD preprocessing: H=64, W=600
        img_h, img_w = 64, 600
        w, h = im.size
        ratio = w / float(max(h, 1))
        resized_w = min(int(np.ceil(img_h * ratio)), img_w)
        resized_crop = im.resize((resized_w, img_h), Image.BICUBIC)

        arr = np.array(resized_crop, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        t = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

        pad = torch.zeros(1, img_h, img_w, dtype=torch.float32)
        pad[:, :, :resized_w] = t
        if img_w != resized_w:
            pad[:, :, resized_w:] = t[:, :, resized_w - 1].unsqueeze(2).expand(1, img_h, img_w - resized_w)

        # Parse ground-truth from filename: {id}_{plate_type}_{d3}_{d4}_{province}.jpg
        # or from parent folder: {plate_type}_{d3}_{d4}_{province}
        parts = p.stem.split("_")
        if len(parts) >= 5 and parts[-1] and parts[-2] and parts[-3]:
            d3 = parts[-3]
            d4 = parts[-2]
            prov = parts[-1]
        else:
            folder_parts = p.parent.name.split("_")
            if len(folder_parts) >= 4:
                d3 = folder_parts[-3]
                d4 = folder_parts[-2]
                prov = folder_parts[-1]
            elif len(folder_parts) == 3:
                d3 = folder_parts[0]
                d4 = folder_parts[1]
                prov = folder_parts[2]
            else:
                d3 = parts[-3] if len(parts) >= 3 else ""
                d4 = parts[-2] if len(parts) >= 2 else ""
                prov = parts[-1] if len(parts) >= 1 else ""

        # Filter out placeholders
        if d3 in ["+", "-"]: d3 = ""
        if d4 in ["+", "-"]: d4 = ""
        if prov in ["+", "-"]: prov = ""

        token_list = encode_plate_to_tokens(d3, d4, prov, max_length=self.max_length)
        targets = torch.tensor(token_list, dtype=torch.long)

        return pad, targets


# =============================================================================
# 2. Plate Reader Training Runner
# =============================================================================

def train_plate_reader(
    data_dir: str = "dataset/ready/202610261140,dataset/dataset-cctv-synthetic/plates,dataset/dataset-mobile-synthetic/plates",
    base_weights: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.0005,
    device: str = "cuda:0",
    output_dir: str = "runs/train/reader",
    log_callback = None
):
    # Backward compatibility path resolution
    if "," not in data_dir:
        resolved_dir = Path(data_dir)
        if not resolved_dir.exists():
            legacy_dir = Path("dataset") / resolved_dir.name
            if legacy_dir.exists():
                resolved_dir = legacy_dir
        data_dir = str(resolved_dir)

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log("=" * 65)
    log("     THAI LICENSE PLATE OCR (legacy baseline) TRAINING PIPELINE")
    log("=" * 65)

    dev = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    use_amp = (dev.type == "cuda")
    log(f"Compute Device : {dev} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    log(f"Mixed Precision: {'AMP (FP16/BF16)' if use_amp else 'FP32'}")
    log(f"Dataset Path   : {data_dir}")
    log(f"Epochs         : {epochs}")
    log(f"Batch Size     : {batch_size}")
    log(f"Learning Rate  : {lr}")

    dataset = PlateOCRDataset(data_dir)
    if len(dataset) == 0:
        log(f"[ERROR] No plate images found in {data_dir}!")
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    # Initialize Model
    model = VisionAttentionOCR(num_classes=187, batch_max_length=25)

    # Resolve base weights
    weights_path = Path(base_weights) if base_weights else (ROOT_DIR / "models" / "reader" / "legacy baseline" / "best_norm_ED.pth")
    if not weights_path.exists():
        weights_path = ROOT_DIR / "best_norm_ED.pth"

    if weights_path.exists():
        log(f"Loading base checkpoint: {weights_path}")
        ckpt = torch.load(str(weights_path), map_location="cpu")
        if isinstance(ckpt, dict):
            clean_sd = {k.replace("module.", ""): v for k, v in ckpt.items()}
            model.load_state_dict(clean_sd, strict=False)
            log("Base checkpoint weights loaded successfully.")
    else:
        log("No base weights found; initializing model from scratch.")

    model.to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path = (ROOT_DIR / output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_model_path = out_path / "best_reader.pth"

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs:02d}", leave=False, mininterval=2.0, ncols=100)

        for images, targets in pbar:
            images = images.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(images)  # [B, 26, 187]
                loss = F.cross_entropy(preds.view(-1, 187), targets.view(-1), ignore_index=0)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        elapsed = time.time() - start_time
        log(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {elapsed:.1f}s")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), str(best_model_path))
            log(f"  [SAVED] New best model saved to {best_model_path.name} (Loss: {best_loss:.4f})")

    # Save final model
    last_model_path = out_path / "last_reader.pth"
    torch.save(model.state_dict(), str(last_model_path))
    log(f"\n[DONE] OCR training completed! Best model saved to: {best_model_path}")
    return str(best_model_path)


# =============================================================================
# 3. Plate Detector & Vehicle Classifier Training Runner (YOLO)
# =============================================================================

def train_yolo_detector(
    data_yaml: str = "dataset/labeled/dataset-vehicle-orientation/data.yaml",
    base_model: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cuda:0",
    project_dir: str = "runs/train",
    run_name: str = "vehicle_plate_detector",
    log_callback = None
):
    # Backward compatibility path resolution
    resolved_yaml = Path(data_yaml)
    if not resolved_yaml.exists():
        legacy_yaml = Path("dataset") / resolved_yaml.parent.name / resolved_yaml.name
        if legacy_yaml.exists():
            resolved_yaml = legacy_yaml
    data_yaml = str(resolved_yaml)
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log("=" * 65)
    log("  PLATE DETECTOR & VEHICLE CLASSIFIER (YOLO) TRAINING")
    log("=" * 65)

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        log(f"[ERROR] Dataset configuration file not found: {yaml_path}")
        return None

    # Resolve base model
    model_path = base_model
    if not model_path:
        for cand in [
            ROOT_DIR / "models" / "detector" / "vehicle_plate_detector_best.pt",
            ROOT_DIR / "models" / "detector" / "vehicle_plate_yolov11n.pt",
            ROOT_DIR / "models" / "detector" / "vehicle_plate_detector.pt",
            ROOT_DIR / "models" / "detector" / "yolov8s.pt",
            ROOT_DIR / "vehicle_plate_detector.pt",
            ROOT_DIR / "yolov8s.pt"
        ]:
            if cand.exists():
                model_path = str(cand)
                break

    if not model_path:
        model_path = "yolo11n.pt"

    abs_project_dir = str((ROOT_DIR / project_dir).resolve())
    dev = device if torch.cuda.is_available() and "cuda" in device else "cpu"
    log(f"Compute Device : {dev}")
    log(f"Base Model     : {model_path}")
    log(f"Dataset YAML   : {yaml_path}")
    log(f"Epochs         : {epochs}")
    log(f"Batch Size     : {batch_size}")
    log(f"Image Size     : {imgsz}")
    log(f"Project / Run  : {abs_project_dir}/{run_name}")

    from ultralytics import YOLO

    try:
        model = YOLO(model_path)
        log("Model architecture loaded. Commencing training run...")
        
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=dev,
            project=abs_project_dir,
            name=run_name,
            plots=True,
            save=True,
            exist_ok=True
        )
        
        best_pt = Path(abs_project_dir) / run_name / "weights" / "best.pt"
        log(f"\n[DONE] YOLO training finished successfully! Checkpoint: {best_pt}")
        return str(best_pt)
    except Exception as e:
        log(f"[ERROR] YOLO training error: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# CLI Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Model Training Hub for Thai LPR")
    parser.add_argument("--task", type=str, choices=["detector", "reader", "both"], default="detector",
                        help="Task: 'detector' (YOLO Plate & Vehicle), 'reader' (OCR legacy baseline), or 'both'")
    parser.add_argument("--data", type=str, default="", help="Path to data.yaml or plate dataset folder")
    parser.add_argument("--weights", type=str, default="", help="Path to base model weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (for OCR)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device ('cuda:0' or 'cpu')")
    parser.add_argument("--output", type=str, default="", help="Output directory")

    args = parser.parse_args()

    if args.task in ["detector", "both"]:
        default_yaml = "dataset/labeled/dataset-vehicle-orientation/data.yaml" if Path("dataset/labeled/dataset-vehicle-orientation/data.yaml").exists() else "dataset/dataset-vehicle-orientation/data.yaml"
        data_yaml = args.data if args.data else default_yaml
        out_dir = args.output if args.output else "runs/train"
        train_yolo_detector(
            data_yaml=data_yaml,
            base_model=args.weights or None,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project_dir=out_dir
        )

    if args.task in ["reader", "both"]:
        default_dir = "dataset/ready/202610261140,dataset/dataset-cctv-synthetic/plates,dataset/dataset-mobile-synthetic/plates"
        if not Path("dataset/ready/202610261140").exists():
            default_dir = "dataset/labeled/dataset-auto-label-a1341" if Path("dataset/labeled/dataset-auto-label-a1341").exists() else "dataset/dataset-auto-label-a1341"
        data_dir = args.data if args.data else default_dir
        out_dir = args.output if args.output else "runs/train/reader"
        train_plate_reader(
            data_dir=data_dir,
            base_weights=args.weights or None,
            epochs=args.epochs,
            batch_size=args.batch if args.batch != 16 else 64,
            lr=args.lr,
            device=args.device,
            output_dir=out_dir
        )


if __name__ == "__main__":
    main()
