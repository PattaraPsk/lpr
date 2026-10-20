"""
PyTorch Dataset Loader for Thai License Plate OCR Training.
Loads cropped plate images and synthetic sequences with CTC token encoding.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .char_vocab import encode_text


class ThaiPlateDataset(Dataset):
    """
    Dataset loader for Thai plate OCR images.
    Expects either:
    1. A folder of images where the filename contains the label: e.g. "1กข8920_กรุงเทพมหานคร_001.jpg"
    2. A text/csv manifest with lines: "image_path\tlabel"
    """

    def __init__(
        self,
        image_dir: str,
        manifest_path: Optional[str] = None,
        img_h: int = 32,
        img_w: int = 128,
        transform=None
    ):
        self.image_dir = Path(image_dir)
        self.img_h = img_h
        self.img_w = img_w
        self.transform = transform
        self.samples: List[Tuple[Path, str]] = []

        if manifest_path and Path(manifest_path).exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        p = Path(parts[0])
                        if not p.is_absolute():
                            p = self.image_dir / p
                        if p.exists():
                            self.samples.append((p, parts[1]))
        elif self.image_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for p in self.image_dir.glob(ext):
                    # Default parsing from filename: e.g. "1กข_8920_กรุงเทพมหานคร.jpg"
                    label = p.stem.split("_")[0] if "_" in p.stem else p.stem
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        img_path, label = self.samples[idx]
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            # Fallback black canvas
            img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize maintaining aspect ratio with right-padding
        h, w = gray.shape
        scale = self.img_h / h
        new_w = int(w * scale)
        if new_w > self.img_w:
            resized = cv2.resize(gray, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(gray, (new_w, self.img_h), interpolation=cv2.INTER_AREA)
            pad_w = self.img_w - new_w
            resized = cv2.copyMakeBorder(resized, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # Normalize to [-1, 1]
        tensor = torch.from_numpy(resized).float().unsqueeze(0) / 127.5 - 1.0

        # Encode label tokens
        tokens = encode_text(label)
        target = torch.tensor(tokens, dtype=torch.long)

        input_len = self.img_w // 4  # Assuming 4x CNN downsampling in width
        target_len = len(tokens)

        return tensor, target, input_len, target_len
