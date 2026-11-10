"""
PyTorch Dataset and DataLoader implementation for License Plate OCR.
Complies with Thai DLT 41-consonant specifications.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.metrics import sanitize_thai_plate_text

# Thai DLT 41 Consonants (excluding obsolete ฃ, ฅ, and optical collision ฏ)
DLT_THAI_CONSONANTS = "กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"
DIGITS = "0123456789"
BLANK_TOKEN = "<blank>"

# Vocabulary mapping
VOCAB_LIST = [BLANK_TOKEN] + list(DIGITS) + list(DLT_THAI_CONSONANTS)
CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(VOCAB_LIST)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(VOCAB_LIST)}
VOCAB_SIZE = len(VOCAB_LIST)  # 52


def encode_label(text: str) -> List[int]:
    """Encodes a string into a list of integer vocabulary indices."""
    cleaned = sanitize_thai_plate_text(text)
    indices: List[int] = []
    for ch in cleaned:
        if ch in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[ch])
    return indices


def decode_indices(indices: Sequence[int], remove_blank: bool = True) -> str:
    """Decodes integer indices back to text string, optionally removing blank token."""
    chars: List[str] = []
    for idx in indices:
        if idx == 0 and remove_blank:
            continue
        if idx in IDX_TO_CHAR:
            chars.append(IDX_TO_CHAR[idx])
    return "".join(chars)


class LicensePlateDataset(Dataset):
    """
    License Plate Dataset for OCR training and evaluation.
    """

    def __init__(
        self,
        split_file: Optional[str | Path] = None,
        samples: Optional[List[Tuple[str, str]]] = None,
        img_height: int = 64,
        img_width: int = 192,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []

        if split_file:
            path = Path(split_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            self.samples.append((parts[0], parts[1]))
                        elif len(parts) == 1:
                            self.samples.append((parts[0], ""))

        if samples:
            self.samples.extend(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, img_path: str) -> np.ndarray:
        """Loads and resizes image, providing black image fallback if read fails."""
        img = cv2.imread(img_path)
        if img is None:
            # Fallback zero-filled image
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        img_path, raw_label = self.samples[idx]
        image = self._load_image(img_path)

        if self.transform is not None:
            image = self.transform(image)

        # Convert HWC (BGR) to CHW float32 [0.0, 1.0]
        image = image.astype(np.float32) / 255.0
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        encoded = encode_label(raw_label)
        target_tensor = torch.tensor(encoded, dtype=torch.long)
        target_len = len(encoded)

        return image_tensor, target_tensor, target_len, raw_label


def ctc_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Collate function for PyTorch CTCLoss.
    Returns:
    - images: [B, C, H, W]
    - targets: 1D concatenated tensor of all target indices [sum(target_lengths)]
    - input_lengths: 1D tensor [B] (width steps after backbone pooling)
    - target_lengths: 1D tensor [B]
    - raw_labels: List[str]
    """
    images, targets, target_lengths, labels = zip(*batch)

    stacked_images = torch.stack(images, dim=0)
    batch_size = len(images)

    # In CRNN, width downsampling ratio is typically 4 or 8.
    # For width 192 downsampled by 4 => 48 time steps.
    feature_width = stacked_images.size(3) // 4
    input_lengths = torch.full((batch_size,), feature_width, dtype=torch.long)

    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    concat_targets = torch.cat([t for t in targets if len(t) > 0], dim=0) if sum(target_lengths) > 0 else torch.empty(0, dtype=torch.long)

    return stacked_images, concat_targets, input_lengths, target_lengths_tensor, list(labels)


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Creates a configured PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
