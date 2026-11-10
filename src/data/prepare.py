"""
Data Preparation & Split Generation Module.
Complies with ISO 25012 Data Quality & ISO/IEC 42001:2023 Traceability.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple
import cv2

from src.utils.metrics import sanitize_thai_plate_text


def scan_dataset_directory(
    root_dir: str | Path,
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> List[Tuple[str, str]]:
    """
    Scans a directory for images.
    Supports either:
    1. Folder-name as label: root_dir/<label>/<image>.jpg
    2. File-name as label: root_dir/<label>.jpg or root_dir/<label>_idx.jpg
    """
    root_path = Path(root_dir)
    samples: List[Tuple[str, str]] = []

    if not root_path.exists():
        return samples

    for entry in root_path.rglob("*"):
        if entry.is_file() and entry.suffix.lower() in valid_extensions:
            # If immediate parent is not root_path, use parent folder name as label
            if entry.parent != root_path:
                raw_label = entry.parent.name
            else:
                raw_label = entry.stem.split("_")[0]

            label = sanitize_thai_plate_text(raw_label)
            samples.append((str(entry.resolve()), label))

    return samples


def create_splits(
    samples: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Splits samples deterministically into train, validation, and test sets.
    """
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_set = shuffled[:n_train]
    val_set = shuffled[n_train : n_train + n_val]
    test_set = shuffled[n_train + n_val :]

    return train_set, val_set, test_set


def write_split_file(output_path: str | Path, samples: List[Tuple[str, str]]) -> None:
    """
    Writes samples to a split text file: <image_path>\t<label>
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Computer Vision Dataset Split\n")
        f.write(f"# Total samples: {len(samples)}\n")
        for img_path, label in samples:
            f.write(f"{img_path}\t{label}\n")


def prepare_dataset(
    source_dir: str = "data/raw",
    splits_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Executes end-to-end dataset discovery, validation, and split export.
    """
    samples = scan_dataset_directory(source_dir)
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    if not samples:
        # Create placeholder splits if no samples found in source_dir
        train_p = splits_path / "train.txt"
        val_p = splits_path / "val.txt"
        test_p = splits_path / "test.txt"
        write_split_file(train_p, [])
        write_split_file(val_p, [])
        write_split_file(test_p, [])
        return 0, 0, 0

    train_set, val_set, test_set = create_splits(
        samples, train_ratio, val_ratio, test_ratio, seed
    )

    write_split_file(splits_path / "train.txt", train_set)
    write_split_file(splits_path / "val.txt", val_set)
    write_split_file(splits_path / "test.txt", test_set)

    return len(train_set), len(val_set), len(test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and generate train/val/test splits.")
    parser.add_argument("--source-dir", type=str, default="data/raw", help="Path to raw image directory")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Path to output splits directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    n_tr, n_va, n_te = prepare_dataset(source_dir=args.source_dir, splits_dir=args.splits_dir, seed=args.seed)
    print(f"[SUCCESS] Dataset prepared: {n_tr} train, {n_va} val, {n_te} test")
