"""
Visualization utilities for Computer Vision and License Plate Recognition.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np


def draw_plate_prediction(
    image: np.ndarray,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    plate_text: str = "",
    confidence: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Overlays bounding box and predicted license plate text onto an image.
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        label_pos = (x1, max(20, y1 - 8))
    else:
        label_pos = (20, 30)

    label_parts = []
    if plate_text:
        label_parts.append(plate_text)
    if confidence is not None:
        label_parts.append(f"{confidence:.2f}")

    if label_parts:
        label_str = " | ".join(label_parts)
        cv2.putText(
            vis,
            label_str,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness + 2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label_str,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return vis


def plot_loss_curves(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plots training and validation loss curves and optionally saves them to disk.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)

    if val_losses is not None and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, "r--", label="Validation Loss", linewidth=2)

    plt.title("Model Training Loss Dynamics")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_p), dpi=150)
    plt.close()
