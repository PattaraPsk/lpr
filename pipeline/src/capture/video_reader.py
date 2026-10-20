"""
src/capture/video_reader.py
Sequential file and directory image reader for offline processing and benchmarks.
"""
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np

VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class VideoFileReader:
    """Iterates through frames of a local video file with metadata."""
    def __init__(self, video_path: str):
        self.path = Path(video_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.path))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self):
        if self.cap:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class ImageDirectoryReader:
    """Iterates through an image file or directory of images."""
    def __init__(self, input_path: Union[str, Path]):
        self.path = Path(input_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")

        if self.path.is_file():
            self.image_files = [self.path]
        else:
            self.image_files = sorted([
                p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
            ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for img_path in self.image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                yield str(img_path), img
