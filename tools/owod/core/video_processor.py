"""
Video frame extraction and sampling for license plate detection.
"""

from pathlib import Path
from typing import Generator, Tuple, Optional
import cv2
import numpy as np

class VideoProcessor:
    """Extracts frames from video files according to sampling strategies."""
    
    @staticmethod
    def get_video_info(video_path: str | Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"valid": False, "frames": 0, "fps": 0, "duration_sec": 0, "width": 0, "height": 0}
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            "valid": True,
            "frames": frame_count,
            "fps": fps,
            "duration_sec": round(duration, 2),
            "width": width,
            "height": height
        }

    @staticmethod
    def extract_frames(
        video_path: str | Path,
        sample_step: int = 15,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Yields (frame_idx, timestamp_sec, frame_rgb_image).
        sample_step: extract 1 frame every `sample_step` frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        extracted_count = 0
        step = max(1, sample_step)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / fps
                yield frame_idx, timestamp, frame_rgb
                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break
                    
            frame_idx += 1
            
        cap.release()
