"""
src/utils/visualizer.py
High-performance visual HUD, bounding box drawing, and telemetry rendering.
"""
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.pipeline.types import LPRResult


class Visualizer:
    def __init__(
        self,
        box_color: Tuple[int, int, int] = (0, 230, 118),  # Emerald Green in BGR
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_path: Optional[str] = None,
    ):
        self.box_color = box_color
        self.text_color = text_color
        self.font_path = font_path

    def draw_results(
        self,
        frame: np.ndarray,
        results: List[LPRResult],
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """Renders bounding boxes, plate text headers, and telemetry HUD on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw HUD header bar
        hud_text = f"THAI EDGE LPR | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Plates: {len(results)}"
        cv2.rectangle(vis, (0, 0), (w, 36), (15, 15, 15), -1)
        cv2.putText(
            vis, hud_text, (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 118), 2, cv2.LINE_AA
        )

        for res in results:
            b = res.bbox
            if b.width == 0 or b.height == 0:
                continue

            x1, y1, x2, y2 = b.xmin, b.ymin, b.xmax, b.ymax
            reading = res.reading
            label_text = reading.formatted_string or "THAI_PLATE"
            if res.track_id is not None:
                label_text = f"#{res.track_id}: {label_text}"

            # Draw bounding box
            color = self.box_color if reading.is_valid_dlt else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw header tag
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tag_y1 = max(36, y1 - th - 8)
            cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 8, tag_y1 + th + 6), color, -1)
            cv2.putText(
                vis, label_text, (x1 + 4, tag_y1 + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

        return vis
