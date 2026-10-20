"""
src/pipeline/lpr_pipeline.py
Master Production Orchestrator for Thai License Plate Recognition.
Decoupled end-to-end flow: Ingest -> Detect -> Crop/Deskew -> OCR -> Format -> Track.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.detector.base_detector import BaseDetector
from src.detector.yolo_detector import YOLODetector
from src.detector.onnx_detector import ONNXDetector
from src.ocr.recognizer import PlateRecognizer
from src.pipeline.types import BoundingBox, LPRResult, PlateReading
from src.utils.config_loader import load_config
from src.utils.metrics import EMAFpsCounter, LatencyTracker
from src.utils.visualizer import Visualizer

logger = logging.getLogger("LPRPipeline")


class BoxTracker:
    """Real-time IoU + EMA Centroid Tracker to eliminate bounding box flicker."""
    def __init__(self, max_lost: int = 15, iou_thresh: float = 0.30, ema_alpha: float = 0.75):
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh
        self.ema_alpha = ema_alpha

    def _iou(self, b1: BoundingBox, b2: BoundingBox) -> float:
        xA = max(b1.xmin, b2.xmin)
        yA = max(b1.ymin, b2.ymin)
        xB = min(b1.xmax, b2.xmax)
        yB = min(b1.ymax, b2.ymax)
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = b1.area + b2.area - inter
        return inter / max(1e-6, float(union))

    def update(self, detections: List[LPRResult]) -> List[LPRResult]:
        unmatched = list(range(len(detections)))
        matched_tracks = set()

        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for idx in unmatched:
                val = self._iou(t["bbox"], detections[idx].bbox)
                if val > best_iou:
                    best_iou = val
                    best_idx = idx

            if best_iou >= self.iou_thresh and best_idx != -1:
                det = detections[best_idx]
                old_b = t["bbox"]
                new_b = det.bbox
                a = self.ema_alpha

                # Smooth coordinates with EMA
                smoothed_bbox = BoundingBox(
                    xmin=int(a * new_b.xmin + (1.0 - a) * old_b.xmin),
                    ymin=int(a * new_b.ymin + (1.0 - a) * old_b.ymin),
                    xmax=int(a * new_b.xmax + (1.0 - a) * old_b.xmax),
                    ymax=int(a * new_b.ymax + (1.0 - a) * old_b.ymax),
                    confidence=new_b.confidence,
                    label=new_b.label,
                )
                t["bbox"] = smoothed_bbox
                t["reading"] = det.reading
                t["lost"] = 0
                det.track_id = tid
                det.bbox = smoothed_bbox
                matched_tracks.add(tid)
                unmatched.remove(best_idx)
            else:
                t["lost"] += 1
                if t["lost"] > self.max_lost:
                    del self.tracks[tid]

        for idx in unmatched:
            det = detections[idx]
            det.track_id = self.next_id
            self.tracks[self.next_id] = {
                "bbox": det.bbox,
                "reading": det.reading,
                "lost": 0,
            }
            self.next_id += 1

        return detections


class LPRPipeline:
    """
    Client-deliverable Thai LPR pipeline orchestrator.
    Decoupled and configurable for both CUDA edge workstations and CPU gateways.
    """
    def __init__(self, config_path: str = "configs/config.default.yaml"):
        self.config = load_config(config_path)
        self.latency_tracker = LatencyTracker()
        self.fps_counter = EMAFpsCounter()

        # Initialize detector backend
        det_cfg = self.config.get("detector", {})
        backend = det_cfg.get("backend", "yolo").lower()

        if backend == "onnx":
            self.detector: BaseDetector = ONNXDetector(
                model_path=det_cfg.get("model_path", "models/thai_plate_detector.onnx"),
                confidence_threshold=det_cfg.get("confidence_threshold", 0.25),
                nms_threshold=det_cfg.get("nms_threshold", 0.45),
                padding_ratio=det_cfg.get("padding_ratio", 0.06),
            )
        else:
            self.detector = YOLODetector(
                model_path=det_cfg.get("model_path", "models/yolov8s-worldv2.pt"),
                confidence_threshold=det_cfg.get("confidence_threshold", 0.25),
                nms_threshold=det_cfg.get("nms_threshold", 0.45),
                padding_ratio=det_cfg.get("padding_ratio", 0.06),
                device=self.config.get("pipeline", {}).get("target_device", "auto"),
            )

        # Initialize OCR Recognizer
        ocr_cfg = self.config.get("ocr", {})
        fmt_cfg = self.config.get("formatting", {})
        self.recognizer = PlateRecognizer(
            engine=ocr_cfg.get("engine", "segmenter"),
            vlm_model_id=ocr_cfg.get("vlm_model", "Qwen/Qwen2-VL-2B-Instruct"),
            target_char_height=ocr_cfg.get("target_char_height", 128),
            default_province=fmt_cfg.get("default_province", "กรุงเทพมหานคร"),
            filter_screws=ocr_cfg.get("filter_screws", True),
            device=self.config.get("pipeline", {}).get("target_device", "auto"),
        )

        # Initialize Tracker & Visualizer
        track_cfg = self.config.get("tracking", {})
        self.tracking_enabled = track_cfg.get("enabled", True)
        self.tracker = BoxTracker(
            max_lost=track_cfg.get("max_lost_frames", 15),
            iou_thresh=track_cfg.get("iou_threshold", 0.30),
            ema_alpha=track_cfg.get("ema_alpha", 0.75),
        )

        vis_cfg = self.config.get("visualization", {})
        self.visualizer = Visualizer(
            box_color=tuple(vis_cfg.get("box_color", [0, 230, 118])),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        render_vis: bool = False,
    ) -> Tuple[List[LPRResult], Optional[np.ndarray]]:
        """
        Executes end-to-end pipeline on a single frame:
        Detection -> Cropping -> Deskewing -> OCR Recognition -> Tracking -> Visualization.
        """
        if frame is None or frame.size == 0:
            return [], None

        t_total_start = time.perf_counter()
        current_fps = self.fps_counter.tick()

        # 1. Detection
        self.latency_tracker.start("detect")
        detections = self.detector.detect(frame)
        self.latency_tracker.stop("detect")

        results: List[LPRResult] = []

        # 2. OCR Recognition for each detected plate
        self.latency_tracker.start("ocr")
        for bbox, plate_crop in detections:
            reading = self.recognizer.recognize(plate_crop)
            result = LPRResult(
                frame_id=frame_id,
                bbox=bbox,
                reading=reading,
                plate_crop=plate_crop,
            )
            results.append(result)

        # Fallback: if input image is already a cropped plate
        if not results:
            fh, fw = frame.shape[:2]
            ar = fw / max(1.0, float(fh))
            if 1.4 <= ar <= 4.0:
                reading = self.recognizer.recognize(frame)
                if reading.consonants or reading.digits:
                    bbox = BoundingBox(xmin=0, ymin=0, xmax=fw, ymax=fh, confidence=0.75, label="plate")
                    results.append(LPRResult(frame_id=frame_id, bbox=bbox, reading=reading, plate_crop=frame))

        self.latency_tracker.stop("ocr")

        # 3. Temporal Tracking & Smoothing
        if self.tracking_enabled and results:
            results = self.tracker.update(results)

        total_latency_ms = (time.perf_counter() - t_total_start) * 1000.0
        for r in results:
            r.latency_ms = total_latency_ms

        # 4. Optional HUD visualization
        vis_frame = None
        if render_vis:
            vis_frame = self.visualizer.draw_results(
                frame=frame,
                results=results,
                fps=current_fps,
                latency_ms=total_latency_ms,
            )

        return results, vis_frame
