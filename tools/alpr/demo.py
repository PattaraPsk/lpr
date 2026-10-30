#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Thai ALPR/ANPR Edge Demo Application.
Compliant with ISO 8601, ISO/IEC 25012, ISO/IEC 27001, and ISO 3166-2:TH.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

from tools.alpr.config import (
    ROOT_DIR,
    DETECTOR_CANDIDATES,
    VEHICLE_CANDIDATES,
    READER_CANDIDATES,
    DEFAULT_CONF_THRESH,
    DEFAULT_VEH_CONF_THRESH,
    DEFAULT_DISPLAY_WIDTH,
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_SKIP_OCR_FRAMES,
    resolve_model_path,
)
from tools.alpr.stream import ThreadedStreamCapture
from tools.alpr.tracker import TrackedPlate, box_iou, box_center_dist
from tools.alpr.detector import (
    detect_vehicles,
    detect_plates_coarse_to_fine,
    detect_plates_direct,
    match_vehicle_and_orientation,
)
from tools.alpr.hud import draw_hud, render_thai_badge, overlay_badge
from tools.DataManipulator.app.plate_reader import PlateReader
from tools.rtsp_collector.security import load_env_file, mask_rtsp_url


class RealtimeLPRDemo:
    """
    Main ALPR demo application managing video streams, neural detection cascades,
    tracking, OCR prediction, HUD telemetry, and user interactions.
    """

    def __init__(
        self,
        source: str | int,
        detector_path: Optional[str] = None,
        reader_path: Optional[str] = None,
        veh_model_path: Optional[str] = None,
        conf_thresh: float = DEFAULT_CONF_THRESH,
        rotate_180: bool = False,
        display_width: int = DEFAULT_DISPLAY_WIDTH,
        display_height: int = DEFAULT_DISPLAY_HEIGHT,
        device: Optional[str] = None,
        skip_ocr_frames: int = DEFAULT_SKIP_OCR_FRAMES,
        save_snapshots_dir: Optional[str] = "runs/demo_snapshots",
        record_path: Optional[str] = None,
        headless: bool = False,
        max_frames: Optional[int] = None,
        auto_advance: bool = False,
        roi_box: Optional[List[int]] = None,
        show_vehicles: bool = True,
        veh_conf_thresh: float = DEFAULT_VEH_CONF_THRESH,
        coarse_to_fine: bool = True
    ):
        self.source_raw = source
        self.conf_thresh = conf_thresh
        self.veh_conf_thresh = veh_conf_thresh
        self.rotate_180 = rotate_180
        self.display_width = display_width
        self.display_height = display_height
        self.skip_ocr_frames = skip_ocr_frames
        self.record_path = record_path
        self.headless = headless
        self.max_frames = max_frames
        self.auto_advance = auto_advance
        self.show_vehicles = show_vehicles
        self.coarse_to_fine = coarse_to_fine

        # Region of Interest (ROI) State
        self.roi_box: Optional[List[int]] = roi_box
        self.roi_enabled: bool = (roi_box is not None)
        self.is_drawing_roi: bool = False
        self.roi_drag_start: Optional[Tuple[int, int]] = None
        self.roi_drag_current: Optional[Tuple[int, int]] = None
        self.last_orig_shape: Optional[Tuple[int, int]] = None

        # Build video playlist
        self.playlist: List[Path] = self._build_playlist(source)
        self.current_video_idx = 0
        if self.playlist and isinstance(source, (str, Path)):
            src_p = Path(source).resolve()
            for idx, p in enumerate(self.playlist):
                if p.resolve() == src_p:
                    self.current_video_idx = idx
                    break
            actual_source = str(self.playlist[self.current_video_idx])
            self.source_raw = actual_source
        elif self.playlist:
            actual_source = str(self.playlist[0])
            self.source_raw = actual_source
        else:
            actual_source = source

        # Visual toast message state
        self.toast_msg: Optional[str] = None
        self.toast_expire: float = 0.0

        self.snapshots_dir = Path(save_snapshots_dir) if save_snapshots_dir else None
        if self.snapshots_dir:
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            (self.snapshots_dir / "plates").mkdir(parents=True, exist_ok=True)
            (self.snapshots_dir / "scenes").mkdir(parents=True, exist_ok=True)

        # Select Compute Device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print("\n" + "=" * 65)
        print(" THAI LICENSE PLATE RECOGNITION (ALPR/ANPR) - REAL-TIME DEMO")
        print("=" * 65)
        print(f" Source          : {mask_rtsp_url(str(self.source_raw))}")
        if self.playlist:
            print(f" Playlist        : {len(self.playlist)} video(s) found (Active: #{self.current_video_idx + 1})")
        if self.roi_box:
            print(f" Active ROI      : {self.roi_box}")
        print(f" Compute Device  : {self.device}")
        print(f" Strategy        : {'Coarse-to-Fine (Vehicle Cascade)' if self.coarse_to_fine else 'Direct (Full Frame)'}")
        print(f" Confidence      : {self.conf_thresh}")
        print(f" 180° Flip       : {'ON' if self.rotate_180 else 'OFF'} (Press 'R' to toggle)")
        print(f" Display Scale   : {self.display_width}x{self.display_height}")
        print(" Hotkeys         : [N/P] Next/Prev VDO  |  [Drag Mouse] Set ROI  |  [X] Clear ROI")
        print("=" * 65 + "\n")

        # Load Models
        self._load_models(detector_path, reader_path, veh_model_path)

        # Initialize Video Capture Stream
        end_cb = self.switch_to_next_video if self.auto_advance else None
        self.stream = ThreadedStreamCapture(
            source=actual_source,
            loop_video=not self.auto_advance,
            on_video_end_callback=end_cb
        ).start()

        # Tracking state
        self.active_tracks: List[TrackedPlate] = []
        self.next_track_id = 1
        self.recent_recognized_log: collections.deque = collections.deque(maxlen=6)

        # Telemetry metrics
        self.fps_meter = collections.deque(maxlen=30)
        self.det_latencies = collections.deque(maxlen=30)
        self.ocr_latencies = collections.deque(maxlen=30)
        self.total_plates_captured = 0

        # UI toggle states
        self.is_paused = False
        self.show_hud = True
        self.show_boxes = True
        self.video_writer: Optional[cv2.VideoWriter] = None

    def _build_playlist(self, source: Any) -> List[Path]:
        """Discovers playlist of video files for sequential playback and hotkey switching."""
        if isinstance(source, int):
            return []
        s_str = str(source)
        if s_str.startswith("rtsp://") or s_str.startswith("http://"):
            return []

        src_p = Path(s_str)
        if src_p.is_dir():
            return sorted([
                p for p in src_p.rglob("*.mp4")
                if not p.name.startswith(".") and p.stat().st_size > 1024 * 1024
            ])
        elif src_p.is_file():
            return sorted([
                p for p in src_p.parent.glob("*.mp4")
                if not p.name.startswith(".") and p.stat().st_size > 1024 * 1024
            ])

        all_samples = []
        search_dirs = [
            ROOT_DIR / "dataset" / "raw" / "19-Oct-2026",
            ROOT_DIR / "dataset" / "raw" / "Raw Video Record",
            ROOT_DIR / "dataset" / "19-Oct-2026",
            ROOT_DIR / "dataset" / "Raw Video Record"
        ]
        for d in search_dirs:
            if d.exists():
                vids = sorted([
                    p for p in d.rglob("*.mp4")
                    if not p.name.startswith(".") and p.stat().st_size > 1024 * 1024
                ])
                all_samples.extend(vids)
        return all_samples

    def switch_to_next_video(self) -> bool:
        """Switches to the next video file in the playlist."""
        if not self.playlist or len(self.playlist) <= 1:
            print("[INFO] No other video in playlist to switch to.")
            self.toast_msg = "Single video (no other videos in playlist)"
            self.toast_expire = time.time() + 2.5
            return False

        self.current_video_idx = (self.current_video_idx + 1) % len(self.playlist)
        next_vid = self.playlist[self.current_video_idx]
        return self._switch_video(next_vid)

    def switch_to_prev_video(self) -> bool:
        """Switches to the previous video file in the playlist."""
        if not self.playlist or len(self.playlist) <= 1:
            print("[INFO] No other video in playlist to switch to.")
            self.toast_msg = "Single video (no other videos in playlist)"
            self.toast_expire = time.time() + 2.5
            return False

        self.current_video_idx = (self.current_video_idx - 1) % len(self.playlist)
        prev_vid = self.playlist[self.current_video_idx]
        return self._switch_video(prev_vid)

    def _switch_video(self, video_path: Path) -> bool:
        """Switches active video capture stream and clears active tracks."""
        print(f"\n[SWITCH] Changing to video [{self.current_video_idx + 1}/{len(self.playlist)}]: {video_path.name}")
        self.source_raw = str(video_path)
        ok = self.stream.switch_source(str(video_path))
        if ok:
            self.active_tracks.clear()
            self.toast_msg = f"Video [{self.current_video_idx + 1}/{len(self.playlist)}]: {video_path.name}"
            self.toast_expire = time.time() + 3.0
            print(f"[SWITCH] Successfully loaded: {video_path.name}")
            return True
        else:
            self.toast_msg = f"Failed to load: {video_path.name}"
            self.toast_expire = time.time() + 3.0
            print(f"[ERROR] Failed to switch to: {video_path.name}")
            return False

    def _mouse_callback(self, event, x, y, flags, param):
        """Interactive mouse callback for drawing ROI on the display window."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing_roi = True
            self.roi_drag_start = (x, y)
            self.roi_drag_current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing_roi:
            self.roi_drag_current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing_roi:
            self.is_drawing_roi = False
            self.roi_drag_current = (x, y)
            if self.roi_drag_start and self.last_orig_shape is not None:
                x1 = min(self.roi_drag_start[0], x)
                y1 = min(self.roi_drag_start[1], y)
                x2 = max(self.roi_drag_start[0], x)
                y2 = max(self.roi_drag_start[1], y)

                if (x2 - x1) >= 25 and (y2 - y1) >= 25:
                    orig_h, orig_w = self.last_orig_shape
                    scale_x = orig_w / float(self.display_width)
                    scale_y = orig_h / float(self.display_height)

                    rx1 = max(0, min(orig_w - 1, int(x1 * scale_x)))
                    ry1 = max(0, min(orig_h - 1, int(y1 * scale_y)))
                    rx2 = max(rx1 + 20, min(orig_w, int(x2 * scale_x)))
                    ry2 = max(ry1 + 20, min(orig_h, int(y2 * scale_y)))

                    self.roi_box = [rx1, ry1, rx2, ry2]
                    self.roi_enabled = True
                    roi_w, roi_h = rx2 - rx1, ry2 - ry1
                    self.toast_msg = f"ROI Set: {roi_w}x{roi_h}px [Press X to Clear]"
                    self.toast_expire = time.time() + 3.0
                    print(f"\n[ROI] Set active Region of Interest: {self.roi_box} ({roi_w}x{roi_h}px)")
                else:
                    print("[ROI] Drag selection too small (<25px), ignored.")
            self.roi_drag_start = None
            self.roi_drag_current = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.roi_box is not None:
                self.roi_box = None
                self.toast_msg = "ROI Cleared (Full Frame Active)"
                self.toast_expire = time.time() + 2.5
                print("\n[ROI] Cleared Region of Interest. Processing full frame.")

    def _resolve_model_path(self, user_path: Optional[str], default_candidates: List[Path]) -> Path:
        return resolve_model_path(user_path, default_candidates)

    def _load_models(self, det_path_arg, reader_path_arg, veh_path_arg):
        print(">>> Loading Neural Inference Engines...")

        # 1. License Plate & Orientation Detector (YOLOv11n)
        det_file = self._resolve_model_path(det_path_arg, DETECTOR_CANDIDATES)
        print(f"  [DETECTOR]   Loading: {det_file.name}")
        self.detector = YOLO(str(det_file)).to(self.device)

        # 2. Vehicle Classifier (YOLOv8s)
        veh_file = self._resolve_model_path(veh_path_arg, VEHICLE_CANDIDATES)
        if veh_file.exists():
            print(f"  [VEHICLE]    Loading: {veh_file.name}")
            self.veh_detector = YOLO(str(veh_file)).to(self.device)
        else:
            print("  [VEHICLE]    YOLOv8s weights not found, using plate detector standalone.")
            self.veh_detector = None

        # 3. Thai Plate Reader (VisionAttentionOCR thai_plate_reader)
        ocr_file = self._resolve_model_path(reader_path_arg, READER_CANDIDATES)
        print(f"  [OCR READER] Loading: {ocr_file.name}")
        self.reader = PlateReader.get_instance(str(ocr_file))
        if self.reader.error_msg:
            print(f"  [ERROR] Reader failed to load: {self.reader.error_msg}")
        else:
            print("  [OCR READER] VisionAttentionOCR ready.")

        print("[SUCCESS] All Neural Models Loaded Successfully.\n")

    def _match_vehicle_and_orientation(
        self,
        plate_box: List[float],
        orientations: List[Tuple[List[float], str, float]],
        vehicles: List[Tuple[List[float], str, float]]
    ) -> Tuple[str, str]:
        return match_vehicle_and_orientation(plate_box, orientations, vehicles)

    def _find_matching_track(self, box: List[float]) -> Optional[TrackedPlate]:
        best_trk = None
        best_iou = 0.25
        for trk in self.active_tracks:
            iou = box_iou(box, trk.current_box)
            dist = box_center_dist(box, trk.current_box)
            if iou > best_iou or (iou > 0.15 and dist < 120):
                best_trk = trk
                best_iou = iou
        return best_trk

    def _update_tracks(self, current_detections: List[Dict[str, Any]], frame_idx: int):
        matched_track_ids = set()

        for det in current_detections:
            box = det["box"]
            crop_bgr = det["crop"]
            ocr_res = det["ocr"]
            veh_type = det["veh_type"]
            orient = det["orientation"]
            conf = det["conf"]

            best_track = None
            best_iou = 0.25

            for trk in self.active_tracks:
                if trk.track_id in matched_track_ids:
                    continue
                iou = box_iou(box, trk.current_box)
                dist = box_center_dist(box, trk.current_box)
                if iou > best_iou or (iou > 0.15 and dist < 120):
                    best_track = trk
                    best_iou = iou

            if best_track is not None:
                best_track.update(
                    box=box,
                    conf=conf,
                    frame_idx=frame_idx,
                    crop_bgr=crop_bgr,
                    ocr_result=ocr_res,
                    vehicle_type=veh_type,
                    orientation=orient
                )
                matched_track_ids.add(best_track.track_id)
                det["track_id"] = best_track.track_id
                det["consensus_plate"] = best_track.consensus_plate
                det["iso_province"] = best_track.iso_province
            else:
                new_trk = TrackedPlate(
                    track_id=self.next_track_id,
                    box=box,
                    conf=conf,
                    frame_idx=frame_idx,
                    crop_bgr=crop_bgr,
                    ocr_result=ocr_res,
                    vehicle_type=veh_type,
                    orientation=orient
                )
                self.next_track_id += 1
                self.active_tracks.append(new_trk)
                matched_track_ids.add(new_trk.track_id)
                self.total_plates_captured += 1

                det["track_id"] = new_trk.track_id
                det["consensus_plate"] = new_trk.consensus_plate
                det["iso_province"] = new_trk.iso_province

                if new_trk.consensus_plate:
                    self.recent_recognized_log.append({
                        "id": new_trk.track_id,
                        "plate": new_trk.consensus_plate,
                        "iso": new_trk.iso_province,
                        "veh": new_trk.vehicle_type,
                        "orient": new_trk.orientation,
                        "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
                    })

        self.active_tracks = [
            t for t in self.active_tracks
            if (frame_idx - t.last_frame) <= 35
        ]

    def _draw_hud(
        self,
        display_frame: np.ndarray,
        orig_w: int,
        orig_h: int,
        fps: float,
        det_ms: float,
        ocr_ms: float,
        tracked_dets: List[Dict[str, Any]],
        vehicles: Optional[List[Tuple[List[float], str, float]]] = None
    ) -> np.ndarray:
        return draw_hud(
            display_frame=display_frame,
            orig_w=orig_w,
            orig_h=orig_h,
            fps=fps,
            det_ms=det_ms,
            ocr_ms=ocr_ms,
            tracked_dets=tracked_dets,
            vehicles=vehicles,
            roi_box=self.roi_box,
            roi_enabled=self.roi_enabled,
            is_drawing_roi=self.is_drawing_roi,
            roi_drag_start=self.roi_drag_start,
            roi_drag_current=self.roi_drag_current,
            show_vehicles=self.show_vehicles,
            show_boxes=self.show_boxes,
            show_hud=self.show_hud,
            is_paused=self.is_paused,
            rotate_180=self.rotate_180,
            coarse_to_fine=self.coarse_to_fine,
            device=self.device,
            source_raw=str(self.source_raw),
            playlist=self.playlist,
            current_video_idx=self.current_video_idx,
            toast_msg=self.toast_msg,
            toast_expire=self.toast_expire,
            active_tracks_count=len(self.active_tracks),
            recent_recognized_log=list(self.recent_recognized_log)
        )

    def _save_manual_snapshot(self, frame: np.ndarray, tracked_dets: List[Dict[str, Any]]):
        if not self.snapshots_dir:
            return

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:19]
        scene_path = self.snapshots_dir / "scenes" / f"snapshot_{ts_str}_scene.jpg"
        cv2.imwrite(str(scene_path), frame)
        print(f"\n[SNAPSHOT] Saved scene: {scene_path.name}")

        for det in tracked_dets:
            crop = det.get("crop")
            if crop is not None and crop.size > 0:
                plate_txt = det.get("consensus_plate") or "unknown"
                clean_plate = "".join([c if c.isalnum() else "_" for c in plate_txt])
                plate_path = self.snapshots_dir / "plates" / f"snapshot_{ts_str}_trk{det.get('track_id', 0)}_{clean_plate}.jpg"
                cv2.imwrite(str(plate_path), crop)
                print(f"  -> Saved plate crop: {plate_path.name}")

    def run(self):
        window_name = "Thai LPR Edge ANPR - Real-Time Demo"
        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.display_width, self.display_height)
            cv2.setMouseCallback(window_name, self._mouse_callback)

        prev_time = time.time()
        frame_count = 0

        try:
            while True:
                if self.max_frames and frame_count >= self.max_frames:
                    print(f"\n[INFO] Reached max frames ({self.max_frames}). Stopping demo.")
                    break

                if not self.is_paused:
                    ret, frame, frame_idx = self.stream.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            break
                        elif key in (ord('n'), ord('N')):
                            self.switch_to_next_video()
                        elif key in (ord('p'), ord('P'), ord('b'), ord('B')):
                            self.switch_to_prev_video()
                        elif key in (ord('x'), ord('X')):
                            self.roi_box = None
                            self.toast_msg = "ROI Cleared (Full Frame Active)"
                            self.toast_expire = time.time() + 2.5
                        continue
                else:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        self.is_paused = False
                        print("[INFO] Resumed stream.")
                    elif key == ord('q') or key == 27:
                        break
                    elif key == ord('r'):
                        self.rotate_180 = not self.rotate_180
                        print(f"[INFO] 180° Flip toggled: {'ON' if self.rotate_180 else 'OFF'}")
                    elif key in (ord('n'), ord('N')):
                        self.switch_to_next_video()
                    elif key in (ord('p'), ord('P'), ord('b'), ord('B')):
                        self.switch_to_prev_video()
                    elif key in (ord('x'), ord('X')):
                        self.roi_box = None
                        self.toast_msg = "ROI Cleared (Full Frame Active)"
                        self.toast_expire = time.time() + 2.5
                    elif key in (ord('z'), ord('Z')):
                        self.roi_enabled = not self.roi_enabled
                        self.toast_msg = f"ROI Filter {'ON' if self.roi_enabled else 'OFF'}"
                        self.toast_expire = time.time() + 2.5
                    elif key in (ord('v'), ord('V')):
                        self.show_vehicles = not self.show_vehicles
                        status = "ON" if self.show_vehicles else "OFF"
                        self.toast_msg = f"Vehicle Boxes: {status}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] Vehicle boxes {status}.")
                    elif key in (ord('f'), ord('F')):
                        self.coarse_to_fine = not self.coarse_to_fine
                        strat = "COARSE-TO-FINE (Vehicle Cascade)" if self.coarse_to_fine else "DIRECT (Full Frame)"
                        self.toast_msg = f"Strategy: {strat}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] Detection strategy switched to: {strat}")
                    continue

                frame_count += 1
                orig_h, orig_w = frame.shape[:2]
                self.last_orig_shape = (orig_h, orig_w)

                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                active_roi = self.roi_box if (self.roi_enabled and self.roi_box is not None) else None
                if active_roi is not None:
                    rx1, ry1, rx2, ry2 = active_roi
                    rx1 = max(0, min(rx1, orig_w - 20))
                    ry1 = max(0, min(ry1, orig_h - 20))
                    rx2 = max(rx1 + 20, min(rx2, orig_w))
                    ry2 = max(ry1 + 20, min(ry2, orig_h))
                    det_frame = frame[ry1:ry2, rx1:rx2]
                    offset_x, offset_y = rx1, ry1
                else:
                    det_frame = frame
                    offset_x, offset_y = 0, 0

                # Coarse Stage: Vehicle Detection
                vehicles: List[Tuple[List[float], str, float]] = []
                if self.veh_detector is not None:
                    vehicles = detect_vehicles(
                        veh_detector=self.veh_detector,
                        det_frame=det_frame,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        conf_thresh=self.veh_conf_thresh,
                        device=self.device
                    )

                # Fine Stage: Plate Detection
                t_det_0 = time.time()
                if self.coarse_to_fine and self.veh_detector is not None and vehicles:
                    plates, orientations = detect_plates_coarse_to_fine(
                        detector=self.detector,
                        frame=frame,
                        vehicles=vehicles,
                        orig_w=orig_w,
                        orig_h=orig_h,
                        conf_thresh=self.conf_thresh,
                        device=self.device
                    )
                else:
                    plates, orientations = detect_plates_direct(
                        detector=self.detector,
                        det_frame=det_frame,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        conf_thresh=self.conf_thresh,
                        device=self.device
                    )

                det_ms = (time.time() - t_det_0) * 1000.0
                self.det_latencies.append(det_ms)

                # OCR Prediction
                current_detections = []
                t_ocr_0 = time.time()

                for p_box, p_conf in plates:
                    px1, py1, px2, py2 = p_box
                    pw = px2 - px1
                    ph = py2 - py1

                    if pw < 14 or ph < 8 or (pw / orig_w) > 0.45:
                        continue

                    pad_x = int(pw * 0.08)
                    pad_y = int(ph * 0.08)
                    cx1 = max(0, int(px1) - pad_x)
                    cy1 = max(0, int(py1) - pad_y)
                    cx2 = min(orig_w, int(px2) + pad_x)
                    cy2 = min(orig_h, int(py2) + pad_y)

                    crop_bgr = frame[cy1:cy2, cx1:cx2]
                    if crop_bgr.size == 0:
                        continue

                    matched_veh, matched_orient = self._match_vehicle_and_orientation(
                        p_box, orientations, vehicles
                    )

                    matched_trk = self._find_matching_track(p_box)
                    needs_ocr = True
                    if matched_trk is not None:
                        if matched_trk.best_conf >= 0.88 and matched_trk.hits >= 2:
                            if (frame_idx % self.skip_ocr_frames) != 0:
                                needs_ocr = False
                                ocr_res = matched_trk.best_ocr

                    if needs_ocr:
                        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        ocr_res = self.reader.predict(pil_crop)

                    current_detections.append({
                        "box": p_box,
                        "conf": p_conf,
                        "crop": crop_bgr,
                        "ocr": ocr_res,
                        "veh_type": matched_veh,
                        "orientation": matched_orient
                    })

                ocr_ms = (time.time() - t_ocr_0) * 1000.0 if current_detections else 0.0
                self.ocr_latencies.append(ocr_ms)

                self._update_tracks(current_detections, frame_idx)

                t_now = time.time()
                cur_fps = 1.0 / max(0.001, t_now - prev_time)
                prev_time = t_now
                self.fps_meter.append(cur_fps)
                avg_fps = sum(self.fps_meter) / len(self.fps_meter)
                avg_det_ms = sum(self.det_latencies) / len(self.det_latencies)
                avg_ocr_ms = sum(self.ocr_latencies) / len(self.ocr_latencies) if self.ocr_latencies else 0.0

                display_frame = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_LINEAR)
                annotated_frame = self._draw_hud(
                    display_frame=display_frame,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    fps=avg_fps,
                    det_ms=avg_det_ms,
                    ocr_ms=avg_ocr_ms,
                    tracked_dets=current_detections,
                    vehicles=vehicles
                )

                if self.record_path:
                    if self.video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        self.video_writer = cv2.VideoWriter(
                            str(self.record_path), fourcc, 25.0, (self.display_width, self.display_height)
                        )
                    self.video_writer.write(annotated_frame)

                if not self.headless:
                    cv2.imshow(window_name, annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n[INFO] Exit key pressed. Closing demo...")
                        break
                    elif key == ord(' '):
                        self.is_paused = not self.is_paused
                        print(f"[INFO] Stream {'PAUSED' if self.is_paused else 'RESUMED'}.")
                    elif key in (ord('n'), ord('N')):
                        self.switch_to_next_video()
                    elif key in (ord('p'), ord('P'), ord('b'), ord('B')):
                        self.switch_to_prev_video()
                    elif key in (ord('l'), ord('L')):
                        self.stream.loop_video = not self.stream.loop_video
                        loop_status = "ON (Loop Current)" if self.stream.loop_video else "OFF (Auto-Advance)"
                        self.toast_msg = f"Loop Mode: {loop_status}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] Loop Mode toggled: {loop_status}")
                    elif key == ord('r'):
                        self.rotate_180 = not self.rotate_180
                        print(f"[INFO] 180° Flip toggled: {'ON' if self.rotate_180 else 'OFF'}")
                    elif key in (ord('x'), ord('X')):
                        self.roi_box = None
                        self.toast_msg = "ROI Cleared (Full Frame Active)"
                        self.toast_expire = time.time() + 2.5
                        print("[INFO] ROI cleared by user. Processing full frame.")
                    elif key in (ord('z'), ord('Z')):
                        self.roi_enabled = not self.roi_enabled
                        status = "ENABLED" if self.roi_enabled else "DISABLED (Full Frame)"
                        self.toast_msg = f"ROI Filter {status}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] ROI filter {status}.")
                    elif key == ord('s'):
                        self._save_manual_snapshot(frame, current_detections)
                    elif key in (ord('v'), ord('V')):
                        self.show_vehicles = not self.show_vehicles
                        status = "ON" if self.show_vehicles else "OFF"
                        self.toast_msg = f"Vehicle Boxes: {status}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] Vehicle boxes {status}.")
                    elif key in (ord('f'), ord('F')):
                        self.coarse_to_fine = not self.coarse_to_fine
                        strat = "COARSE-TO-FINE (Vehicle Cascade)" if self.coarse_to_fine else "DIRECT (Full Frame)"
                        self.toast_msg = f"Strategy: {strat}"
                        self.toast_expire = time.time() + 2.5
                        print(f"[INFO] Detection strategy switched to: {strat}")
                    elif key == ord('t'):
                        self.show_boxes = not self.show_boxes
                    elif key == ord('h'):
                        self.show_hud = not self.show_hud
                    elif key == ord('c'):
                        self.recent_recognized_log.clear()
                else:
                    if frame_count % 10 == 0:
                        print(f"  [HEADLESS] Frame {frame_count:04d} | FPS: {avg_fps:.1f} | Dets: {len(current_detections)} | Tracks: {len(self.active_tracks)}")

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received.")
        finally:
            self.stream.stop()
            if self.video_writer:
                self.video_writer.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            print("\n=======================================================")
            print(f" Demo Finished. Total Unique Plates Captured: {self.total_plates_captured}")
            if self.snapshots_dir:
                print(f" Snapshots saved to: {self.snapshots_dir.resolve()}")
            print("=======================================================\n")


def find_default_sample_video() -> Optional[Path]:
    """Finds a valid sample video from dataset folders if no source is given."""
    search_dirs = [
        ROOT_DIR / "dataset" / "raw" / "19-Oct-2026",
        ROOT_DIR / "dataset" / "raw" / "Raw Video Record",
        ROOT_DIR / "dataset" / "19-Oct-2026",
        ROOT_DIR / "dataset" / "Raw Video Record"
    ]
    for d in search_dirs:
        if d.exists():
            vids = sorted(list(d.rglob("*.mp4")))
            valid_vids = [v for v in vids if not v.name.startswith(".") and v.stat().st_size > 1024 * 1024]
            if valid_vids:
                return valid_vids[0]
    return None


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        description="Thai License Plate Recognition (ALPR/ANPR) - Real-Time Demo"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: camera index (e.g. 0), video file path (.mp4), or RTSP URL."
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="Use RTSP_URL from .env as the input source."
    )
    parser.add_argument(
        "--webcam",
        type=int,
        default=None,
        help="Use local webcam by index (e.g. 0, 1)."
    )
    parser.add_argument(
        "--detector",
        type=str,
        default=None,
        help="Path to YOLO plate detector checkpoint (.pt)."
    )
    parser.add_argument(
        "--reader",
        type=str,
        default=None,
        help="Path to Thai OCR reader model checkpoint (.pth)."
    )
    parser.add_argument(
        "--vehicle-model",
        type=str,
        default=None,
        help="Path to YOLO vehicle classifier model (.pt)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF_THRESH,
        help="Detection confidence threshold (default: 0.25)."
    )
    parser.add_argument(
        "--veh-conf",
        type=float,
        default=DEFAULT_VEH_CONF_THRESH,
        help="Vehicle detection confidence threshold (default: 0.35)."
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=DEFAULT_SKIP_OCR_FRAMES,
        help="Re-verification frame interval for already confident tracked plates (default: 5)."
    )
    parser.add_argument(
        "--hide-vehicles",
        action="store_true",
        help="Hide vehicle bounding boxes by default (toggleable with 'V')."
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run direct full-frame plate detection instead of coarse-to-fine vehicle cascade."
    )
    parser.add_argument(
        "--rotate-180",
        action="store_true",
        help="Enable 180° rotation flip (default: off, toggleable with 'R')."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_DISPLAY_WIDTH,
        help="Display window width (default: 1280)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_DISPLAY_HEIGHT,
        help="Display window height (default: 720)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: 'cuda:0' or 'cpu' (default: auto)."
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Optional path to save annotated output video (.mp4)."
    )
    parser.add_argument(
        "--snapshots-dir",
        type=str,
        default="runs/demo_snapshots",
        help="Directory to save plate and scene snapshots (default: runs/demo_snapshots)."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless benchmark mode without opening an OpenCV GUI window."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop automatically after processing N frames (useful for benchmarking)."
    )
    parser.add_argument(
        "--auto-advance",
        action="store_true",
        help="Automatically advance to the next video clip when current video reaches the end."
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Initial Region of Interest bounding box as 'x1,y1,x2,y2' in frame pixels."
    )

    args = parser.parse_args()

    chosen_source = args.source
    if args.webcam is not None:
        chosen_source = args.webcam
    elif args.rtsp:
        env_rtsp = os.environ.get("RTSP_URL")
        if not env_rtsp:
            print("[ERROR] --rtsp requested but RTSP_URL is not defined in .env.")
            sys.exit(1)
        chosen_source = env_rtsp

    if chosen_source is None:
        default_vid = find_default_sample_video()
        if default_vid:
            chosen_source = str(default_vid)
            print(f"[INFO] No source specified. Automatically selected sample video: {default_vid.name}")
        else:
            chosen_source = 0
            print("[INFO] No video sample found. Defaulting to local Webcam (Index 0).")

    if isinstance(chosen_source, str) and chosen_source.isdigit():
        chosen_source = int(chosen_source)

    roi_box = None
    if args.roi:
        try:
            parts = [int(float(p.strip())) for p in args.roi.split(",")]
            if len(parts) == 4:
                roi_box = parts
        except Exception as e:
            print(f"[WARN] Failed to parse --roi '{args.roi}': {e}")

    demo = RealtimeLPRDemo(
        source=chosen_source,
        detector_path=args.detector,
        reader_path=args.reader,
        veh_model_path=args.vehicle_model,
        conf_thresh=args.conf,
        rotate_180=args.rotate_180,
        display_width=args.width,
        display_height=args.height,
        device=args.device,
        save_snapshots_dir=args.snapshots_dir,
        record_path=args.record,
        headless=args.headless,
        max_frames=args.max_frames,
        auto_advance=args.auto_advance,
        roi_box=roi_box,
        show_vehicles=not args.hide_vehicles,
        veh_conf_thresh=args.veh_conf,
        coarse_to_fine=not args.direct,
        skip_ocr_frames=args.ocr_interval
    )

    demo.run()


if __name__ == "__main__":
    main()
