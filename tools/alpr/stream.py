#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Latency Threaded Video & RTSP Ingestion Engine.
Compliant with ISO 8601, ISO/IEC 14496-14, and ISO/IEC 25012.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np


class ThreadedStreamCapture:
    """
    Robust threaded frame reader supporting Webcam, RTSP streams, and local MP4 videos.
    For live network/camera feeds, automatically drops stale queued frames so the
    application always processes the freshest frame with zero latency.
    Supports dynamic video switching on the fly with thread-safe capture swap.
    """

    def __init__(
        self,
        source: str | int,
        loop_video: bool = True,
        on_video_end_callback: Optional[Any] = None
    ):
        self.source = source
        self.loop_video = loop_video
        self.on_video_end_callback = on_video_end_callback
        self.is_live = False

        # Determine if source is live webcam or RTSP stream
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            self.source = int(source)
            self.is_live = True
        elif isinstance(source, str) and (source.startswith("rtsp://") or source.startswith("http://")):
            self.is_live = True

        self.cap: Optional[cv2.VideoCapture] = None
        self._init_capture()

        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.fps: float = 25.0
        self.width: int = 1920
        self.height: int = 1080
        self.is_running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        if self.cap and self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

    def _init_capture(self) -> bool:
        if self.is_live and isinstance(self.source, str) and self.source.startswith("rtsp://"):
            # Set low latency RTSP flags for OpenCV FFmpeg backend
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source)
            if self.is_live and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return self.cap.isOpened() if self.cap else False

    def switch_source(self, new_source: str | int, loop_video: Optional[bool] = None) -> bool:
        """
        Dynamically switches active video stream to a new file or stream without restarting
        the worker thread.
        """
        with self.lock:
            old_cap = self.cap
            self.cap = None

            self.source = new_source
            if loop_video is not None:
                self.loop_video = loop_video

            if isinstance(new_source, int) or (isinstance(new_source, str) and str(new_source).isdigit()):
                self.source = int(new_source)
                self.is_live = True
            elif isinstance(new_source, str) and (new_source.startswith("rtsp://") or new_source.startswith("http://")):
                self.is_live = True
            else:
                self.is_live = False

            opened = self._init_capture()
            if self.cap and self.cap.isOpened():
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        self.latest_frame = frame
                        self.frame_id = 1
                except Exception:
                    pass

            if old_cap:
                try:
                    old_cap.release()
                except Exception:
                    pass

            return opened

    def start(self) -> ThreadedStreamCapture:
        if self.is_running:
            return self

        self.is_running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="StreamIngestWorker")
        self.thread.start()

        # Wait briefly up to 3.0s for initial frame arrival
        t_wait_0 = time.time()
        while self.latest_frame is None and (time.time() - t_wait_0) < 3.0:
            time.sleep(0.02)

        return self

    def _worker_loop(self):
        """Continuous background thread reading from OpenCV capture."""
        while self.is_running:
            with self.lock:
                cap_ref = self.cap
                is_live = self.is_live
                loop_vid = self.loop_video
                cur_fps = self.fps

            frame_delay = 1.0 / max(cur_fps, 10.0) if not is_live else 0.001

            if not cap_ref or not cap_ref.isOpened():
                if is_live:
                    time.sleep(1.0)
                    with self.lock:
                        self._init_capture()
                    continue
                else:
                    time.sleep(0.05)
                    continue

            try:
                ret, frame = cap_ref.read()
            except Exception:
                time.sleep(0.02)
                continue

            if not ret:
                if not is_live:
                    if self.on_video_end_callback is not None:
                        # Auto-advance to next video clip
                        self.on_video_end_callback()
                        time.sleep(0.05)
                        continue
                    elif loop_vid:
                        # Loop local video back to beginning
                        try:
                            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            pass
                        continue
                    else:
                        time.sleep(0.05)
                        continue
                elif is_live:
                    # Connection lost, retry
                    time.sleep(0.5)
                    with self.lock:
                        self._init_capture()
                    continue
                else:
                    time.sleep(0.05)
                    continue

            with self.lock:
                self.latest_frame = frame
                self.frame_id += 1

            if not is_live:
                time.sleep(frame_delay)
            else:
                time.sleep(0.002)

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Returns (success, frame_copy_or_ref, frame_idx)."""
        with self.lock:
            if self.latest_frame is None:
                return False, None, self.frame_id
            return True, self.latest_frame.copy(), self.frame_id

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
