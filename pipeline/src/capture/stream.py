"""
src/capture/stream.py
Multi-threaded RTSP and video ingestion with zero-latency frame dropping.
Prevents OpenCV buffer bloat by maintaining a bounded queue with automatic eviction.
"""
import logging
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("LPRStream")


class ThreadedStreamCapture:
    """
    Thread-safe video/RTSP stream reader that eliminates latency buildup.
    Evicts stale frames when the processing pipeline runs slower than stream FPS.
    """
    def __init__(
        self,
        source: str,
        queue_size: int = 2,
        drop_frames: bool = True,
        reconnect_delay: float = 5.0,
        max_fps: Optional[float] = 30.0,
    ):
        self.source = source
        self.queue_size = max(1, queue_size)
        self.drop_frames = drop_frames
        self.reconnect_delay = reconnect_delay
        self.max_fps = max_fps
        self.frame_delay = (1.0 / max_fps) if max_fps and max_fps > 0 else 0.0

        # Resolve camera device index if integer string
        if isinstance(source, str) and source.isdigit():
            self.resolved_source = int(source)
        else:
            self.resolved_source = source

        self.frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.total_frames_read = 0
        self.total_frames_dropped = 0
        self.is_video_file = isinstance(self.resolved_source, str) and not self.resolved_source.startswith("rtsp://")

    def start(self) -> "ThreadedStreamCapture":
        if self.running:
            return self

        self._connect()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name="StreamCaptureThread")
        self.thread.start()
        logger.info(f"Stream capture thread started for source: '{self.source}'")
        return self

    def _connect(self):
        """Initializes or reconnects cv2.VideoCapture."""
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Connecting to video source: '{self.source}'...")
        self.cap = cv2.VideoCapture(self.resolved_source)

        if not self.cap.isOpened():
            logger.warning(f"Failed to open video source: '{self.source}'")

    def _capture_loop(self):
        while self.running:
            t0 = time.perf_counter()

            if self.cap is None or not self.cap.isOpened():
                if not self.running:
                    break
                logger.info(f"Reconnecting in {self.reconnect_delay:.1f}s...")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_video_file:
                    # Loop video files on EOF
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        logger.info("Video file EOF reached.")
                        break
                else:
                    logger.warning("Stream frame read failed. Triggering reconnect...")
                    self.cap.release()
                    continue

            self.total_frames_read += 1

            # Evict stale frame if queue is full to prevent pipeline lag
            if self.frame_queue.full():
                if self.drop_frames:
                    try:
                        self.frame_queue.get_nowait()
                        self.total_frames_dropped += 1
                    except queue.Empty:
                        pass
                else:
                    time.sleep(0.005)

            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.total_frames_dropped += 1

            # FPS pacing for local video playback
            if self.is_video_file and self.frame_delay > 0.0:
                elapsed = time.perf_counter() - t0
                sleep_time = self.frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves latest fresh frame from queue with timeout."""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logger.info(f"Stream stopped. Read: {self.total_frames_read}, Dropped: {self.total_frames_dropped}")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
