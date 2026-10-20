"""
src/utils/metrics.py
Telemetry collector, latency breakdown, and EMA FPS calculation.
"""
import time
from typing import Dict


class LatencyTracker:
    """High-precision per-stage latency timer."""
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}

    def start(self, stage: str):
        self.timers[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        if stage in self.timers:
            elapsed_ms = (time.perf_counter() - self.timers[stage]) * 1000.0
            self.durations[stage] = elapsed_ms
            return elapsed_ms
        return 0.0

    def get_summary(self) -> Dict[str, float]:
        return dict(self.durations)


class EMAFpsCounter:
    """Exponential Moving Average (EMA) FPS calculator for smooth telemetry."""
    def __init__(self, alpha: float = 0.85):
        self.alpha = alpha
        self.last_time = time.perf_counter()
        self.current_fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self.last_time
        self.last_time = now

        if delta > 0.0:
            instant_fps = 1.0 / delta
            if self.current_fps == 0.0:
                self.current_fps = instant_fps
            else:
                self.current_fps = (self.alpha * self.current_fps) + ((1.0 - self.alpha) * instant_fps)

        return self.current_fps
