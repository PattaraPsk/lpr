#!/usr/bin/env python3
"""
main.py
Production CLI Entrypoint for Thai License Plate Recognition (ALPR/ANPR).
Supports Stream, Video, Image, and Benchmark execution modes out-of-the-box.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import cv2

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.capture.stream import ThreadedStreamCapture
from src.capture.video_reader import ImageDirectoryReader, VideoFileReader
from src.pipeline.lpr_pipeline import LPRPipeline
from src.utils.config_loader import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ThaiLPR")


def run_stream_mode(pipeline: LPRPipeline, source: str, show: bool = True, output: Optional[str] = None):
    """Processes real-time RTSP or webcam stream with zero-latency frame dropping."""
    logger.info(f"Starting Stream Mode on source: '{source}'")
    capture = ThreadedStreamCapture(source=source, queue_size=2, drop_frames=True)
    capture.start()

    writer = None
    frame_idx = 0

    try:
        while True:
            ret, frame = capture.read(timeout=2.0)
            if not ret or frame is None:
                logger.warning("Stream read timeout. Waiting for reconnection...")
                continue

            frame_idx += 1
            results, vis = pipeline.process_frame(frame, frame_id=frame_idx, render_vis=True)

            # Log read plates
            for res in results:
                logger.info(f"[Frame {frame_idx:05d}] Plate: {res.reading.formatted_string} (DLT Valid: {res.reading.is_valid_dlt})")

            if output and vis is not None:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, 25.0, (w, h))
                writer.write(vis)

            if show and vis is not None:
                cv2.imshow("Thai Edge LPR - Live Stream", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Termination key 'q' pressed.")
                    break

    except KeyboardInterrupt:
        logger.info("Ctrl+C received. Terminating stream...")
    finally:
        capture.stop()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def run_video_mode(pipeline: LPRPipeline, source: str, show: bool = False, output: Optional[str] = None):
    """Processes a local video file frame-by-frame."""
    logger.info(f"Starting Video Processing on file: '{source}'")
    reader = VideoFileReader(source)

    writer = None
    total = reader.total_frames
    logger.info(f"Video file opened: {reader.width}x{reader.height} @ {reader.fps:.1f} FPS ({total} frames)")

    try:
        for idx, frame in reader:
            results, vis = pipeline.process_frame(frame, frame_id=idx, render_vis=(show or output is not None))

            for res in results:
                logger.info(f"[Frame {idx:05d}/{total}] Plate: {res.reading.formatted_string} (DLT Valid: {res.reading.is_valid_dlt})")

            if output and vis is not None:
                if writer is None:
                    h, w = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output, fourcc, reader.fps, (w, h))
                writer.write(vis)

            if show and vis is not None:
                cv2.imshow("Thai Edge LPR - Video Playback", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        reader.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def run_image_mode(pipeline: LPRPipeline, source: str, output_dir: Optional[str] = "output"):
    """Processes an image file or directory of images."""
    reader = ImageDirectoryReader(source)
    logger.info(f"Starting Image Processing mode on: '{source}' ({len(reader)} images)")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for path, img in reader:
        results, vis = pipeline.process_frame(img, frame_id=0, render_vis=True)
        readings_str = ", ".join([r.reading.formatted_string for r in results]) or "NO_PLATE"
        logger.info(f"[{Path(path).name}] -> {readings_str}")

        if output_dir and vis is not None:
            out_path = Path(output_dir) / f"lpr_{Path(path).name}"
            cv2.imwrite(str(out_path), vis)


def run_benchmark_mode(pipeline: LPRPipeline, source: Optional[str] = None, max_frames: int = 100):
    """Runs a performance, throughput, and accuracy benchmark."""
    logger.info(f"============================================================")
    logger.info(f"       THAI LPR PIPELINE - EDGE RUNTIME BENCHMARK")
    logger.info(f"============================================================")

    # Use sample test video or synthetic frame fixture
    test_source = source or "tests/samples/sample.mp4"
    if not Path(test_source).exists():
        # Fallback synthetic frame generator
        logger.info("Using synthetic test frame fixture for benchmark...")
        frames = [np.full((720, 1280, 3), 120, dtype=np.uint8) for _ in range(max_frames)]
    else:
        reader = VideoFileReader(test_source)
        frames = [frame for _, frame in reader][:max_frames]
        reader.release()

    count = len(frames)
    logger.info(f"Benchmarking inference over {count} frames...")

    # Warmup
    if count > 0:
        pipeline.process_frame(frames[0], frame_id=0)

    t0 = time.perf_counter()
    detected_plates = 0
    valid_dlt_count = 0

    for idx, frame in enumerate(frames):
        results, _ = pipeline.process_frame(frame, frame_id=idx)
        detected_plates += len(results)
        for r in results:
            if r.reading.is_valid_dlt:
                valid_dlt_count += 1

    total_time = time.perf_counter() - t0
    avg_latency = (total_time / max(1, count)) * 1000.0
    throughput_fps = count / max(1e-4, total_time)

    print("\n" + "=" * 65)
    print("                    BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Frames Processed : {count}")
    print(f"Total Wall-Clock Time  : {total_time:.2f} seconds")
    print(f"Average Frame Latency  : {avg_latency:.2f} ms")
    print(f"Inference Throughput   : {throughput_fps:.2f} FPS")
    print(f"Plates Detected        : {detected_plates}")
    print(f"Valid DLT Formats      : {valid_dlt_count}")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thai Edge License Plate Recognition (ALPR/ANPR) System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "video", "image", "benchmark"],
        default="auto",
        help="Pipeline operation mode ('auto' infers from --source).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input video file path, RTSP stream URL, camera device index (e.g. 0), or image folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output annotated video file or image directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV live visualization window.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Shortcut to trigger benchmark mode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum number of frames for benchmark mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    pipeline = LPRPipeline(config_path=args.config)

    # Determine execution mode
    if args.benchmark or args.mode == "benchmark":
        run_benchmark_mode(pipeline, source=args.source, max_frames=args.max_frames)
        return

    source = args.source
    if not source:
        # Default to stream from config or local webcam
        stream_cfg_path = Path("configs/camera_streams.yaml")
        if stream_cfg_path.exists():
            import yaml
            with open(stream_cfg_path, "r", encoding="utf-8") as f:
                s_data = yaml.safe_load(f) or {}
                streams = s_data.get("streams", {})
                if "dashcam_sample" in streams and Path(streams["dashcam_sample"].get("url", "")).exists():
                    source = streams["dashcam_sample"]["url"]
                elif "webcam_local" in streams:
                    source = streams["webcam_local"]["url"]
        source = source or "0"

    mode = args.mode
    if mode == "auto":
        src_lower = str(source).lower()
        if src_lower.startswith("rtsp://") or src_lower.isdigit():
            mode = "stream"
        elif any(src_lower.endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov"]):
            mode = "video"
        elif Path(source).is_dir() or any(src_lower.endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
            mode = "image"
        else:
            mode = "stream"

    logger.info(f"Executing in '{mode.upper()}' mode on source: {source}")

    if mode == "stream":
        run_stream_mode(pipeline, source=source, show=args.show, output=args.output)
    elif mode == "video":
        run_video_mode(pipeline, source=source, show=args.show, output=args.output)
    elif mode == "image":
        run_image_mode(pipeline, source=source, output_dir=args.output)


if __name__ == "__main__":
    main()
