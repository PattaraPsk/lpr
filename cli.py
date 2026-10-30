"""
Unified Command-Line Interface for Thai LPR Edge System.
This interface replaces all root batch files with an argparse CLI.
The code strictly follows ASD-STE100 and ISO 24495-1:2023 standards.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent


def get_python_executable() -> str:
    """Return path to active Python executable."""
    venv_py_win = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    venv_py_nix = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_py_win.is_file():
        return str(venv_py_win)
    if venv_py_nix.is_file():
        return str(venv_py_nix)
    return sys.executable


def run_command(args: List[str], cwd: Optional[Path] = None) -> int:
    """Execute command synchronously and stream output."""
    work_dir = str(cwd or PROJECT_ROOT)
    print(f"[CLI] Execute: {' '.join(args)}")
    try:
        proc = subprocess.run(args, cwd=work_dir)
        return proc.returncode
    except KeyboardInterrupt:
        print("\n[CLI] Process interrupted by user.")
        return 130


# ---------------------------------------------------------------------------
# Command Handlers (ASD-STE100 Compliant: Active voice, clear purpose)
# ---------------------------------------------------------------------------

def cmd_train_detector(extra: List[str]) -> int:
    """Train YOLO vehicle plate detector and classifier."""
    py = get_python_executable()
    data_yaml = PROJECT_ROOT / "dataset" / "labeled" / "dataset-vehicle-orientation" / "data.yaml"
    if not data_yaml.is_file():
        data_yaml = PROJECT_ROOT / "dataset" / "dataset-vehicle-orientation" / "data.yaml"
    cmd = [py, "tools/train_models.py", "--task", "detector", "--data", str(data_yaml)] + extra
    return run_command(cmd)


def cmd_train_reader(extra: List[str]) -> int:
    """Train Thai plate reader OCR with baseline VisionAttentionOCR."""
    py = get_python_executable()
    data_dir = "dataset/ready/202610261140,dataset/dataset-cctv-synthetic/plates,dataset/dataset-mobile-synthetic/plates"
    cmd = [py, "tools/train_models.py", "--task", "reader", "--data", data_dir, "--batch", "64", "--epochs", "30"] + extra
    return run_command(cmd)


def cmd_train_reader_a2(extra: List[str]) -> int:
    """Train EdgeAttentionOCR Thai reader with MobileNetV3 and attention."""
    py = get_python_executable()
    cmd = [py, "tools/train_edge_attention_ocr.py", "--epochs", "15", "--batch-size", "64"] + extra
    return run_command(cmd)


def cmd_train_selen_lpr(extra: List[str]) -> int:
    """Train SelenLPR unified multi-task model with causal masking."""
    py = get_python_executable()
    cmd = [py, "tools/train_selen_lpr.py", "--epochs", "15", "--batch-size", "64"] + extra
    return run_command(cmd)


def cmd_train_selen_lpr_v2(extra: List[str]) -> int:
    """Train SelenLPR v2 continued training runner to 500 epochs."""
    py = get_python_executable()
    resume_path = "models/reader/selen-lpr-0.0.1/best_reader.pth"
    script_path = "tools/archive/train_selen_lpr_v2.py"
    cmd = [
        py, script_path,
        "--epochs", "500",
        "--batch-size", "64",
        "--lr", "0.0003",
        "--loss-mode", "clamped_kendall",
        "--resume", resume_path,
        "--device", "cuda:0"
    ] + extra
    return run_command(cmd)


def cmd_demo_realtime(extra: List[str]) -> int:
    """Run real-time edge ALPR demo on video, webcam, or RTSP stream."""
    py = get_python_executable()
    cmd = [py, "demo_realtime.py"] + extra
    return run_command(cmd)


def cmd_track_lpr(extra: List[str]) -> int:
    """Run multi-video tracker and extract optimal plate snapshots."""
    py = get_python_executable()
    cmd = [py, "tools/process_and_track_lpr.py"] + extra
    return run_command(cmd)


def cmd_detect_cam(extra: List[str]) -> int:
    """Run plate detection and auto-tracking on recorded camera folders."""
    py = get_python_executable()
    cam_dir = "dataset/raw/Raw Video Record/CAM01/2026/10/19"
    cmd = [py, "tools/detect_cam_videos.py", "--input-dir", cam_dir] + extra
    return run_command(cmd)


def cmd_collect_vehicles(extra: List[str]) -> int:
    """Collect vehicle images and license plate crops across videos."""
    py = get_python_executable()
    cmd = [py, "tools/collect_vehicles_and_plates.py"] + extra
    return run_command(cmd)


def cmd_generate_cctv(extra: List[str]) -> int:
    """Generate synthetic CCTV license plate images with weather effects."""
    py = get_python_executable()
    cmd = [py, "tools/generate_cctv_synthetic_dataset.py"] + extra
    return run_command(cmd)


def cmd_generate_mobile(extra: List[str]) -> int:
    """Generate synthetic mobile CCTV license plate images."""
    py = get_python_executable()
    cmd = [py, "tools/generate_mobile_cctv_synthetic_dataset.py"] + extra
    return run_command(cmd)


def cmd_plate_augmentor(extra: List[str]) -> int:
    """Launch Plate Augmentor desktop application."""
    py = get_python_executable()
    cmd = [py, "tools/PlateAugmentor/main.py"] + extra
    return run_command(cmd)


def cmd_owod_studio(extra: List[str]) -> int:
    """Launch OWOD Studio desktop application for pruning and auto-labeling."""
    py = get_python_executable()
    cmd = [py, "tools/owod/main.py"] + extra
    return run_command(cmd, cwd=PROJECT_ROOT / "tools" / "owod")


def cmd_autorun_owod(extra: List[str]) -> int:
    """Run automatic OWOD dataset labeling on all files in dataset folder."""
    py = get_python_executable()
    cmd = [py, "tools/owod/autorun_dataset.py"] + extra
    return run_command(cmd)


def cmd_datamanipulator(extra: List[str]) -> int:
    """Launch DataManipulator dataset annotation and review tool."""
    py = get_python_executable()
    cmd = [py, "tools/DataManipulator/main.py"] + extra
    return run_command(cmd)


def cmd_test(extra: List[str]) -> int:
    """Run ISO compliance and system verification test suites."""
    py = get_python_executable()
    cmd = [py, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"] + extra
    return run_command(cmd)


def cmd_audit_quality(extra: List[str]) -> int:
    """Run ISO/IEC 5259 data quality audit on target dataset."""
    py = get_python_executable()
    target = extra[0] if extra else "dataset/yolo_dlt_lpr_dataset"
    cmd = [py, "tools/owod/core/quality_auditor.py", target]
    return run_command(cmd)


def cmd_dashboard_api(extra: List[str]) -> int:
    """Launch FastAPI backend server for unified web dashboard."""
    py = get_python_executable()
    cmd = [py, "-m", "dashboard.api"] + extra
    return run_command(cmd)


def cmd_dashboard_ui(extra: List[str]) -> int:
    """Launch Streamlit frontend application for unified web dashboard."""
    cmd = ["streamlit", "run", "dashboard/app.py"] + extra
    return run_command(cmd)


def cmd_interactive_menu() -> int:
    """Display interactive CLI menu to select and launch tools."""
    print("=" * 60)
    print("  Thai LPR Edge System - Command-Line Interface")
    print("=" * 60)
    print("  [0] Run Real-Time Demo (demo_realtime.py)")
    print("  [1] Launch OWOD Studio GUI")
    print("  [2] Auto-Run OWOD Labeling")
    print("  [3] Launch DataManipulator GUI")
    print("  [4] Run CAM01 Detection & Tracking")
    print("  [5] Run ISO Compliance Tests")
    print("  [6] Train Plate Detector (YOLO)")
    print("  [7] Train Reader OCR (VisionAttentionOCR)")
    print("  [8] Train Reader A2 (EdgeAttentionOCR)")
    print("  [9] Train SelenLPR Multi-Task Model")
    print("  [10] Launch Plate Augmentor GUI")
    print("  [11] Launch Web Dashboard API")
    print("  [12] Launch Web Dashboard Streamlit UI")
    print("  [q] Quit")
    print("=" * 60)
    choice = input("Select an option (default 0): ").strip()
    if choice in ("q", "quit", "exit"):
        return 0

    menu_map = {
        "0": cmd_demo_realtime,
        "1": cmd_owod_studio,
        "2": cmd_autorun_owod,
        "3": cmd_datamanipulator,
        "4": cmd_detect_cam,
        "5": cmd_test,
        "6": cmd_train_detector,
        "7": cmd_train_reader,
        "8": cmd_train_reader_a2,
        "9": cmd_train_selen_lpr,
        "10": cmd_plate_augmentor,
        "11": cmd_dashboard_api,
        "12": cmd_dashboard_ui,
    }
    action = menu_map.get(choice, cmd_demo_realtime)
    return action([])


# ---------------------------------------------------------------------------
# CLI Argument Parser Construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build root CLI parser with subcommands matching all tasks."""
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Unified Command-Line Interface for Thai LPR Edge System.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available system commands")

    # Command registrations
    cmds = [
        ("train-detector", "Train plate detector and vehicle classifier (YOLO)", cmd_train_detector),
        ("train-reader", "Train OCR reader with VisionAttentionOCR", cmd_train_reader),
        ("train-reader-a2", "Train EdgeAttentionOCR Thai reader", cmd_train_reader_a2),
        ("train-selen-lpr", "Train SelenLPR unified multi-task model", cmd_train_selen_lpr),
        ("train-selen-lpr-v2", "Train SelenLPR v2 continued runner to 500 epochs", cmd_train_selen_lpr_v2),
        ("demo-realtime", "Run real-time ALPR demo on camera or video", cmd_demo_realtime),
        ("track-lpr", "Process video and track plate snapshots", cmd_track_lpr),
        ("detect", "Run detection on CAM01 recorded directory", cmd_detect_cam),
        ("collect-vehicles", "Collect vehicles and plates from video", cmd_collect_vehicles),
        ("generate-cctv", "Generate synthetic CCTV license plates", cmd_generate_cctv),
        ("generate-mobile", "Generate synthetic mobile CCTV license plates", cmd_generate_mobile),
        ("plate-augmentor", "Launch Plate Augmentor desktop GUI", cmd_plate_augmentor),
        ("owod-studio", "Launch OWOD Studio desktop GUI", cmd_owod_studio),
        ("autorun-owod", "Auto-run OWOD dataset labeling pipeline", cmd_autorun_owod),
        ("datamanipulator", "Launch DataManipulator dataset GUI", cmd_datamanipulator),
        ("test", "Run test suite for compliance verification", cmd_test),
        ("audit-quality", "Run ISO/IEC 5259 data quality audit", cmd_audit_quality),
        ("dashboard-api", "Launch FastAPI web dashboard backend", cmd_dashboard_api),
        ("dashboard-ui", "Launch Streamlit web dashboard frontend", cmd_dashboard_ui),
        ("menu", "Display interactive selection menu", lambda extra: cmd_interactive_menu()),
    ]

    for name, desc, handler in cmds:
        sp = subparsers.add_parser(name, help=desc)
        sp.add_argument("extra", nargs=argparse.REMAINDER, help="Forwarded arguments")
        sp.set_defaults(handler=handler)

    return parser


def main() -> None:
    """Parse command-line arguments and route to targeted handler."""
    parser = build_parser()
    if len(sys.argv) == 1:
        sys.exit(cmd_interactive_menu())

    args = parser.parse_args()
    if hasattr(args, "handler"):
        extra_args = getattr(args, "extra", [])
        # Strip leading '--' if present
        if extra_args and extra_args[0] == "--":
            extra_args = extra_args[1:]
        sys.exit(args.handler(extra_args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
