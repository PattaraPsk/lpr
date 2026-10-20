#!/usr/bin/env python3
"""
scripts/check_environment.py
Diagnostic utility to verify Python runtime, GPU acceleration, and subsystem dependencies.
"""
import importlib
import os
import platform
import sys

def check_package(pkg_name: str, import_name: str = None) -> bool:
    target = import_name or pkg_name
    try:
        mod = importlib.import_module(target)
        ver = getattr(mod, "__version__", "installed")
        print(f"  [+] {pkg_name:<20}: {ver}")
        return True
    except ImportError:
        print(f"  [-] {pkg_name:<20}: NOT FOUND")
        return False

def main():
    print("=" * 70)
    print("           THAI LPR PIPELINE - SYSTEM ENVIRONMENT AUDIT")
    print("=" * 70)
    print(f"OS Platform       : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Python Runtime    : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Executable Path   : {sys.executable}")
    
    is_py310 = sys.version_info[:2] == (3, 10)
    print(f"Python 3.10 Check : {'PASS (Optimal)' if is_py310 else 'WARNING (Target is 3.10.x)'}")
    print("-" * 70)

    # CUDA & PyTorch
    print("[1] PyTorch & Hardware Acceleration:")
    try:
        import torch
        print(f"  [+] PyTorch Version   : {torch.__version__}")
        print(f"  [+] CUDA Available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [+] CUDA Version      : {torch.version.cuda}")
            print(f"  [+] GPU Device Name   : {torch.cuda.get_device_name(0)}")
            print(f"  [+] Device Count      : {torch.cuda.device_count()}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [+] Total VRAM        : {vram_gb:.2f} GB")
        else:
            print("  [!] Running in CPU-only execution mode.")
    except ImportError:
        print("  [-] PyTorch is not installed in the active environment.")

    print("-" * 70)
    print("[2] Core Vision & Image Processing:")
    check_package("OpenCV", "cv2")
    check_package("Pillow", "PIL")
    check_package("NumPy", "numpy")
    check_package("SciPy", "scipy")
    check_package("FontTools", "fontTools")

    print("-" * 70)
    print("[3] Deep Learning & Vision-Language Backbones:")
    check_package("Transformers", "transformers")
    check_package("Accelerate", "accelerate")
    check_package("Ultralytics (YOLO)", "ultralytics")
    check_package("Timm", "timm")
    check_package("ONNX Runtime", "onnxruntime")

    print("-" * 70)
    print("[4] Web, Scraping & Utilities:")
    check_package("Requests", "requests")
    check_package("Tqdm", "tqdm")
    check_package("FastAPI", "fastapi")
    check_package("Uvicorn", "uvicorn")
    check_package("Selenium", "selenium")

    print("=" * 70)
    print("Audit completed.")

if __name__ == "__main__":
    main()
