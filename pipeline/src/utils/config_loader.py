"""
src/utils/config_loader.py
Configuration loader with default fallback support for Thai LPR.
"""
import os
from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "name": "Thai Edge LPR Production Pipeline",
        "version": "2.0.0",
        "target_device": "auto",
        "log_level": "INFO",
    },
    "detector": {
        "backend": "yolo",
        "model_path": "models/yolov8s-worldv2.pt",
        "confidence_threshold": 0.25,
        "nms_threshold": 0.45,
        "padding_ratio": 0.06,
        "target_classes": ["license plate", "vehicle registration plate", "number plate"],
        "input_size": [640, 640],
        "half_precision": True,
    },
    "ocr": {
        "engine": "segmenter",
        "target_char_height": 128,
        "deskew_plate": True,
        "max_deskew_angle": 25.0,
        "filter_screws": True,
        "vlm_model": "Qwen/Qwen2-VL-2B-Instruct",
    },
    "formatting": {
        "enforce_thai_dlt_rules": True,
        "default_province": "กรุงเทพมหานคร",
        "fuzzy_province_matching": True,
        "output_format": "{consonants}_{digits}_{province}",
    },
    "stream": {
        "queue_size": 2,
        "drop_frames_when_full": True,
        "reconnect_delay_sec": 5.0,
        "max_fps": 30.0,
    },
    "tracking": {
        "enabled": True,
        "iou_threshold": 0.30,
        "max_lost_frames": 15,
        "ema_alpha": 0.75,
    },
    "visualization": {
        "show_bounding_boxes": True,
        "show_labels": True,
        "show_fps": True,
        "box_thickness": 2,
        "font_scale": 0.55,
        "box_color": [0, 230, 118],
    },
}


def load_config(config_path: str = "configs/config.default.yaml") -> Dict[str, Any]:
    """Loads configuration YAML file and merges missing keys with DEFAULT_CONFIG."""
    config = dict(DEFAULT_CONFIG)
    path = Path(config_path)

    if not path.exists():
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        alt_path = repo_root / config_path
        if alt_path.exists():
            path = alt_path

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
                # Deep merge top-level dictionaries
                for k, v in user_cfg.items():
                    if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                        config[k].update(v)
                    else:
                        config[k] = v
        except Exception as e:
            print(f"[ConfigLoader] Warning: Failed to parse '{path}' ({e}). Using defaults.")
    else:
        print(f"[ConfigLoader] Notice: Config file '{config_path}' not found. Using default config.")

    return config
