"""
YOLO Dataset Generator with Unicode Support.
Exports detected license plates to YOLO standard format with data.yaml, images, labels, and crops.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import random
import cv2
import numpy as np
import yaml
from .owod_engine import Detection
from .quality_auditor import audit_dataset

def save_image_unicode(path: Union[str, Path], img_bgr: np.ndarray) -> bool:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix if p.suffix else ".jpg"
        success, buf = cv2.imencode(ext, img_bgr)
        if success:
            with open(str(p), "wb") as f:
                f.write(buf)
            return True
    except Exception:
        pass
    return False


class DatasetGenerator:
    """Creates a standard YOLO detection dataset from labeled frames and images."""
    
    def __init__(
        self,
        output_dir: str | Path,
        train_ratio: float = 0.8,
        save_crops: bool = True,
        save_visualizations: bool = True
    ):
        self.output_dir = Path(output_dir).resolve()
        self.train_ratio = max(0.1, min(0.95, train_ratio))
        self.save_crops = save_crops
        self.save_visualizations = save_visualizations
        
        self.img_train_dir = self.output_dir / "images" / "train"
        self.img_val_dir = self.output_dir / "images" / "val"
        self.lbl_train_dir = self.output_dir / "labels" / "train"
        self.lbl_val_dir = self.output_dir / "labels" / "val"
        self.crops_dir = self.output_dir / "crops" / "license_plate"
        self.vis_dir = self.output_dir / "visualizations"
        
        self.total_saved_images = 0
        self.total_saved_plates = 0
        self._prepare_directories()

    def _prepare_directories(self):
        for d in [self.img_train_dir, self.img_val_dir, self.lbl_train_dir, self.lbl_val_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if self.save_crops:
            self.crops_dir.mkdir(parents=True, exist_ok=True)
        if self.save_visualizations:
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def save_sample(
        self,
        base_name: str,
        image_rgb: np.ndarray,
        detections: List[Detection],
        force_split: Optional[str] = None,
        allow_empty: bool = False
    ) -> Tuple[bool, str]:
        """
        Saves image and its YOLO label file.
        force_split: 'train' or 'val', or None (random based on train_ratio).
        """
        if not detections and not allow_empty:
            return False, "No detections to save"
            
        h, w = image_rgb.shape[:2]
        
        split = force_split
        if split not in ('train', 'val'):
            split = 'train' if random.random() < self.train_ratio else 'val'
            
        target_img_dir = self.img_train_dir if split == 'train' else self.img_val_dir
        target_lbl_dir = self.lbl_train_dir if split == 'train' else self.lbl_val_dir
        
        clean_name = Path(base_name).stem.replace(" ", "_")
        img_filename = f"{clean_name}.jpg"
        lbl_filename = f"{clean_name}.txt"
        
        img_path = target_img_dir / img_filename
        lbl_path = target_lbl_dir / lbl_filename
        
        # Save image (Unicode-safe)
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        save_image_unicode(img_path, img_bgr)
        
        # Write YOLO labels
        lines = []
        for i, det in enumerate(detections):
            cls_id, xc, yc, bw, bh = det.to_yolo_format(w, h)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            
            # Save crop if enabled
            if self.save_crops:
                x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
                x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))
                if x2 > x1 and y2 > y1:
                    crop_bgr = img_bgr[y1:y2, x1:x2]
                    crop_path = self.crops_dir / f"{clean_name}_crop_{i}.jpg"
                    save_image_unicode(crop_path, crop_bgr)
                    
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        # Save visualization if enabled and has detections
        if self.save_visualizations and detections:
            vis_img = image_rgb.copy()
            for det in detections:
                x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            save_image_unicode(self.vis_dir / f"{clean_name}_vis.jpg", vis_bgr)
            
        self.total_saved_images += 1
        self.total_saved_plates += len(detections)
        return True, split

    def generate_data_yaml(self) -> Path:
        """Generates YOLO dataset config data.yaml file."""
        yaml_content = {
            'path': self.output_dir.as_posix(),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'license_plate'
            }
        }
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        return yaml_path

    def generate_data_quality_report(self) -> dict:
        """
        Audits and generates an ISO/IEC 5259-2:2023 & ISO 25012 AI Data Quality Report
        for the generated dataset, saving data_quality_report.json to output_dir.
        """
        return audit_dataset(self.output_dir, save_report=True)
