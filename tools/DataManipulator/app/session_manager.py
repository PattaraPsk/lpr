"""
Session and File Management for LPR Dataset Annotation.
Supports recursive subfolders, grouping by subfolders, custom rename,
single-image delete, and batch removing '_dup' duplicate images.
Filename structure: {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
"""

import os
import re
import json
import shutil
import unicodedata
from datetime import datetime, timezone
from .provinces import normalize_thai_script, get_iso_code
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
_new_staging = ROOT_DIR / "dataset" / "verified" / "dataset-staging-approved"
_legacy_staging = ROOT_DIR / "dataset" / "dataset-staging-approved"
DEFAULT_STAGING_DIR = _new_staging if _new_staging.exists() or not _legacy_staging.exists() else _legacy_staging

try:
    import yaml
except ImportError:
    yaml = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SESSION_FILENAME = ".lpr_session.json"

class ImageItem:
    def __init__(self, full_path: Path, root_folder: Optional[Path] = None):
        self.path = full_path.resolve()
        self.filename = self.path.name
        self.ext = self.path.suffix
        self.stem = self.path.stem
        
        # Subfolder & Relative Path handling
        if root_folder:
            try:
                self.rel_path = str(self.path.relative_to(root_folder.resolve())).replace("\\", "/")
                parent_rel = str(self.path.parent.relative_to(root_folder.resolve())).replace("\\", "/")
                self.subfolder = "" if parent_rel == "." else parent_rel
            except ValueError:
                self.rel_path = self.filename
                self.subfolder = ""
        else:
            self.rel_path = self.filename
            self.subfolder = ""
            
        # Parsed elements
        self.name: str = ""
        self.plate_type: str = "WhiteBlackCar"
        self.d3: str = ""
        self.d4: str = ""
        self.province: str = ""
        
        # AI Reader (edge_reader) predictions
        self.ai_d3: str = ""
        self.ai_d4: str = ""
        self.ai_province: str = ""
        self.ai_confidence: float = 0.0
        
        # Status: "pending", "approved", "not_plate", "cannot_read", "skipped"
        self.status: str = "pending"
        
        # Bounding boxes for YOLO detection
        self.boxes: List[Dict[str, Any]] = []
        self.label_path: Optional[Path] = None
        self._boxes_loaded: bool = False
        
        self.parse_filename()
        
    def parse_filename(self):
        """
        Parses filename structure: {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
        Supports Uppercase-separated plate types (e.g. WhiteBlackCar, WhiteBlackMotor, YellowBlue),
        legacy underscore aliases, special codes (+, -), and allows underscores inside {name}.
        """
        from .plate_types import PLATE_TYPES_DATA, SPECIAL_PLATE_TYPES, match_plate_type
        
        parts = self.stem.rsplit('_', 3)
        if len(parts) == 4:
            prefix, self.d3, self.d4, self.province = parts
            
            # Map of token -> canonical id (including aliases)
            tokens_map = {}
            for pt in PLATE_TYPES_DATA:
                tokens_map[pt["id"]] = pt["id"]
                for a in pt.get("aliases", []):
                    tokens_map[a] = pt["id"]
            for s, _ in SPECIAL_PLATE_TYPES:
                tokens_map[s] = s
                
            sorted_tokens = sorted(tokens_map.keys(), key=len, reverse=True)
            matched = False
            for token in sorted_tokens:
                suffix = f"_{token}"
                if prefix.endswith(suffix):
                    self.name = prefix[:-len(suffix)]
                    self.plate_type = tokens_map[token]
                    matched = True
                    break
                    
            if not matched:
                if '_' in prefix:
                    p_name, p_type = prefix.rsplit('_', 1)
                    self.name = p_name
                    self.plate_type = match_plate_type(p_type)
                else:
                    self.name = prefix
                    self.plate_type = "WhiteBlackCar"
        elif len(parts) == 3:
            self.name = parts[0]
            self.plate_type = "WhiteBlackCar"
            self.d3 = parts[1]
            self.d4 = parts[2]
            self.province = ""
        elif len(parts) == 2:
            self.name = parts[0]
            self.plate_type = "WhiteBlackCar"
            self.d3 = parts[1]
            self.d4 = ""
            self.province = ""
        else:
            self.name = self.stem
            self.plate_type = "WhiteBlackCar"
            self.d3 = ""
            self.d4 = ""
            self.province = ""
            
        # Pre-detect status if named with + or -
        if self.d3 == "+" and self.d4 == "+" and self.province == "+":
            self.status = "not_plate"
            self.plate_type = "+"
        elif self.d3 == "-" and self.d4 == "-" and self.province == "-":
            self.status = "cannot_read"
            self.plate_type = "-"

    def set_ai_prediction(self, d3: str, d4: str, province: str, confidence: float = 0.0):
        self.ai_d3 = d3
        self.ai_d4 = d4
        self.ai_province = province
        self.ai_confidence = confidence

    def get_ai_comparison(self) -> Dict[str, Any]:
        """Compares current manual/file values with AI predictions."""
        norm_d3 = normalize_thai_script(self.d3).strip()
        norm_ai_d3 = normalize_thai_script(self.ai_d3).strip()
        norm_d4 = self.d4.strip()
        norm_ai_d4 = self.ai_d4.strip()
        norm_prov = normalize_thai_script(self.province).strip()
        norm_ai_prov = normalize_thai_script(self.ai_province).strip()
        
        has_ai = bool(self.ai_d3 or self.ai_d4 or self.ai_province)
        d3_match = (norm_d3 == norm_ai_d3) if has_ai else False
        d4_match = (norm_d4 == norm_ai_d4) if has_ai else False
        prov_match = (norm_prov == norm_ai_prov) if has_ai else False
        all_match = (d3_match and d4_match and prov_match) if has_ai else False

        return {
            "has_ai": has_ai,
            "d3_match": d3_match,
            "d4_match": d4_match,
            "prov_match": prov_match,
            "all_match": all_match,
            "curr_d3": norm_d3,
            "ai_d3": norm_ai_d3,
            "curr_d4": norm_d4,
            "ai_d4": norm_ai_d4,
            "curr_prov": norm_prov,
            "ai_prov": norm_ai_prov,
            "confidence": self.ai_confidence
        }

    def find_label_path(self, root_folder: Optional[Path] = None) -> Optional[Path]:
        """Finds corresponding YOLO .txt label file on disk."""
        # 1. Fast path: string substitution /images/ -> /labels/
        p_str = str(self.path)
        if "/images/" in p_str:
            cand = Path(p_str.replace("/images/", "/labels/")).with_suffix('.txt')
            if cand.exists():
                return cand
        if "\\images\\" in p_str:
            cand = Path(p_str.replace("\\images\\", "\\labels\\")).with_suffix('.txt')
            if cand.exists():
                return cand

        # 2. Same directory
        cand = self.path.with_suffix('.txt')
        if cand.exists():
            return cand
                
        # 3. Via root_folder structure
        if root_folder:
            try:
                rel = self.path.relative_to(root_folder)
                rel_parts = list(rel.parts)
                if rel_parts and rel_parts[0] == "images":
                    rel_parts[0] = "labels"
                    cand = (root_folder / Path(*rel_parts)).with_suffix('.txt')
                    if cand.exists():
                        return cand
            except Exception:
                pass
            try:
                cand = (root_folder.parent / "labels" / self.path.name).with_suffix('.txt')
                if cand.exists():
                    return cand
            except Exception:
                pass

        return None

    def load_yolo_boxes(self, class_names: Optional[Dict[int, str]] = None, root_folder: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Loads bounding boxes from YOLO label file."""
        if class_names is None:
            class_names = {}
        self.label_path = self.find_label_path(root_folder)
        self.boxes = []
        if self.label_path and self.label_path.exists():
            try:
                with open(self.label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            cls_name = class_names.get(cls_id, f"Class {cls_id}")
                            self.boxes.append({
                                "cls_id": cls_id,
                                "cls_name": cls_name,
                                "bbox_norm": (cx, cy, w, h)
                            })
            except Exception as e:
                print(f"Warning: Failed to load labels from {self.label_path}: {e}")
        self._boxes_loaded = True
        return self.boxes

    def save_yolo_boxes(self, boxes: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Saves bounding boxes to disk in YOLO format."""
        if boxes is not None:
            self.boxes = boxes
        if not self.label_path:
            p_str = str(self.path)
            if "/images/" in p_str:
                self.label_path = Path(p_str.replace("/images/", "/labels/")).with_suffix('.txt')
            elif "\\images\\" in p_str:
                self.label_path = Path(p_str.replace("\\images\\", "\\labels\\")).with_suffix('.txt')
            else:
                self.label_path = self.path.with_suffix('.txt')
                
        try:
            self.label_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for b in self.boxes:
                cls_id = b.get("cls_id", 0)
                if "bbox_norm" in b:
                    cx, cy, w, h = b["bbox_norm"]
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                elif "box_xyxy" in b:
                    x1, y1, x2, y2 = b["box_xyxy"]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    w = x2 - x1
                    h = y2 - y1
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    
            with open(self.label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            return True
        except Exception as e:
            print(f"Error saving yolo boxes to {self.label_path}: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        norm_prov = normalize_thai_script(self.province)
        d = {
            "filename": self.filename,
            "rel_path": self.rel_path,
            "subfolder": self.subfolder,
            "name": self.name,
            "plate_type": self.plate_type,
            "d3": normalize_thai_script(self.d3),
            "d4": self.d4.strip() if self.d4 else "",
            "province": norm_prov,
            "iso_3166_2": get_iso_code(norm_prov),
            "status": self.status,
            "ai_d3": self.ai_d3,
            "ai_d4": self.ai_d4,
            "ai_province": self.ai_province,
            "ai_confidence": self.ai_confidence,
        }
        if self.boxes:
            d["boxes"] = self.boxes
        return d
        
    def from_dict(self, data: Dict[str, Any]):
        self.status = data.get("status", self.status)
        if "name" in data:
            self.name = data["name"]
        if "plate_type" in data:
            self.plate_type = data["plate_type"]
        if "d3" in data:
            self.d3 = data["d3"]
        if "d4" in data:
            self.d4 = data["d4"]
        if "province" in data:
            self.province = data["province"]
        if "ai_d3" in data:
            self.ai_d3 = data["ai_d3"]
        if "ai_d4" in data:
            self.ai_d4 = data["ai_d4"]
        if "ai_province" in data:
            self.ai_province = data["ai_province"]
        if "ai_confidence" in data:
            self.ai_confidence = float(data["ai_confidence"])
        if "boxes" in data:
            self.boxes = data["boxes"]


class SessionManager:
    def __init__(self):
        self.folder_path: Optional[Path] = None
        self.items: List[ImageItem] = []
        self.current_index: int = 0
        self.active_subfolder: Optional[str] = None # None means All
        self.dataset_type: str = "ocr" # "ocr" or "yolo"
        self.class_names: Dict[int, str] = {
            0: "Front",
            1: "Rear",
            2: "LicensePlate",
            3: "Car",
            4: "Motorcycle",
            5: "Truck",
            6: "Bus"
        }
        self.active_class_filter: Optional[int] = None
        self.active_status_filter: str = "all"
        self.staging_dir: Path = DEFAULT_STAGING_DIR
        self.auto_staging: bool = True
        
    def load_folder(self, folder_str: str) -> bool:
        """Recursively loads images in folder (including all subfolders) and restores session."""
        folder = Path(folder_str)
        if not folder.is_dir():
            return False
            
        self.folder_path = folder
        self.items = []
        self.active_subfolder = None
        self.active_class_filter = None
        self.active_status_filter = "all"
        
        # Recursively collect all image files, skipping hidden dirs and cache
        image_files = []
        for root, dirs, files in os.walk(folder):
            # Ignore hidden dirs (.git, .trash, .cache, etc.) and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_files.append(Path(root) / f)
                    
        # Sort files naturally by relative path
        image_files.sort(key=lambda p: str(p.relative_to(folder)).lower())
        
        for file in image_files:
            self.items.append(ImageItem(file, folder))
            
        # Auto-detect dataset type and class definitions
        self.dataset_type = "ocr"
        self.class_names = {
            0: "Front",
            1: "Rear",
            2: "LicensePlate",
            3: "Car",
            4: "Motorcycle",
            5: "Truck",
            6: "Bus"
        }
        
        yaml_path = None
        for cand in [folder / "data.yaml", folder.parent / "data.yaml"]:
            try:
                if cand.resolve().exists():
                    yaml_path = cand.resolve()
                    break
            except Exception:
                pass
                
        if not yaml_path:
            for root, dirs, files in os.walk(folder):
                if "data.yaml" in files:
                    yaml_path = Path(root) / "data.yaml"
                    break

        if yaml_path and yaml_path.exists():
            try:
                if yaml:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        ydata = yaml.safe_load(f)
                        if isinstance(ydata, dict) and "names" in ydata:
                            raw_names = ydata["names"]
                            if isinstance(raw_names, list):
                                self.class_names = {i: str(n) for i, n in enumerate(raw_names)}
                            elif isinstance(raw_names, dict):
                                self.class_names = {int(k): str(v) for k, v in raw_names.items()}
                self.dataset_type = "yolo"
            except Exception as ye:
                print(f"Warning: Failed to parse data.yaml: {ye}")

        if self.dataset_type != "yolo":
            # Quick check first 10 items for matching YOLO label files
            for item in self.items[:10]:
                lbl = item.find_label_path(folder)
                if lbl and lbl.exists():
                    self.dataset_type = "yolo"
                    break

        # Check and load saved session
        session_file = folder / SESSION_FILENAME
        if not session_file.exists():
            repo_annotations = Path(__file__).resolve().parent.parent.parent.parent / "annotations"
            f_lower = folder.name.lower()
            if f_lower == "plates" and (repo_annotations / "plates_session.json").exists():
                session_file = repo_annotations / "plates_session.json"
            elif f_lower in ("curated_lpr_dataset", "curated_dataset") and (repo_annotations / "curated_lpr_session.json").exists():
                session_file = repo_annotations / "curated_lpr_session.json"

        saved_index = 0
        if session_file.exists():
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    saved_items = data.get("items", {})
                    saved_index = data.get("current_index", 0)
                    
                    for item in self.items:
                        # Match by rel_path first, fallback to filename
                        if item.rel_path in saved_items:
                            item.from_dict(saved_items[item.rel_path])
                        elif item.filename in saved_items:
                            item.from_dict(saved_items[item.filename])
            except Exception as e:
                print(f"Warning: Failed to load session file: {e}")
                
        # Set current index
        if 0 <= saved_index < len(self.items):
            self.current_index = saved_index
        else:
            self.current_index = 0
            
        return True

    def save_session(self):
        """Saves current state into .lpr_session.json in folder."""
        if not self.folder_path:
            return
            
        session_file = self.folder_path / SESSION_FILENAME
        # Store key by rel_path to uniquely identify items across subfolders
        items_dict = {item.rel_path: item.to_dict() for item in self.items}
        data = {
            "version": "1.1",
            "standards": ["ISO 8601:2019", "ISO 3166-2:TH", "ISO/IEC 10646", "ISO/IEC 25012:2008"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "folder": str(self.folder_path),
            "current_index": self.current_index,
            "items": items_dict
        }
        
        try:
            temp_file = session_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_file.replace(session_file)
            
            # Mirror to annotations for Git version control sync if working inside canonical dataset folders
            repo_annotations = Path(__file__).resolve().parent.parent.parent.parent / "annotations"
            if repo_annotations.exists():
                try:
                    f_lower = self.folder_path.name.lower()
                    m_target = None
                    if f_lower == "plates":
                        m_target = repo_annotations / "plates_session.json"
                    elif f_lower in ("curated_lpr_dataset", "curated_dataset"):
                        m_target = repo_annotations / "curated_lpr_session.json"

                    if m_target:
                        m_temp = m_target.with_suffix(".tmp")
                        with open(m_temp, "w", encoding="utf-8") as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        m_temp.replace(m_target)
                except Exception as me:
                    print(f"Warning: Failed to mirror session: {me}")
        except Exception as e:
            print(f"Error saving session: {e}")

    def get_subfolders(self) -> List[str]:
        """Returns sorted unique list of subfolders."""
        subs = set()
        for item in self.items:
            if item.subfolder:
                subs.add(item.subfolder)
        return sorted(list(subs))

    def ensure_all_boxes_loaded(self):
        """Loads bounding boxes for all items if not already loaded."""
        for item in self.items:
            if not getattr(item, "_boxes_loaded", False):
                item.load_yolo_boxes(self.class_names, self.folder_path)

    def get_filtered_indices(self, subfolder: Optional[str] = None, class_filter: Optional[int] = None, status_filter: Optional[str] = None) -> List[int]:
        """Returns list of item indices matching subfolder, class filter, and status filter."""
        if class_filter is not None and class_filter != -1:
            self.ensure_all_boxes_loaded()

        if status_filter is None:
            status_filter = getattr(self, "active_status_filter", "all")

        indices = []
        for i, it in enumerate(self.items):
            # 1. Subfolder filter
            if subfolder and subfolder != "(All)":
                if subfolder == "(Root)":
                    if it.subfolder:
                        continue
                elif it.subfolder != subfolder:
                    continue
            
            # 2. Class filter (YOLO mode)
            if class_filter is not None and class_filter != -1:
                if class_filter == -2:
                    # Filter unannotated (no boxes)
                    if len(it.boxes) > 0:
                        continue
                else:
                    if not any(b.get("cls_id") == class_filter for b in it.boxes):
                        continue

            # 3. Status filter (OCR / Annotation mode)
            if status_filter and status_filter != "all":
                if status_filter == "not_plate":
                    if not (it.status == "not_plate" or (it.d3 == "+" and it.d4 == "+" and it.province == "+") or it.plate_type == "+"):
                        continue
                elif status_filter == "cannot_read":
                    if not (it.status == "cannot_read" or (it.d3 == "-" and it.d4 == "-" and it.province == "-") or it.plate_type == "-"):
                        continue
                elif status_filter == "pending":
                    is_special = (it.d3 in ("+", "-") and it.d4 in ("+", "-")) or it.plate_type in ("+", "-")
                    if it.status != "pending" or is_special:
                        continue
                elif status_filter == "approved":
                    if it.status != "approved":
                        continue
                elif status_filter == "skipped":
                    if it.status != "skipped":
                        continue
                elif status_filter == "dup":
                    if not (re.search(r'_dup(\d+)?(_|$|\.)', it.filename, re.IGNORECASE) or "_dup" in it.filename.lower()):
                        continue
                elif status_filter == "ai_diff":
                    comp = it.get_ai_comparison()
                    if not comp["has_ai"] or comp["all_match"]:
                        continue
                elif it.status != status_filter:
                    continue
                        
            indices.append(i)
        return indices

    def get_next_index(self, current_idx: int, subfolder: Optional[str] = None, class_filter: Optional[int] = None, status_filter: Optional[str] = None) -> Optional[int]:
        """Finds next index respecting subfolder, class filter, and status filter."""
        indices = self.get_filtered_indices(subfolder, class_filter, status_filter)
        if not indices:
            return None
        for i in indices:
            if i > current_idx:
                return i
        return indices[-1] # Stay at last if reached end

    def get_prev_index(self, current_idx: int, subfolder: Optional[str] = None, class_filter: Optional[int] = None, status_filter: Optional[str] = None) -> Optional[int]:
        """Finds previous index respecting subfolder, class filter, and status filter."""
        indices = self.get_filtered_indices(subfolder, class_filter, status_filter)
        if not indices:
            return None
        for i in reversed(indices):
            if i < current_idx:
                return i
        return indices[0] # Stay at first if reached start

    def get_yolo_stats(self, subfolder: Optional[str] = None) -> Dict[str, Any]:
        """Calculates dataset statistics for YOLO object detection."""
        self.ensure_all_boxes_loaded()
        indices = self.get_filtered_indices(subfolder=subfolder, status_filter="all")
        items = [self.items[i] for i in indices]
        total_images = len(items)
        images_with_boxes = 0
        class_counts = {cid: 0 for cid in self.class_names.keys()}
        total_boxes = 0
        for it in items:
            if it.boxes:
                images_with_boxes += 1
                for b in it.boxes:
                    cid = b.get("cls_id", 0)
                    class_counts[cid] = class_counts.get(cid, 0) + 1
                    total_boxes += 1
        return {
            "total_images": total_images,
            "images_with_boxes": images_with_boxes,
            "total_boxes": total_boxes,
            "class_counts": class_counts
        }

    def apply_ai_to_item(self, idx: int, replace_d3: bool = True, replace_d4: bool = True, replace_prov: bool = True) -> bool:
        """Applies AI predictions to the specified item fields."""
        if not (0 <= idx < len(self.items)):
            return False
        item = self.items[idx]
        if replace_d3 and item.ai_d3:
            item.d3 = item.ai_d3
        if replace_d4 and item.ai_d4:
            item.d4 = item.ai_d4
        if replace_prov and item.ai_province:
            item.province = item.ai_province
        return True

    def get_current_item(self) -> Optional[ImageItem]:
        if 0 <= self.current_index < len(self.items):
            item = self.items[self.current_index]
            if not getattr(item, "_boxes_loaded", False):
                item.load_yolo_boxes(self.class_names, self.folder_path)
            return item
        return None

    def rename_file(self, item: ImageItem, new_filename: str) -> bool:
        """Renames a file safely on disk and updates item in place."""
        if item.filename == new_filename:
            return True
            
        new_path = item.path.parent / new_filename
        
        # If target file exists and isn't the same file, handle collision safely
        if new_path.exists() and new_path != item.path:
            stem = new_path.stem
            ext = new_path.suffix
            counter = 1
            while new_path.exists():
                new_path = item.path.parent / f"{stem}_{counter}{ext}"
                counter += 1
            new_filename = new_path.name
            
        try:
            os.replace(item.path, new_path)
            item.path = new_path
            item.filename = new_filename
            item.stem = new_path.stem
            if self.folder_path:
                item.rel_path = str(new_path.relative_to(self.folder_path.resolve())).replace("\\", "/")
            return True
        except Exception as e:
            print(f"Error renaming file {item.path} to {new_path}: {e}")
            return False

    def custom_rename(self, index: int, new_filename: str) -> bool:
        """Custom manual renaming of an image file."""
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        new_filename = new_filename.strip()
        if not new_filename:
            return False
            
        # Append original extension if user didn't specify one
        if not Path(new_filename).suffix:
            new_filename += item.ext
            
        if self.rename_file(item, new_filename):
            item.parse_filename()
            self.save_session()
            return True
        return False

    def delete_item(self, index: int) -> bool:
        """Deletes file on disk and removes it from the session."""
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        try:
            if item.path.exists():
                item.path.unlink()
        except Exception as e:
            print(f"Error deleting file {item.path}: {e}")
            return False
            
        self.items.pop(index)
        if self.current_index >= len(self.items) and self.items:
            self.current_index = len(self.items) - 1
            
        self.save_session()
        return True

    def find_dup_items(self, subfolder: Optional[str] = None) -> List[int]:
        """Finds all indices of items containing '_dup' in their filename or stem."""
        dup_indices = []
        for idx, item in enumerate(self.items):
            if subfolder and subfolder != "(All)":
                if subfolder == "(Root)" and item.subfolder:
                    continue
                elif subfolder != "(Root)" and item.subfolder != subfolder:
                    continue
                    
            # Matches _dup, _dup1, _dup_2, etc. (case-insensitive)
            if re.search(r'_dup(\d+)?(_|$|\.)', item.filename, re.IGNORECASE) or "_dup" in item.filename.lower():
                dup_indices.append(idx)
                
        return dup_indices

    def remove_dup_images(self, subfolder: Optional[str] = None) -> int:
        """Batch removes all images matching '_dup' from disk and session."""
        dup_indices = self.find_dup_items(subfolder)
        if not dup_indices:
            return 0
            
        # Delete from highest index to lowest so pop indices remain valid
        count = 0
        for idx in sorted(dup_indices, reverse=True):
            if self.delete_item(idx):
                count += 1
                
        return count

    def mark_not_plate(self, index: int) -> bool:
        """Rename to {name}_+_+_+_+{ext} and mark status as not_plate"""
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        name = item.name if item.name else "unknown"
        ext = item.ext if item.ext else ".jpg"
        new_filename = f"{name}_+_+_+_+{ext}"
        
        if self.rename_file(item, new_filename):
            item.plate_type = "+"
            item.d3 = "+"
            item.d4 = "+"
            item.province = "+"
            item.status = "not_plate"
            self.save_session()
            return True
        return False

    def mark_cannot_read(self, index: int) -> bool:
        """Rename to {name}_-_-_-_-(ext} and mark status as cannot_read"""
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        name = item.name if item.name else "unknown"
        ext = item.ext if item.ext else ".jpg"
        new_filename = f"{name}_-_-_-_-{ext}"
        
        if self.rename_file(item, new_filename):
            item.plate_type = "-"
            item.d3 = "-"
            item.d4 = "-"
            item.province = "-"
            item.status = "cannot_read"
            self.save_session()
            return True
        return False

    def approve_and_save(self, index: int, new_name: str, new_plate_type: str, new_d3: str, new_d4: str, new_province: str) -> bool:
        """
        Saves changes to name, plate_type, d3, d4, province.
        If any changed, renames to {name}_{plate_type}_{d3}_{d4}_{province}.{ext}.
        Marks status as approved.
        """
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        
        new_name = new_name.strip() if new_name.strip() else item.name
        new_plate_type = new_plate_type.strip()
        new_d3 = new_d3.strip()
        new_d4 = new_d4.strip()
        new_province = new_province.strip()
        
        changed = (
            new_name != item.name or
            new_plate_type != item.plate_type or
            new_d3 != item.d3 or
            new_d4 != item.d4 or
            new_province != item.province
        )
        
        item.name = new_name
        item.plate_type = new_plate_type
        item.d3 = new_d3
        item.d4 = new_d4
        item.province = new_province
        
        if new_plate_type == "+" and new_d3 == "+" and new_d4 == "+" and new_province == "+":
            item.status = "not_plate"
        elif new_plate_type == "-" and new_d3 == "-" and new_d4 == "-" and new_province == "-":
            item.status = "cannot_read"
        else:
            item.status = "approved"
        
        if changed:
            name = item.name if item.name else "unknown"
            ext = item.ext if item.ext else ".jpg"
            new_filename = f"{name}_{new_plate_type}_{new_d3}_{new_d4}_{new_province}{ext}"
            self.rename_file(item, new_filename)
            
        if item.status == "approved" and getattr(self, "auto_staging", True):
            self.add_to_staging(item)

        self.save_session()
        return True

    def set_staging_dir(self, staging_path: Union[str, Path]):
        """Sets custom dataset staging directory."""
        self.staging_dir = Path(staging_path).resolve()

    def add_to_staging(self, item: ImageItem) -> Optional[Path]:
        """
        Adds/copies an approved item to the dataset staging directory waiting for dataset building.
        For license plate reading tasks:
            Folder named by {type}_{d3}_{d4}_{province}
            Destination: {staging_dir}/{type}_{d3}_{d4}_{province}/{filename}
        For detection tasks:
            Destination: {staging_dir}/detector/images/{filename} and labels/{stem}.txt
        """
        if item.status != "approved":
            return None
            
        if not item.path.exists():
            return None

        self.staging_dir.mkdir(parents=True, exist_ok=True)

        is_reading_task = (self.dataset_type == "ocr") or (
            bool(item.d3 and item.d4 and item.province and item.d3 not in ["+", "-"])
        )

        if is_reading_task:
            clean_type = item.plate_type.strip() if item.plate_type else "WhiteBlackCar"
            clean_d3 = normalize_thai_script(item.d3).strip()
            clean_d4 = item.d4.strip()
            clean_prov = normalize_thai_script(item.province).strip()

            raw_folder_name = f"{clean_type}_{clean_d3}_{clean_d4}_{clean_prov}"
            folder_name = re.sub(r'[<>:"/\\|?*]', '_', raw_folder_name).strip(' .')
            if not folder_name:
                folder_name = "unknown_plate"

            target_folder = self.staging_dir / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)

            dest_path = target_folder / item.path.name
            shutil.copy2(item.path, dest_path)
            setattr(item, "_staged_path", str(dest_path))
            self._update_staging_manifest(item, dest_path, is_reading=True)
            return dest_path
        else:
            det_img_dir = self.staging_dir / "detector" / "images"
            det_lbl_dir = self.staging_dir / "detector" / "labels"
            det_img_dir.mkdir(parents=True, exist_ok=True)
            det_lbl_dir.mkdir(parents=True, exist_ok=True)

            dest_path = det_img_dir / item.path.name
            shutil.copy2(item.path, dest_path)
            if item.label_path and item.label_path.exists():
                shutil.copy2(item.label_path, det_lbl_dir / item.label_path.name)
            setattr(item, "_staged_path", str(dest_path))
            self._update_staging_manifest(item, dest_path, is_reading=False)
            return dest_path

    def _update_staging_manifest(self, item: ImageItem, dest_path: Path, is_reading: bool):
        """Maintains .staging_manifest.json in staging directory."""
        manifest_file = self.staging_dir / ".staging_manifest.json"
        manifest = {}
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        items_dict = manifest.get("items", {})
        try:
            rel_dest = str(dest_path.relative_to(self.staging_dir)).replace("\\", "/")
        except ValueError:
            rel_dest = str(dest_path).replace("\\", "/")

        items_dict[item.filename] = {
            "source_path": str(item.path),
            "staged_path": rel_dest,
            "plate_type": item.plate_type,
            "d3": item.d3,
            "d4": item.d4,
            "province": item.province,
            "iso_3166_2": get_iso_code(item.province) if item.province else None,
            "is_reading_task": is_reading,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        manifest["version"] = "1.0"
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["total_staged"] = len(items_dict)
        manifest["items"] = items_dict

        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def export_all_approved_to_staging(self, subfolder: Optional[str] = None) -> Tuple[int, Path]:
        """Copies all approved items matching subfolder to dataset staging directory."""
        items = self.items
        if subfolder and subfolder != "(All)":
            if subfolder == "(Root)":
                items = [it for it in items if not it.subfolder]
            else:
                items = [it for it in items if it.subfolder == subfolder]

        approved_items = [it for it in items if it.status == "approved"]
        count = 0
        for it in approved_items:
            res = self.add_to_staging(it)
            if res:
                count += 1
        return count, self.staging_dir


    def mark_skipped(self, index: int):
        """Mark as skipped if pending."""
        if 0 <= index < len(self.items):
            item = self.items[index]
            if item.status == "pending":
                item.status = "skipped"
                self.save_session()

    def get_stats(self, subfolder: Optional[str] = None) -> Dict[str, int]:
        target_items = [self.items[i] for i in self.get_filtered_indices(subfolder=subfolder, status_filter="all")]
        total = len(target_items)
        approved = sum(1 for it in target_items if it.status == "approved")
        not_plate = sum(1 for it in target_items if it.status == "not_plate")
        cannot_read = sum(1 for it in target_items if it.status == "cannot_read")
        skipped = sum(1 for it in target_items if it.status == "skipped")
        pending = sum(1 for it in target_items if it.status == "pending")
        
        processed = approved + not_plate + cannot_read
        left = pending + skipped
        
        return {
            "total": total,
            "approved": approved,
            "not_plate": not_plate,
            "cannot_read": cannot_read,
            "skipped": skipped,
            "pending": pending,
            "processed": processed,
            "left": left
        }
