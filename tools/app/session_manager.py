"""
Session and File Management for LPR Dataset Annotation.
Supports recursive subfolders, grouping by subfolders, custom rename,
single-image delete, and batch removing '_dup' duplicate images.
Filename structure: {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

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
        
        # Status: "pending", "approved", "not_plate", "cannot_read", "skipped"
        self.status: str = "pending"
        
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "rel_path": self.rel_path,
            "subfolder": self.subfolder,
            "name": self.name,
            "plate_type": self.plate_type,
            "d3": self.d3,
            "d4": self.d4,
            "province": self.province,
            "status": self.status
        }
        
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


class SessionManager:
    def __init__(self):
        self.folder_path: Optional[Path] = None
        self.items: List[ImageItem] = []
        self.current_index: int = 0
        self.active_subfolder: Optional[str] = None # None means All
        
    def load_folder(self, folder_str: str) -> bool:
        """Recursively loads images in folder (including all subfolders) and restores session."""
        folder = Path(folder_str)
        if not folder.is_dir():
            return False
            
        self.folder_path = folder
        self.items = []
        self.active_subfolder = None
        
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
            
        # Check and load saved session
        session_file = folder / SESSION_FILENAME
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
            "folder": str(self.folder_path),
            "current_index": self.current_index,
            "items": items_dict
        }
        
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving session: {e}")

    def get_subfolders(self) -> List[str]:
        """Returns sorted unique list of subfolders."""
        subs = set()
        for item in self.items:
            if item.subfolder:
                subs.add(item.subfolder)
        return sorted(list(subs))

    def get_filtered_indices(self, subfolder: Optional[str] = None) -> List[int]:
        """Returns list of item indices matching subfolder filter (None or '' means All)."""
        if not subfolder or subfolder == "(All)":
            return list(range(len(self.items)))
            
        if subfolder == "(Root)":
            return [i for i, it in enumerate(self.items) if not it.subfolder]
            
        return [i for i, it in enumerate(self.items) if it.subfolder == subfolder]

    def get_next_index(self, current_idx: int, subfolder: Optional[str] = None) -> Optional[int]:
        """Finds next index respecting subfolder filter."""
        indices = self.get_filtered_indices(subfolder)
        if not indices:
            return None
        for i in indices:
            if i > current_idx:
                return i
        return indices[-1] # Stay at last if reached end

    def get_prev_index(self, current_idx: int, subfolder: Optional[str] = None) -> Optional[int]:
        """Finds previous index respecting subfolder filter."""
        indices = self.get_filtered_indices(subfolder)
        if not indices:
            return None
        for i in reversed(indices):
            if i < current_idx:
                return i
        return indices[0] # Stay at first if reached start

    def get_current_item(self) -> Optional[ImageItem]:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
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
            
        self.save_session()
        return True


    def mark_skipped(self, index: int):
        """Mark as skipped if pending."""
        if 0 <= index < len(self.items):
            item = self.items[index]
            if item.status == "pending":
                item.status = "skipped"
                self.save_session()

    def get_stats(self, subfolder: Optional[str] = None) -> Dict[str, int]:
        target_items = [self.items[i] for i in self.get_filtered_indices(subfolder)]
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
