"""
Session and File Management for LPR Dataset Annotation.
Handles directory scanning, filename parsing with plate_type, renaming, and state persistence.
Filename structure: {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SESSION_FILENAME = ".lpr_session.json"

class ImageItem:
    def __init__(self, full_path: Path):
        self.path = full_path.resolve()
        self.filename = self.path.name
        self.ext = self.path.suffix
        self.stem = self.path.stem
        
        # Parsed elements
        self.name: str = ""
        self.plate_type: str = "white_black"
        self.d3: str = ""
        self.d4: str = ""
        self.province: str = ""
        
        # Status: "pending", "approved", "not_plate", "cannot_read", "skipped"
        self.status: str = "pending"
        
        self.parse_filename()
        
    def parse_filename(self):
        """
        Parses filename structure: {name}_{plate_type}_{d3}_{d4}_{province}.{ext}
        Supports compound plate_types with underscores (e.g. white_black, yellow_blue),
        special codes (+, -), and allows underscores inside {name}.
        """
        from .plate_types import PLATE_TYPES_DATA, SPECIAL_PLATE_TYPES
        
        parts = self.stem.rsplit('_', 3)
        if len(parts) == 4:
            prefix, self.d3, self.d4, self.province = parts
            
            # Known plate type tokens (longest first)
            known_tokens = [pt["id"] for pt in PLATE_TYPES_DATA] + [s[0] for s in SPECIAL_PLATE_TYPES]
            known_tokens.sort(key=len, reverse=True)
            
            matched = False
            for token in known_tokens:
                suffix = f"_{token}"
                if prefix.endswith(suffix):
                    self.name = prefix[:-len(suffix)]
                    self.plate_type = token
                    matched = True
                    break
                    
            if not matched:
                # If prefix has an underscore, split it into name and plate_type
                if '_' in prefix:
                    p_name, p_type = prefix.rsplit('_', 1)
                    self.name = p_name
                    self.plate_type = p_type
                else:
                    self.name = prefix
                    self.plate_type = "white_black"
        elif len(parts) == 3:
            self.name = parts[0]
            self.plate_type = "white_black"
            self.d3 = parts[1]
            self.d4 = parts[2]
            self.province = ""
        elif len(parts) == 2:
            self.name = parts[0]
            self.plate_type = "white_black"
            self.d3 = parts[1]
            self.d4 = ""
            self.province = ""
        else:
            self.name = self.stem
            self.plate_type = "white_black"
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
        
    def load_folder(self, folder_str: str) -> bool:
        """Loads images in folder and restores session if available."""
        folder = Path(folder_str)
        if not folder.is_dir():
            return False
            
        self.folder_path = folder
        self.items = []
        
        # Collect all image files
        image_files = []
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(file)
                
        # Sort files naturally
        image_files.sort(key=lambda f: f.name.lower())
        
        for file in image_files:
            self.items.append(ImageItem(file))
            
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
                        if item.filename in saved_items:
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
        items_dict = {item.filename: item.to_dict() for item in self.items}
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

    def get_current_item(self) -> Optional[ImageItem]:
        if 0 <= self.current_index < len(self.items):
            return self.items[self.current_index]
        return None

    def rename_file(self, item: ImageItem, new_filename: str) -> bool:
        """Renames a file safely on disk and updates item."""
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
            return True
        except Exception as e:
            print(f"Error renaming file {item.path} to {new_path}: {e}")
            return False

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

    def approve_and_save(self, index: int, new_plate_type: str, new_d3: str, new_d4: str, new_province: str) -> bool:
        """
        Saves changes to plate_type, d3, d4, province.
        If any changed, renames to {name}_{plate_type}_{d3}_{d4}_{province}.{ext}.
        Marks status as approved.
        """
        if not (0 <= index < len(self.items)):
            return False
        item = self.items[index]
        
        new_plate_type = new_plate_type.strip()
        new_d3 = new_d3.strip()
        new_d4 = new_d4.strip()
        new_province = new_province.strip()
        
        changed = (
            new_plate_type != item.plate_type or
            new_d3 != item.d3 or
            new_d4 != item.d4 or
            new_province != item.province
        )
        
        item.plate_type = new_plate_type
        item.d3 = new_d3
        item.d4 = new_d4
        item.province = new_province
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

    def get_stats(self) -> Dict[str, int]:
        total = len(self.items)
        approved = sum(1 for it in self.items if it.status == "approved")
        not_plate = sum(1 for it in self.items if it.status == "not_plate")
        cannot_read = sum(1 for it in self.items if it.status == "cannot_read")
        skipped = sum(1 for it in self.items if it.status == "skipped")
        pending = sum(1 for it in self.items if it.status == "pending")
        
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
