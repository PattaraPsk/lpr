"""
Media Scanner Module for OWOD Dataset Studio.
Supports adding multiple root folders from different drives/paths,
recursively scanning folders/subfolders for images and videos with selection tracking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import os

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}

@dataclass
class MediaItem:
    path: Path
    root_path: Path
    relative_path: str
    media_type: str  # 'image' or 'video'
    size_bytes: int
    is_selected: bool = True
    
    @property
    def filename(self) -> str:
        return self.path.name
        
    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def unique_identifier(self) -> str:
        """Returns safe collision-free base name combining root folder and relative path."""
        rel_clean = self.relative_path.replace("/", "_").replace("\\", "_")
        return f"{self.root_path.name}_{rel_clean}"


class MediaScanner:
    """Indexes and manages media across multiple root directories and their subfolders."""
    
    def __init__(self):
        self.root_dirs: List[Path] = []
        self.items: List[MediaItem] = []
        self.subfolders_by_root: Dict[str, Set[str]] = {}
        
    @property
    def root_dir(self) -> Optional[Path]:
        """Backwards compatibility: returns first root dir if available."""
        return self.root_dirs[0] if self.root_dirs else None

    def add_directory(self, root_path: str | Path) -> List[MediaItem]:
        """Scans and adds a root directory to the scanner without clearing existing ones."""
        resolved = Path(root_path).resolve()
        if not resolved.exists() or not resolved.is_dir():
            return []
            
        if resolved in self.root_dirs:
            return [it for it in self.items if it.root_path == resolved]
            
        self.root_dirs.append(resolved)
        root_key = str(resolved)
        self.subfolders_by_root[root_key] = set()
        
        new_items = []
        for root, dirs, files in os.walk(resolved):
            root_p = Path(root)
            try:
                rel_dir = root_p.relative_to(resolved).as_posix()
            except ValueError:
                rel_dir = "."
                
            if rel_dir != ".":
                self.subfolders_by_root[root_key].add(rel_dir)
                
            for file in files:
                ext = Path(file).suffix.lower()
                media_type = None
                if ext in IMAGE_EXTENSIONS:
                    media_type = 'image'
                elif ext in VIDEO_EXTENSIONS:
                    media_type = 'video'
                    
                if media_type:
                    file_path = root_p / file
                    try:
                        size = file_path.stat().st_size
                    except OSError:
                        size = 0
                    rel_path = file_path.relative_to(resolved).as_posix()
                    item = MediaItem(
                        path=file_path,
                        root_path=resolved,
                        relative_path=rel_path,
                        media_type=media_type,
                        size_bytes=size,
                        is_selected=True
                    )
                    self.items.append(item)
                    new_items.append(item)
                    
        return new_items

    def remove_directory(self, root_path: str | Path):
        """Removes a root directory and all its indexed media items."""
        resolved = Path(root_path).resolve()
        if resolved in self.root_dirs:
            self.root_dirs.remove(resolved)
        root_key = str(resolved)
        if root_key in self.subfolders_by_root:
            del self.subfolders_by_root[root_key]
        self.items = [it for it in self.items if it.root_path != resolved]

    def clear_all(self):
        """Clears all roots and items."""
        self.root_dirs.clear()
        self.items.clear()
        self.subfolders_by_root.clear()

    def scan_directory(self, root_path: str | Path) -> List[MediaItem]:
        """Clears and scans a single directory (backwards compatible)."""
        self.clear_all()
        return self.add_directory(root_path)

    def get_selected_items(self) -> List[MediaItem]:
        return [item for item in self.items if item.is_selected]

    def get_selected_images(self) -> List[MediaItem]:
        return [item for item in self.items if item.is_selected and item.media_type == 'image']

    def get_selected_videos(self) -> List[MediaItem]:
        return [item for item in self.items if item.is_selected and item.media_type == 'video']

    def set_all_selected(self, selected: bool):
        for item in self.items:
            item.is_selected = selected

    def set_root_selected(self, root_path: Path, selected: bool):
        for item in self.items:
            if item.root_path == root_path:
                item.is_selected = selected

    def set_type_selected(self, media_type: str, selected: bool):
        for item in self.items:
            if item.media_type == media_type:
                item.is_selected = selected

    def invert_selection(self):
        for item in self.items:
            item.is_selected = not item.is_selected

    def get_stats(self) -> Dict[str, int]:
        total_images = sum(1 for i in self.items if i.media_type == 'image')
        total_videos = sum(1 for i in self.items if i.media_type == 'video')
        sel_images = sum(1 for i in self.items if i.is_selected and i.media_type == 'image')
        sel_videos = sum(1 for i in self.items if i.is_selected and i.media_type == 'video')
        total_subfolders = sum(len(sfs) for sfs in self.subfolders_by_root.values())
        return {
            'total_roots': len(self.root_dirs),
            'total_files': len(self.items),
            'total_images': total_images,
            'total_videos': total_videos,
            'selected_files': sel_images + sel_videos,
            'selected_images': sel_images,
            'selected_videos': sel_videos,
            'subfolder_count': total_subfolders
        }
