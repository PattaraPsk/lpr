"""
Media Discovery & Selection Tab.
Supports selecting and managing multiple root folders from different paths/drives,
hierarchical checkbox trees per root, individual image/video toggles, and video frame sampling.
"""

from pathlib import Path
from typing import Dict, List, Optional
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QLineEdit, QFileDialog, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QSpinBox, QFormLayout, QFrame, QHeaderView, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon
import cv2

from ..core.media_scanner import MediaScanner, MediaItem, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from ..core.video_processor import VideoProcessor

class MediaSelectorTab(QWidget):
    """Tab 1: Multi-Root Media Discovery, Tree Selection, and Sampling Configuration."""
    
    selection_changed = pyqtSignal(dict)  # emits stats dict
    
    def __init__(self, scanner: MediaScanner):
        super().__init__()
        self.scanner = scanner
        self._tree_updating = False
        self._init_ui()
        
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # 1. Multi-Root Folder Action Bar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        
        lbl_folder = QLabel("📂 Source Roots:")
        lbl_folder.setStyleSheet("font-weight: bold; color: #38bdf8;")
        top_bar.addWidget(lbl_folder)
        
        self.btn_add_folder = QPushButton("➕ Add Folder from Any Root...")
        self.btn_add_folder.setObjectName("btnSuccess")
        self.btn_add_folder.setToolTip("Select and add a folder from any local/network drive")
        self.btn_add_folder.clicked.connect(self._on_add_folder_clicked)
        top_bar.addWidget(self.btn_add_folder)
        
        self.btn_remove_folder = QPushButton("➖ Remove Selected Root")
        self.btn_remove_folder.setObjectName("btnSecondary")
        self.btn_remove_folder.setToolTip("Remove the currently selected root folder and its contents")
        self.btn_remove_folder.clicked.connect(self._on_remove_folder_clicked)
        top_bar.addWidget(self.btn_remove_folder)
        
        self.btn_clear_all = QPushButton("🗑️ Clear All Roots")
        self.btn_clear_all.setObjectName("btnDanger")
        self.btn_clear_all.clicked.connect(self._on_clear_all_clicked)
        top_bar.addWidget(self.btn_clear_all)
        
        self.btn_load_default = QPushButton("⚡ Load 'dataset/' Folder")
        self.btn_load_default.setStyleSheet("background-color: #6366f1; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px;")
        self.btn_load_default.setToolTip("Quick-load all images and videos from the workspace 'dataset/' directory")
        self.btn_load_default.clicked.connect(self._on_load_default_clicked)
        top_bar.addWidget(self.btn_load_default)
        
        top_bar.addStretch()
        
        self.lbl_roots_badge = QLabel("0 Folders Loaded")
        self.lbl_roots_badge.setStyleSheet("background-color: #1e293b; color: #38bdf8; font-weight: bold; padding: 4px 10px; border-radius: 4px; border: 1px solid #334155;")
        top_bar.addWidget(self.lbl_roots_badge)
        
        layout.addLayout(top_bar)
        
        # 2. Main Central Splitter (Left: Tree + Controls, Right: Preview + Video Settings)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)
        
        # Left Container
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        # Selection Action Bar
        action_bar = QHBoxLayout()
        action_bar.setSpacing(6)
        
        self.btn_select_all = QPushButton("☑️ Select All")
        self.btn_select_all.clicked.connect(lambda: self._bulk_select(True))
        action_bar.addWidget(self.btn_select_all)
        
        self.btn_deselect_all = QPushButton("⬜ Deselect All")
        self.btn_deselect_all.setObjectName("btnSecondary")
        self.btn_deselect_all.clicked.connect(lambda: self._bulk_select(False))
        action_bar.addWidget(self.btn_deselect_all)
        
        self.btn_images_only = QPushButton("🖼️ Images Only")
        self.btn_images_only.setObjectName("btnSecondary")
        self.btn_images_only.clicked.connect(lambda: self._filter_type_select('image'))
        action_bar.addWidget(self.btn_images_only)
        
        self.btn_videos_only = QPushButton("🎬 Videos Only")
        self.btn_videos_only.setObjectName("btnSecondary")
        self.btn_videos_only.clicked.connect(lambda: self._filter_type_select('video'))
        action_bar.addWidget(self.btn_videos_only)
        
        self.btn_invert = QPushButton("🔄 Invert")
        self.btn_invert.setObjectName("btnSecondary")
        self.btn_invert.clicked.connect(self._invert_selection)
        action_bar.addWidget(self.btn_invert)
        
        action_bar.addStretch()
        
        # Filter search bar
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("🔍 Filter filename...")
        self.edit_filter.textChanged.connect(self._filter_tree_items)
        action_bar.addWidget(self.edit_filter)
        
        left_layout.addLayout(action_bar)
        
        # Media Tree Widget
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Media Name / Folder Hierarchy", "Type", "Size", "Full Path"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        self.tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.tree, 1)
        
        splitter.addWidget(left_widget)
        
        # Right Container (Video Settings & Media Preview)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Video Sampling Settings Group
        grp_video = QGroupBox("🎬 Video Frame Sampling Settings")
        form_v = QFormLayout(grp_video)
        form_v.setContentsMargins(12, 14, 12, 12)
        form_v.setSpacing(8)
        
        self.spin_frame_step = QSpinBox()
        self.spin_frame_step.setRange(1, 600)
        self.spin_frame_step.setValue(15)
        self.spin_frame_step.setToolTip("Extract 1 frame every N frames (e.g. 15 = ~2 frames/sec at 30fps)")
        form_v.addRow("Sample Every N Frames:", self.spin_frame_step)
        
        self.spin_max_frames = QSpinBox()
        self.spin_max_frames.setRange(0, 100000)
        self.spin_max_frames.setValue(120)
        self.spin_max_frames.setSpecialValueText("Unlimited (0)")
        self.spin_max_frames.setToolTip("Maximum frames extracted per video (0 = unlimited)")
        form_v.addRow("Max Frames Per Video:", self.spin_max_frames)
        
        right_layout.addWidget(grp_video)
        
        # Preview Pane
        grp_preview = QGroupBox("👁️ Media Preview")
        v_prev = QVBoxLayout(grp_preview)
        v_prev.setContentsMargins(10, 14, 10, 10)
        v_prev.setSpacing(6)
        
        self.lbl_preview = QLabel("Select an image or video to preview")
        self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_preview.setStyleSheet("background-color: #0b1120; border-radius: 6px; color: #64748b; font-style: italic;")
        self.lbl_preview.setMinimumHeight(240)
        v_prev.addWidget(self.lbl_preview, 1)
        
        self.lbl_item_info = QLabel("No item selected")
        self.lbl_item_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_prev.addWidget(self.lbl_item_info)
        
        right_layout.addWidget(grp_preview, 1)
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)
        
        # 3. Bottom Summary Status Bar
        self.lbl_summary = QLabel("📊 Roots: 0 | Total Files: 0 | Selected: 0 (Images: 0, Videos: 0) | Subfolders: 0")
        self.lbl_summary.setStyleSheet("background-color: #1e293b; padding: 6px 12px; border-radius: 6px; font-weight: bold; color: #a78bfa;")
        layout.addWidget(self.lbl_summary)

    def _on_add_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Add Folder from Any Root/Drive", str(Path.cwd()))
        if folder:
            self.add_directory(folder)

    def _on_load_default_clicked(self):
        candidates = [
            Path("dataset").resolve(),
            Path("../dataset").resolve(),
            Path("../../dataset").resolve()
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                self.add_directory(str(c))
                return
        QMessageBox.information(self, "Dataset Not Found", "Could not locate 'dataset/' directory in the workspace.")

    def add_directory(self, folder_path: str):
        self.scanner.add_directory(folder_path)
        self._populate_tree()
        self._update_stats()

    def _on_remove_folder_clicked(self):
        selected_items = self.tree.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Select Root Folder", "Please click on a root folder item in the tree to remove it.")
            return
            
        target_item = selected_items[0]
        # Traverse up to root node
        while target_item.parent() is not None:
            target_item = target_item.parent()
            
        root_path_str = target_item.data(0, Qt.ItemDataRole.UserRole + 1)
        if root_path_str:
            self.scanner.remove_directory(root_path_str)
            self._populate_tree()
            self._update_stats()

    def _on_clear_all_clicked(self):
        if not self.scanner.root_dirs:
            return
        confirm = QMessageBox.question(
            self, "Clear All Roots",
            "Are you sure you want to clear all loaded source folders?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.scanner.clear_all()
            self._populate_tree()
            self._update_stats()

    def _populate_tree(self):
        self._tree_updating = True
        self.tree.clear()
        
        if not self.scanner.root_dirs:
            self._tree_updating = False
            self.lbl_roots_badge.setText("0 Folders Loaded")
            return
            
        self.lbl_roots_badge.setText(f"{len(self.scanner.root_dirs)} Folders Loaded")
        
        for r_idx, root_dir in enumerate(self.scanner.root_dirs):
            root_items = [it for it in self.scanner.items if it.root_path == root_dir]
            root_subfolders = self.scanner.subfolders_by_root.get(str(root_dir), set())
            
            # Root Node
            root_node = QTreeWidgetItem(self.tree)
            root_node.setText(0, f"📁 [{r_idx+1}] {root_dir.name} ({len(root_items)} items)")
            root_node.setText(1, "Root Folder")
            root_node.setText(3, str(root_dir))
            root_node.setFlags(root_node.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            root_node.setCheckState(0, Qt.CheckState.Checked)
            root_node.setData(0, Qt.ItemDataRole.UserRole + 1, str(root_dir))
            
            subfolder_nodes: Dict[str, QTreeWidgetItem] = {".": root_node}
            
            # Subfolder nodes
            for subf in sorted(list(root_subfolders)):
                parts = subf.split("/")
                parent_key = "."
                for i, part in enumerate(parts):
                    curr_key = "/".join(parts[:i+1])
                    if curr_key not in subfolder_nodes:
                        parent_node = subfolder_nodes[parent_key]
                        sub_node = QTreeWidgetItem(parent_node)
                        sub_node.setText(0, f"📁 {part}")
                        sub_node.setText(1, "Subfolder")
                        sub_node.setText(3, str(root_dir / curr_key))
                        sub_node.setFlags(sub_node.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
                        sub_node.setCheckState(0, Qt.CheckState.Checked)
                        subfolder_nodes[curr_key] = sub_node
                    parent_key = curr_key
                    
            # Media items under this root
            for media in root_items:
                rel_dir = Path(media.relative_path).parent.as_posix()
                parent_node = subfolder_nodes.get(rel_dir, root_node)
                
                leaf = QTreeWidgetItem(parent_node)
                icon_str = "🖼️" if media.media_type == 'image' else "🎬"
                leaf.setText(0, f"{icon_str} {media.filename}")
                leaf.setText(1, media.media_type.capitalize())
                
                size_kb = media.size_bytes / 1024
                size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{(size_kb / 1024):.1f} MB"
                leaf.setText(2, size_str)
                leaf.setText(3, str(media.path))
                
                leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                leaf.setCheckState(0, Qt.CheckState.Checked if media.is_selected else Qt.CheckState.Unchecked)
                leaf.setData(0, Qt.ItemDataRole.UserRole, media)
                
        self.tree.expandAll()
        self._tree_updating = False

    def _on_tree_item_changed(self, item: QTreeWidgetItem, column: int):
        if self._tree_updating or column != 0:
            return
            
        media: Optional[MediaItem] = item.data(0, Qt.ItemDataRole.UserRole)
        check_state = item.checkState(0)
        
        if media is not None:
            # Leaf media item
            media.is_selected = (check_state == Qt.CheckState.Checked)
        else:
            # Folder node: propagate to children
            self._propagate_folder_check(item, check_state == Qt.CheckState.Checked)
            
        self._update_stats()

    def _propagate_folder_check(self, folder_item: QTreeWidgetItem, is_checked: bool):
        self._tree_updating = True
        state = Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked
        
        def recurse(node: QTreeWidgetItem):
            for i in range(node.childCount()):
                child = node.child(i)
                child.setCheckState(0, state)
                m = child.data(0, Qt.ItemDataRole.UserRole)
                if m is not None:
                    m.is_selected = is_checked
                recurse(child)
                
        recurse(folder_item)
        self._tree_updating = False

    def _bulk_select(self, select: bool):
        self._tree_updating = True
        self.scanner.set_all_selected(select)
        state = Qt.CheckState.Checked if select else Qt.CheckState.Unchecked
        
        def recurse(node: QTreeWidgetItem):
            node.setCheckState(0, state)
            for i in range(node.childCount()):
                recurse(node.child(i))
                
        for i in range(self.tree.topLevelItemCount()):
            recurse(self.tree.topLevelItem(i))
            
        self._tree_updating = False
        self._update_stats()

    def _filter_type_select(self, media_type: str):
        self._tree_updating = True
        for media in self.scanner.items:
            media.is_selected = (media.media_type == media_type)
            
        def recurse(node: QTreeWidgetItem):
            m = node.data(0, Qt.ItemDataRole.UserRole)
            if m is not None:
                state = Qt.CheckState.Checked if m.is_selected else Qt.CheckState.Unchecked
                node.setCheckState(0, state)
            for i in range(node.childCount()):
                recurse(node.child(i))
                
        for i in range(self.tree.topLevelItemCount()):
            recurse(self.tree.topLevelItem(i))
            
        self._tree_updating = False
        self._update_stats()

    def _invert_selection(self):
        self._tree_updating = True
        self.scanner.invert_selection()
        
        def recurse(node: QTreeWidgetItem):
            m = node.data(0, Qt.ItemDataRole.UserRole)
            if m is not None:
                state = Qt.CheckState.Checked if m.is_selected else Qt.CheckState.Unchecked
                node.setCheckState(0, state)
            for i in range(node.childCount()):
                recurse(node.child(i))
                
        for i in range(self.tree.topLevelItemCount()):
            recurse(self.tree.topLevelItem(i))
            
        self._tree_updating = False
        self._update_stats()

    def _filter_tree_items(self, text: str):
        query = text.lower().strip()
        
        def filter_node(node: QTreeWidgetItem) -> bool:
            node_text = node.text(0).lower()
            any_child_visible = False
            for i in range(node.childCount()):
                child_visible = filter_node(node.child(i))
                if child_visible:
                    any_child_visible = True
                    
            is_match = (query in node_text) or any_child_visible or (not query)
            node.setHidden(not is_match)
            return is_match
            
        for i in range(self.tree.topLevelItemCount()):
            filter_node(self.tree.topLevelItem(i))

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        media: Optional[MediaItem] = item.data(0, Qt.ItemDataRole.UserRole)
        if media is None:
            self.lbl_item_info.setText(f"Folder: {item.text(3)}")
            return
            
        self.lbl_item_info.setText(
            f"File: {media.filename} | Root: {media.root_path.name} | "
            f"Type: {media.media_type.capitalize()} | {media.size_bytes / 1024:.1f} KB"
        )
        
        if media.media_type == 'image':
            pixmap = QPixmap(str(media.path))
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.lbl_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lbl_preview.setPixmap(scaled)
        elif media.media_type == 'video':
            info = VideoProcessor.get_video_info(media.path)
            cap = cv2.VideoCapture(str(media.path))
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(
                    self.lbl_preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lbl_preview.setPixmap(pix)
            self.lbl_item_info.setText(
                f"🎬 Video: {media.filename} ({media.root_path.name}) | {info['width']}x{info['height']} | "
                f"{info['fps']} FPS | {info['duration_sec']}s ({info['frames']} frames)"
            )

    def _update_stats(self):
        stats = self.scanner.get_stats()
        self.lbl_summary.setText(
            f"📊 Roots: {stats['total_roots']} | Total Files: {stats['total_files']} | "
            f"Selected: {stats['selected_files']} (Images: {stats['selected_images']}, Videos: {stats['selected_videos']}) | "
            f"Subfolders: {stats['subfolder_count']}"
        )
        self.selection_changed.emit(stats)

    def get_sampling_config(self) -> dict:
        return {
            'frame_step': self.spin_frame_step.value(),
            'max_frames': self.spin_max_frames.value() if self.spin_max_frames.value() > 0 else None
        }
