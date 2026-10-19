# ⚡ OWOD License Plate Studio & YOLO Pruning Tool

A state-of-the-art GUI application to auto-label license plates with Open-World Object Detection (OWOD) such as YOLO-World, inspect model layers, freeze weights, slice architectures, prune nodes, and fine-tune YOLO models with GPU acceleration.

## How to Run:
```bash
# Inside tools/owod directory:
python main.py

# Or double-click:
run.bat (or root run_owod_studio.bat)
```

## Key Capabilities:
1. **📁 Multi-Root Media Discovery & Selection**:
   - **Multiple Roots Support**: Add folders from completely different roots, hard drives, or network locations (`➕ Add Folder from Any Root...`).
   - Recursively scans folders and subfolders for images (`.jpg`, `.png`, `.bmp`, `.webp`) and videos (`.mp4`, `.avi`, `.mov`, `.mkv`).
   - Tree view organizes each root separately with its own subfolders, items, and counts.
   - Individual checkbox selection per folder, subfolder, and media item.
   - Bulk controls: **Select All**, **Deselect All**, **Images Only**, **Videos Only**, **Invert**, and filename search filter.
   - Video sampling rate settings (sample every $N$ frames, max frames limit).
   - Live media preview pane with video duration and resolution metrics.
   - Collision-safe unique file identifiers to prevent collisions across multiple roots.
2. **🏷️ OWOD Auto-Labeling**:
   - Zero-shot Open-World Object Detection prompted with `"license plate"` (or custom class prompts).
   - Models: YOLO-World v2 (Small, Medium, Large, XLarge), SAM prompts, or custom `.pt` checkpoints.
   - Interactive Confidence and IoU / NMS threshold sliders with GPU (`cuda:0`) / CPU selector.
   - Real-time detection canvas rendering bounding boxes live as frames process.
   - Exports standard YOLO dataset with `images/train`, `images/val`, `labels/train`, `labels/val`, plate crops, and `data.yaml`.
3. **✂️ Model Surgery & Pruning**:
   - Supports YOLO base models (YOLO11, YOLOv8, YOLOv9, YOLOv10, or custom checkpoints).
   - Layer inspection table displaying index, name, module type, total params, trainable params, and sparsity.
   - **Layer Freezing**: Select individual layers or use quick presets (*Freeze Backbone 0-9*, *Freeze All Except Detect Head*).
   - **Slicing & Truncating**: Slice model network depth up to layer index $k$.
   - **Node Removal**: Remove specific layer nodes from the forward graph.
   - **Pruning**: Structured channel pruning and L1 unstructured weight pruning with custom reduction ratios (10% - 80%).
   - Export modified `.pt` checkpoints directly to disk.
4. **🚀 YOLO Fine-Tuning Studio**:
   - Connects the generated license plate dataset with the pruned/configured YOLO model.
   - Live training dashboard with epoch progress bar, Box Loss, Cls Loss, DFL Loss, and mAP50 / mAP50-95 cards.
   - Live console log stream with one-click access to trained weights.
