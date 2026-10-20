# Thai License Plate Dataset Relabeling Studio

An interactive, high-speed dataset annotation and relabeling application engineered specifically for Thai License Plate Recognition (Detection & OCR).

---

## 🌟 Key Features

### 1. 🎯 YOLO Detection Relabeler (Bounding Box Editor)
- **High-Performance Canvas**: Pan (Space/Alt+Drag or middle-click) and Zoom (mouse scroll wheel).
- **Interactive Box Manipulation**: Click-and-drag to draw new boxes, drag corners/handles to resize, drag centers to move.
- **Class Management**: Easily switch between classes (0: `license_plate`, 1: `character`, etc.) or add new classes.
- **YOLO Format Output**: Writes directly to normalized YOLO `.txt` files (`class_id xc yc w h`).
- **🤖 1-Click AI Auto-Detect**: Powered by YOLOv8 (`yolov8n.pt`) with adjustable confidence threshold to automatically pre-annotate images.
- **Hotkeys**:
  - `←` / `→` : Previous / Next image
  - `Ctrl + S` : Save labels
  - `A` : Auto-detect
  - `Del` / `Backspace` : Delete selected box
  - `0-9` : Switch active class

### 2. 🔤 License Plate OCR Relabeler (Transcription Studio)
- **High-Resolution Crop Preview**: View plate crops with live non-destructive enhancement filters:
  - Brightness & Contrast sliders
  - Grayscale conversion
  - Color Inversion (crucial for reading dark, nighttime, or washed-out plates)
- **Thai Plate Transcription**:
  - Plate number input (e.g. `1กข 8920`, `ขข 1234`) with instant typing and auto-focus.
  - **🇹🇭 Thai Quick-Pad**: Built-in on-screen pad with all 44 Thai consonants and numbers 0–9 for rapid input without switching system keyboards.
- **77 Thai Provinces**:
  - Searchable autocomplete input matching canonical Thai provinces.
  - Quick-click chips for the top 10 most common provinces (กรุงเทพมหานคร, เชียงใหม่, ชลบุรี, นนทบุรี, ปทุมธานี, ฯลฯ).
- **Plate Classification & Quality Tagging**:
  - `real_physical` (Standard vehicle plate)
  - `graphic_auction` (ป้ายประมูลแท้)
  - `red_plate` (ป้ายแดง)
  - `yellow_commercial` (แท็กซี่ / รถสาธารณะ)
  - `green_pickup` (รถกระบะ / บรรทุก)
  - `blue_van` (รถตู้โดยสาร)
  - `blurry_invalid` (Crop error / unreadable)
- **Flexible Export / Save Options**:
  - Direct Windows-safe file rename (e.g., `{plate_number}_{province}_{id}.jpg`)
  - Appends/updates `metadata.jsonl`
  - Optional routing into category subfolders
- **Rapid Navigation**:
  - Press `Enter` to save and advance immediately to the next image.

---

## 🚀 How to Run

### Method 1: Double-Click Launcher (Recommended)
Simply double-click:
```
D:\Projects-anti\lpr\run_relabel_gui.bat
```
This automatically selects the `pipeline/.venv` environment and opens the GUI in your default browser at `http://127.0.0.1:8501`.

### Method 2: Command Line
```powershell
cd D:\Projects-anti\lpr\tools\relabel_gui
python server.py 8501
```
