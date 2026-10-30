# Thai License Plate Augmentor & Synthetic Studio

Windows desktop application built with PyQt6 for:
1. **Interactive Reference Plate Canvas**: Load real plate photos or templates and freely move, resize, rotate, and style each character.
2. **Synthetic Plate Generator**: DLT-compliant plate rendering with 41 valid Thai consonants, 77 provinces, and official colors.
3. **Multi-Stage Augmentation Pipeline**: 14 camera, motion, lighting, and weather distortion filters with live side-by-side previews.
4. **Batch Dataset Exporter**: Export training datasets with normalized YOLO bounding boxes (`LicensePlate=0`, `ThaiChar=1`, `Digit=2`), `data.yaml`, and metadata.

---

## 🎨 Tab 1: Interactive Canvas Editor (Move / Resize / Rotate)

### Features:
- **Load Any Reference Plate**:
  - Click **📁 Load Reference Image...** to load a real car plate photo, a blank template, or any crop.
  - Or choose from built-in DLT templates: *Blank White Car (1-line & 2-line)*, *Blank Motorcycle*, *Blank Red Dealer*, *Blank Yellow Commercial*.
- **Freely Manipulate Each Character**:
  - **Move**: Click and drag any character to any position on the plate (or use Arrow keys / Shift+Arrow for pixel-precision nudging).
  - **Rotate**:
    - Interactive Gizmo: Click and drag the yellow circular handle above the character to rotate it smoothly.
    - Inspector Slider / Spinbox: Set exact rotation from **-180° to +180°**.
  - **Resize & Scale**:
    - Interactive Gizmo: Drag the blue square handle at the bottom-right corner of the character. Hold `Shift` to preserve aspect ratio.
    - Inspector: Adjust font size (pt) or independent Scale X and Scale Y multipliers.
  - **Color & Opacity**: Change character color (black, white, green, blue, custom) and adjust opacity (10% to 100%).
  - **Duplicate & Delete**: Press `Delete` or `Backspace` to remove characters, or click **Duplicate**.
- **Auto-Place**: Type consonants (e.g. `1กข`), numbers (e.g. `1234`), and select a province, then click **Auto-Place All on Plate** to position all characters evenly.
- **Export & Pipeline Actions**:
  - **🚀 Send to Augmentation Pipeline**: Renders your custom plate arrangement and directly pipes it into the Live Augmentor tab.
  - **💾 Export Image + YOLO**: Saves the high-res composite image and computes exact rotated YOLO bounding boxes for all characters.
  - **📁 Save / Load Project (.json)**: Save your exact character layout to resume work anytime.

---

## ⚡ Tab 2: Live Augmentation Pipeline & Batch Generator

- **14 Configurable Augmentations**:
  - *Geometric*: Perspective warp (pitch/yaw), Rotation (-15° to +15°).
  - *Blur*: Gaussian focus blur, Directional motion blur.
  - *Noise*: Gaussian sensor noise, Salt & Pepper, JPEG compression degradation.
  - *Lighting*: Brightness/contrast shifts, color jitter, shadow bands, sunlight glare.
  - *Weather & Occlusion*: Rain droplets, dirt/mud splatters.
  - *Resolution*: Low-resolution camera downsampling & reconstruction.
- **Presets**: `None`, `Light`, `Medium`, `Heavy`, `Extreme`.
- **Live Preview Viewport**: Side-by-side comparison of base plate vs. augmented output, with optional YOLO bounding box visualization.
- **Batch Generator**: Generate up to 50,000 augmented synthetic samples with train/val splits and `data.yaml`.

---

## 📁 Tab 3: Batch Augment Existing Dataset

- Load a folder of existing cropped plates.
- Generate $N$ augmented variations per plate.
- Bulk export to target directory.

---

## Launching the App

Double-click:
```
run_plate_augmentor.bat
```
Or from PowerShell:
```powershell
cd d:\Projects-anti\lpr
.\.venv\Scripts\activate.bat
python tools\PlateAugmentor\main.py
```
