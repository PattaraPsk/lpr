# Thai License Plate (LPR) Annotator & Reviewer GUI

A desktop application designed for inspecting, editing, verifying, and AI-assisted reading of Thai license plate dataset images.

## Filename Structure

The application reads and saves image files following the 5-element structure:
```text
{name}_{plate_type}_{d3}_{d4}_{province}.jpg
```
For example:
- `car01_WhiteBlackCar_1กก_1234_กรุงเทพมหานคร.jpg`
- `motor02_WhiteBlackMotor_1กข_999_เชียงใหม่.jpg`
- `front_cam01_YellowBlack_ทส_9999_กรุงเทพมหานคร.jpg`
- `gate_in_car4_YellowBlue_10_4567_ขอนแก่น.jpg`
- `sample_not_plate_+_+_+_+.jpg`
- `sample_unreadable_-_-_-_-.jpg`

### Supported Plate Types (Uppercase Separation):
1. **White / Black - Car** (`WhiteBlackCar`): Private passenger cars (≤ 7 seats)
2. **White / Black - Motor** (`WhiteBlackMotor`): Standard private motorcycles
3. **White / Blue** (`WhiteBlue`): Small buses, vans, passenger vehicles > 7 seats
4. **White / Green** (`WhiteGreen`): Private pickup trucks
5. **Yellow / Black** (`YellowBlack`): Commercial hire vehicles, taxis, motorcycle taxis
6. **Yellow / Blue** (`YellowBlue`): Fixed-route local buses or songthaews
7. **Yellow / Green** (`YellowGreen`): Public tuk-tuks
8. **Green / White** (`GreenWhite`): Tour, hotel, airport, or limousine services
9. **Red / Black** (`RedBlack`): Temporary or dealer registration plates
10. **Orange / Black** (`OrangeBlack`): Agricultural vehicles, tractors, construction rollers
- **Special codes**: `+` (Not license plate) and `-` (Cannot read)

---

## Key Features

- **Subfolder Support & Grouping**:
  - Recursively scans all subfolders when opening a folder.
  - **Group Subfolder Dropdown**: Filter and review images folder by folder or view `All Subfolders`.
  - Displays the active subfolder path for each image (`📂 Folder: camera1/batch1`).
  - Session state persists image records by their relative path across subfolders.

- **Interactive Image Viewer**:
  - **Zoom In / Zoom Out**: Mouse wheel or `＋` / `－` buttons.
  - **Drag / Pan**: Click and drag to pan across zoomed-in images.
  - **Fit View** & **1:1** actual size buttons.

- **Plate Annotation Fields**:
  - **`Base Name`**: Editable image prefix/identifier.
  - **`Plate Type`**: Dropdown with visual color preview and badge (`CAR`, `MC`, `VAN`, `TAXI`, etc.).
  - **`d3`**: License plate category characters / letters (e.g. `1กข`, `ขก`).
  - **`d4`**: License plate numbers (e.g. `1234`, `9999`).
  - **`Province`**: Searchable dropdown autocomplete with all 77 Thai provinces plus `+` and `-`.

- **🤖 AI Plate Reader (`Edge AI Reader`) & Live Comparison**:
  - **Integrated `Edge AI Reader` Model**: Authentic Pure PyTorch OCR architecture (RCNN + BiLSTM + Attention Decoder) loading `models/reader/Edge AI Reader/best_norm_ED.pth`.
  - **Auto-Read**: Automatically runs background inference when navigating through images (toggleable).
  - **Live Comparison Grid**: Shows side-by-side comparison between Current values and AI Predicted values for **d3**, **d4**, and **Province** with color-coded diff badges (`Matched ✅`, `Differences ⚠️`).
  - **Selective & Individual Replacement**:
    - **Single Field Buttons**: `⬅ d3`, `⬅ d4`, `⬅ Prov` to replace only that field.
    - **Replace Selected**: Checkboxes next to each field to replace any combination of fields.
  - **Bulk Replacement**:
    - **`⚡ Replace All (Ctrl+R)`**: Replaces `d3`, `d4`, and `Province` simultaneously with AI predictions.
  - **Batch AI Read (`🤖 AI Read All`)**: Runs Edge AI Reader across all images in the active folder/subfolder with a live progress dialog.

- **Management & Action Buttons**:
  - **Correct / Approve** (`Enter`): Saves and approves the current image. Renames file according to the filled boxes (e.g. `{name}_{plate_type}_{d3}_{d4}_{province}.jpg`, `{name}_+_+_+_+.jpg`, or `{name}_-_-_-_-.jpg`) and advances to the next image.
  - **Not Plate** (`Ctrl+N`): Fills all plate boxes with `+`. Save and advance by clicking **Approve**.
  - **Cannot Read** (`Ctrl+D`): Fills all plate boxes with `-`. Save and advance by clicking **Approve**.
  - **Skip** (`Ctrl+S`): Marks image as skipped for later review.
  - **Rename File** (`F2`): Custom manual rename dialog for the entire filename.
  - **Delete Image** (`Delete` / `Ctrl+Delete`): Permanently removes unwanted image from disk with confirmation.
  - **Remove _dup Images** (`🧹 Remove _dup`): Scans and batch removes all images matching `_dup` from disk.
  - **Back / Next** (`Left Arrow` / `Right Arrow`): Navigates through images within current subfolder filter.

- **Status Dashboard**:
  - Shows counters for **Total**, **Left**, **Approved**, **Not Plate**, **Cannot Read**, and a progress bar with percentage.

- **Check Menu**:
  - Click **📋 Check Menu** to open a comprehensive 9-column status table including **Subfolder** and **Plate Type**.
  - Filter images by: **All**, **Skipped ⚠️**, **Pending ⏳**, **Approved ✅**, **Not Plate 🚫**, **Cannot Read ❓**, **Duplicates (_dup) 🧹**, and **AI Diff ⚠️**.
  - Subfolder dropdown filter inside the check menu.
  - **Replace Selected with AI** button directly from the table.
  - **Delete Image** button directly from the table.
  - Double-click or select any row and click **Jump to Selected Image** to immediately jump to that image.

---

## Installation & Running

```bash
# Install dependencies
pip install PyQt6 Pillow torch torchvision

# Run application
python main.py
```

Or double-click `run.bat` on Windows.

### Generating Sample Test Images
Run:
```bash
python create_sample_data.py
```
This generates mock sample images in `sample_images/` for instant testing.

---

## ⚡ OWOD License Plate Studio & YOLO Pruning Tool
For the OWOD Auto-Labeler and YOLO Model Pruning studio, see [`../owod/README.md`](../owod/README.md).
You can launch it directly from the Thai LPR Annotator header bar by clicking the **⚡ OWOD Studio** button, or run:
```bash
cd tools/owod
python main.py
```
