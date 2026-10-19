# Thai License Plate (LPR) Annotator & Reviewer GUI

A desktop application designed for inspecting, editing, and verifying Thai license plate dataset images.

## Filename Structure

The application reads and saves image files following the 5-element structure:
```text
{name}_{plate_type}_{d3}_{d4}_{province}.jpg
```
For example:
- `car01_white_black_1กก_1234_กรุงเทพมหานคร.jpg`
- `front_cam01_yellow_black_ทส_9999_กรุงเทพมหานคร.jpg`
- `gate_in_car4_yellow_blue_10_4567_ขอนแก่น.jpg`
- `sample_not_plate_+_+_+_+.jpg`
- `sample_unreadable_-_-_-_-.jpg`

### Supported Plate Types:
1. **White / Black** (`white_black`): Private passenger cars (≤ 7 seats) & motorcycles
2. **White / Blue** (`white_blue`): Small buses, vans, passenger vehicles > 7 seats
3. **White / Green** (`white_green`): Private pickup trucks
4. **Yellow / Black** (`yellow_black`): Commercial hire vehicles, taxis, motorcycle taxis
5. **Yellow / Blue** (`yellow_blue`): Fixed-route local buses or songthaews
6. **Yellow / Green** (`yellow_green`): Public tuk-tuks
7. **Green / White** (`green_white`): Tour, hotel, airport, or limousine services
8. **Red / Black** (`red_black`): Temporary or dealer registration plates
9. **Orange / Black** (`orange_black`): Agricultural vehicles, tractors, construction rollers
- **Special codes**: `+` (Not license plate) and `-` (Cannot read)

  - **Zoom In / Zoom Out**: Mouse wheel or `＋` / `－` buttons.
  - **Drag / Pan**: Click and drag to pan across zoomed-in images.
  - **Fit View** & **1:1** actual size buttons.
- **Plate Annotation Fields**:
  - **`d3`**: License plate category characters / letters (e.g. `1กข`, `ขก`).
  - **`d4`**: License plate numbers (e.g. `1234`, `9999`).
  - **`Province`**: Searchable dropdown autocomplete with all 77 Thai provinces (searchable in Thai or English) plus special codes:
    - `+` (Not License Plate)
    - `-` (Cannot Read)
- **Quick Action Buttons & Shortcuts**:
  - **Correct / Approve** (`Enter`): Marks as approved. If `d3`, `d4`, or `province` were edited, renames the file to `{name}_{d3}_{d4}_{province}.jpg`.
  - **Not Plate** (`Ctrl+N`): Saves and renames file to `{name}_+_+_+.jpg`.
  - **Cannot Read** (`Ctrl+D`): Saves and renames file to `{name}_-_-_-.jpg`.
  - **Skip** (`Ctrl+S`): Marks image as skipped for later review.
  - **Back / Next** (`Left Arrow` / `Right Arrow`): Navigate through images.
- **Status Dashboard**:
  - Shows counters for **Total**, **Left**, **Approved**, **Not Plate**, **Cannot Read**, and a progress bar with percentage.
- **Check Menu**:
  - Click **📋 Check Menu** to open a comprehensive status table.
  - Filter images by: **All**, **Skipped ⚠️**, **Pending ⏳**, **Approved ✅**, **Not Plate 🚫**, **Cannot Read ❓**.
  - Search by filename, characters, numbers, or province.
  - Double-click or select any row and click **Jump to Selected Image** to immediately jump to that image.

---

## Installation & Running

```bash
# Install dependencies
pip install PyQt6 Pillow

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
