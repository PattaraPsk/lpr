import os
import glob
import cv2
import random
from pathlib import Path

DATASET_DIR = Path("dataset/yolo_dlt_lpr_dataset")
IMG_DIR = DATASET_DIR / "images" / "train"
LBL_DIR = DATASET_DIR / "labels" / "train"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "runs/visualize"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["car_plate", "motorcycle_plate", "truck_plate", "bus_plate", "license_plate"]
COLORS = {
    0: (0, 220, 0),     # Car: Green
    1: (0, 220, 255),   # Motorcycle: Yellow
    2: (255, 120, 0),   # Truck: Blue/Orange
    3: (255, 0, 200),   # Bus: Magenta
    4: (200, 200, 200)  # General: Gray
}

def draw_boxes(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if line.strip()]
            
        for line in lines:
            cls_id = int(line[0])
            cx = float(line[1]) * w
            cy = float(line[2]) * h
            bw = float(line[3]) * w
            bh = float(line[4]) * h
            
            x1 = int(cx - bw / 2.0)
            y1 = int(cy - bh / 2.0)
            x2 = int(cx + bw / 2.0)
            y2 = int(cy + bh / 2.0)
            
            color = COLORS.get(cls_id, (0, 255, 0))
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls_{cls_id}"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # Draw label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(0, y1)), color, -1)
            cv2.putText(img, label, (x1 + 3, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    return img

def main():
    random.seed(42)
    # Pick representative samples from each source prefix
    prefixes = ['lpr', 'ann', 'surv', 'extra']
    selected_samples = []
    
    for prefix in prefixes:
        matching = sorted(glob.glob(str(IMG_DIR / f"{prefix}_*.jpg")))
        if matching:
            # Pick 2 samples from each prefix
            step = max(1, len(matching) // 3)
            for s in matching[::step][:2]:
                selected_samples.append(Path(s))
                
    # Also find at least 1 motorcycle_plate and 1 truck_plate specifically
    mc_found = None
    for lbl_file in glob.glob(str(LBL_DIR / "*.txt")):
        with open(lbl_file, "r") as f:
            for l in f:
                if l.startswith("1 "):
                    mc_found = Path(lbl_file).stem
                    break
        if mc_found:
            break
            
    if mc_found:
        mc_img = IMG_DIR / f"{mc_found}.jpg"
        if mc_img.exists() and mc_img not in selected_samples:
            selected_samples.append(mc_img)

    print(f"Rendering {len(selected_samples)} sample visualizations...")
    rendered_imgs = []
    for s_path in selected_samples[:6]:
        lbl_path = LBL_DIR / f"{s_path.stem}.txt"
        vis = draw_boxes(s_path, lbl_path)
        if vis is not None:
            # Resize to standard preview size (640x360)
            vis_resized = cv2.resize(vis, (640, 360))
            rendered_imgs.append(vis_resized)
            # Also save individual artifact
            out_indiv = ARTIFACT_DIR / f"dlt_sample_{s_path.stem}.jpg"
            cv2.imwrite(str(out_indiv), vis)

    # Build 2x3 or 3x2 grid collage
    if len(rendered_imgs) >= 4:
        row1 = cv2.hconcat(rendered_imgs[:2])
        row2 = cv2.hconcat(rendered_imgs[2:4])
        grid = cv2.vconcat([row1, row2])
        grid_out = ARTIFACT_DIR / "dlt_lpr_dataset_montage.jpg"
        cv2.imwrite(str(grid_out), grid)
        print(f"Grid montage saved to: {grid_out}")

    print("Completed visualizations successfully.")

if __name__ == "__main__":
    main()
