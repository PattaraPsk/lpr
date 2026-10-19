import os
import glob
import cv2
import torch
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Configuration
OUTPUT_DIR = Path("dataset/yolo_dlt_lpr_dataset")
IMG_TRAIN_DIR = OUTPUT_DIR / "images" / "train"
IMG_VAL_DIR = OUTPUT_DIR / "images" / "val"
LBL_TRAIN_DIR = OUTPUT_DIR / "labels" / "train"
LBL_VAL_DIR = OUTPUT_DIR / "labels" / "val"

VAL_SPLIT = 0.15
RANDOM_SEED = 42
PLATE_CONF = 0.25
VEHICLE_CONF = 0.25
IMGSZ = 1280

CLASS_NAMES = [
    "car_plate",         # 0: Private passenger cars <= 7 seats (sedan, SUV, hatchback)
    "motorcycle_plate",  # 1: Motorcycles / scooters
    "truck_plate",       # 2: Pickup trucks, flatbeds, heavy commercial trucks
    "bus_plate",         # 3: Vans (> 7 seats), minibuses, public transit buses
    "license_plate"      # 4: General license plate fallback
]

# COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_MAP = {2: 0, 3: 1, 7: 2, 5: 3}

# Directories to STRICTLY skip (pre-cropped plates)
EXCLUDE_DIRS = [
    "curated_lpr_dataset",
    "ood_anomalies",
    "plates",
    "yolo_plate_ocr_dataset"
]

def setup_directories():
    for d in [IMG_TRAIN_DIR, IMG_VAL_DIR, LBL_TRAIN_DIR, LBL_VAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Dataset directories ready at {OUTPUT_DIR.resolve()}")

def is_excluded_path(p: str) -> bool:
    norm_p = os.path.normpath(p).lower()
    for excl in EXCLUDE_DIRS:
        if excl.lower() in norm_p:
            return True
    return False

def match_plates_to_vehicles(v_boxes, p_boxes, img_shape):
    """
    v_boxes: list of (box_xyxy, cls_id, conf)
    p_boxes: list of (box_xyxy, conf)
    returns: list of (plate_xyxy, target_cls, conf)
    """
    h, w = img_shape[:2]
    matched = []
    
    for p_box, p_conf in p_boxes:
        px1, py1, px2, py2 = p_box
        pw = px2 - px1
        ph = py2 - py1
        p_area = pw * ph
        if p_area <= 0 or pw < 10 or ph < 6:
            continue
            
        p_cx = (px1 + px2) / 2.0
        p_cy = (py1 + py2) / 2.0
        
        best_v = None
        best_score = -1.0
        
        for v_box, v_cls, v_conf in v_boxes:
            vx1, vy1, vx2, vy2 = v_box
            # Overlap calculation
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area = iw * ih
            overlap_ratio = inter_area / p_area
            
            # Check containment of center
            is_center_inside = (vx1 <= p_cx <= vx2) and (vy1 <= p_cy <= vy2)
            
            if is_center_inside or overlap_ratio > 0.35:
                v_area = (vx2 - vx1) * (vy2 - vy1)
                # Prioritize smaller vehicle bounding box (e.g. motorcycle in front of bus)
                score = overlap_ratio * 10.0 + (1.0 / (v_area + 1e-5)) * 10000.0
                if score > best_score:
                    best_score = score
                    best_v = v_cls
                    
        if best_v is not None and best_v in COCO_VEHICLE_MAP:
            matched.append((p_box, COCO_VEHICLE_MAP[best_v], p_conf))
        elif p_conf >= 0.5:
            # High confidence plate detected without vehicle context (e.g. vehicle boundary cut off)
            matched.append((p_box, 4, p_conf))
            
    return matched

def save_sample(img, labels, prefix, index, split):
    """
    labels: list of (box_xyxy, class_id, conf)
    """
    img_name = f"{prefix}_{index:05d}.jpg"
    lbl_name = f"{prefix}_{index:05d}.txt"
    
    img_dir = IMG_VAL_DIR if split == "val" else IMG_TRAIN_DIR
    lbl_dir = LBL_VAL_DIR if split == "val" else LBL_TRAIN_DIR
    
    img_path = img_dir / img_name
    lbl_path = lbl_dir / lbl_name
    
    h, w = img.shape[:2]
    cv2.imwrite(str(img_path), img)
    
    lines = []
    for (x1, y1, x2, y2), cls_id, _ in labels:
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    return img_path, lbl_path

def create_yaml():
    data = {
        "path": str(OUTPUT_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"data.yaml written to {yaml_path}")

def main():
    random.seed(RANDOM_SEED)
    setup_directories()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    vehicle_model = YOLO("yolov8s.pt")
    plate_model = YOLO("yolov11x_license_plate.pt")
    
    stats = {name: 0 for name in CLASS_NAMES}
    total_images = 0
    train_count = 0
    val_count = 0
    
    # 1. Process Annotated_Images (1012 full-scene images)
    ann_imgs = glob.glob("dataset/Extra_Videos-20261019T045943Z-1-001/Extra_Videos/Annotated_Images/imgs/*.*")
    print(f"\nProcessing Annotated_Images ({len(ann_imgs)} images, sampling every 2)...")
    for i, img_path in enumerate(tqdm(ann_imgs[::2])):
        if is_excluded_path(img_path):
            continue
        frame = cv2.imread(img_path)
        if frame is None or frame.shape[0] < 300 or frame.shape[1] < 400:
            continue
            
        v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=VEHICLE_CONF, verbose=False, device=device)[0]
        p_res = plate_model(frame, conf=PLATE_CONF, imgsz=IMGSZ, verbose=False, device=device)[0]
        
        v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
        p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]
        
        matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)
        if matched:
            split = "val" if random.random() < VAL_SPLIT else "train"
            save_sample(frame, matched, "ann_scene", i, split)
            total_images += 1
            if split == "val":
                val_count += 1
            else:
                train_count += 1
            for _, cls_id, _ in matched:
                stats[CLASS_NAMES[cls_id]] += 1

    # 2. Process LPR Dashcam Videos
    lpr_vids = glob.glob("dataset/LPR-20261019T045913Z-1-001/LPR/*.*")
    print(f"\nProcessing {len(lpr_vids)} LPR Videos...")
    for v_idx, vpath in enumerate(lpr_vids):
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = int(max(fps * 0.5, 12)) # Sample twice per second
        f_idx = 0
        v_base = Path(vpath).stem
        
        pbar = tqdm(desc=f"LPR: {v_base}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % step == 0:
                pbar.update(step)
                v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=VEHICLE_CONF, verbose=False, device=device)[0]
                p_res = plate_model(frame, conf=PLATE_CONF, imgsz=IMGSZ, verbose=False, device=device)[0]
                
                v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
                p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]
                
                matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)
                if matched:
                    split = "val" if random.random() < VAL_SPLIT else "train"
                    save_sample(frame, matched, f"lpr_{v_base}", f_idx, split)
                    total_images += 1
                    if split == "val":
                        val_count += 1
                    else:
                        train_count += 1
                    for _, cls_id, _ in matched:
                        stats[CLASS_NAMES[cls_id]] += 1
            f_idx += 1
        cap.release()
        pbar.close()

    # 3. Process Raw Video Record Surveillance Videos
    raw_vids = glob.glob("dataset/Raw Video Record/*.*")
    print(f"\nProcessing {len(raw_vids)} Raw Surveillance Videos...")
    for v_idx, vpath in enumerate(raw_vids):
        if not vpath.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            continue
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        step = int(max(fps * 1.5, 25)) # Sample every 1.5 seconds
        f_idx = 0
        v_base = Path(vpath).stem
        
        pbar = tqdm(desc=f"Raw: {v_base[:20]}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % step == 0:
                pbar.update(step)
                v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=VEHICLE_CONF, verbose=False, device=device)[0]
                p_res = plate_model(frame, conf=PLATE_CONF, imgsz=IMGSZ, verbose=False, device=device)[0]
                
                v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
                p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]
                
                matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)
                if matched:
                    split = "val" if random.random() < VAL_SPLIT else "train"
                    save_sample(frame, matched, f"surv_{v_base[:15]}", f_idx, split)
                    total_images += 1
                    if split == "val":
                        val_count += 1
                    else:
                        train_count += 1
                    for _, cls_id, _ in matched:
                        stats[CLASS_NAMES[cls_id]] += 1
            f_idx += 1
        cap.release()
        pbar.close()

    # 4. Process Extra_Videos
    extra_vids = glob.glob("dataset/Extra_Videos-20261019T045943Z-1-001/Extra_Videos/*.*")
    print(f"\nProcessing Extra Videos...")
    for vpath in extra_vids:
        if not vpath.endswith(('.mp4', '.avi', '.mkv', '.mov')):
            continue
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = int(max(fps * 2.0, 45))
        f_idx = 0
        v_base = Path(vpath).stem
        pbar = tqdm(desc=f"Extra: {v_base}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % step == 0:
                pbar.update(step)
                v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=VEHICLE_CONF, verbose=False, device=device)[0]
                p_res = plate_model(frame, conf=PLATE_CONF, imgsz=IMGSZ, verbose=False, device=device)[0]
                
                v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
                p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]
                
                matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)
                if matched:
                    split = "val" if random.random() < VAL_SPLIT else "train"
                    save_sample(frame, matched, f"extra_{v_base}", f_idx, split)
                    total_images += 1
                    if split == "val":
                        val_count += 1
                    else:
                        train_count += 1
                    for _, cls_id, _ in matched:
                        stats[CLASS_NAMES[cls_id]] += 1
            f_idx += 1
        cap.release()
        pbar.close()

    create_yaml()
    print("\n================ DATASET GENERATION SUMMARY ================")
    print(f"Total images saved: {total_images} (Train: {train_count}, Val: {val_count})")
    print("Class instance counts:")
    for name, cnt in stats.items():
        print(f"  - {name}: {cnt}")
    print(f"Saved to: {OUTPUT_DIR.resolve()}")
    print("============================================================")

if __name__ == "__main__":
    main()
