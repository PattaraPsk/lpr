import sys
import glob
from pathlib import Path
import cv2
from ultralytics import YOLO

rf_model_path = Path("RAER-TNORF.v5-gray-blur.yolov8.i640.e1000/train/weights/best.pt")
plate_model_path = Path("yolov11x_license_plate.pt")

if not rf_model_path.exists():
    for alt in [Path("rear_front_best.pt"), Path("models/rear_front_best.pt"), Path("models/detector/front_rear_lpr_yolov11n.pt")]:
        if alt.exists():
            rf_model_path = alt
            break

if not plate_model_path.exists():
    for alt_p in [Path("models/detector/front_rear_lpr_yolov11n.pt"), Path("rear_front_best.pt"), Path("yolov8s-worldv2.pt")]:
        if alt_p.exists():
            plate_model_path = alt_p
            break

if not rf_model_path.exists() or not plate_model_path.exists():
    print(f"[INFO] Model not found: rf={rf_model_path.exists()}, plate={plate_model_path.exists()}")
    sys.exit(0)

rf_model = YOLO(str(rf_model_path))
plate_model = YOLO(str(plate_model_path))

samples = glob.glob("dataset/yolo_dlt_lpr_dataset/images/train/*.jpg")[:4]

for test_img_path in samples:
    frame = cv2.imread(test_img_path)
    if frame is None:
        continue
    rf_res = rf_model(frame, conf=0.25, verbose=False)[0]
    p_res = plate_model(frame, conf=0.25, imgsz=1280, verbose=False)[0]
    
    print(f"\nResults on {test_img_path}:")
    print(f"  Front/Back detections: {len(rf_res.boxes)}")
    for b in rf_res.boxes:
        cls_name = rf_model.names[int(b.cls[0])]
        print(f"    - Orientation: {cls_name} (conf: {float(b.conf[0]):.2f}) at {[round(x,1) for x in b.xyxy[0].tolist()]}")
        
    print(f"  Plate detections: {len(p_res.boxes)}")
    for b in p_res.boxes:
        print(f"    - License Plate (conf: {float(b.conf[0]):.2f}) at {[round(x,1) for x in b.xyxy[0].tolist()]}")
