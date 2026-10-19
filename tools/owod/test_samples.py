import cv2
import os
import glob
from ultralytics import YOLO
from test_matcher import match_plates_to_vehicles, CLASS_NAMES

vehicle_model = YOLO("yolov8s.pt")
plate_model = YOLO("yolov11x_license_plate.pt")

os.makedirs("test_samples", exist_ok=True)

test_videos = [
    ("dataset/LPR-20261019T045913Z-1-001/LPR/20260520_120057.mp4", 150),
    ("dataset/LPR-20261019T045913Z-1-001/LPR/YDXJ1232.mp4", 600),
    ("dataset/Extra_Videos-20261019T045943Z-1-001/Extra_Videos/1.mp4", 200),
    ("dataset/Raw Video Record/vlc-record-2026-08-19-CAM01-Main01.avi", 100),
]

for idx, (vpath, frame_idx) in enumerate(test_videos):
    cap = cv2.VideoCapture(vpath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read {vpath}")
        continue
        
    v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=0.25, verbose=False)[0]
    p_res = plate_model(frame, conf=0.25, imgsz=1280, verbose=False)[0]
    
    v_boxes = [(b.xyxy[0].tolist(), int(b.cls[0]), float(b.conf[0])) for b in v_res.boxes]
    p_boxes = [(b.xyxy[0].tolist(), float(b.conf[0])) for b in p_res.boxes]
    
    matched = match_plates_to_vehicles(v_boxes, p_boxes, frame.shape)
    print(f"Video {os.path.basename(vpath)} frame {frame_idx}: {len(v_boxes)} vehicles, {len(p_boxes)} raw plates -> {len(matched)} matched plates")
    
    # Draw
    vis = frame.copy()
    for p_box, cls_id, conf in matched:
        x1, y1, x2, y2 = map(int, p_box)
        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        print(f"  -> {label} at [{x1}, {y1}, {x2}, {y2}]")
        color = (0, 255, 0) if cls_id == 0 else (0, 255, 255) if cls_id == 1 else (255, 128, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    cv2.imwrite(f"test_samples/sample_{idx}.jpg", vis)

print("Saved test samples to test_samples/")
