import cv2
import torch
from ultralytics import YOLO

print("Loading models...")
vehicle_model = YOLO("yolov8s.pt")
plate_model = YOLO("yolov11x_license_plate.pt")

print("Plate model names:", plate_model.names)
print("Vehicle model names:", {k: vehicle_model.names[k] for k in [2, 3, 5, 7]})

# Test on 1 frame from LPR
cap = cv2.VideoCapture("dataset/LPR-20261019T045913Z-1-001/LPR/20260520_115924.mp4")
ret, frame = cap.read()
cap.release()

if ret:
    v_res = vehicle_model(frame, classes=[2, 3, 5, 7], conf=0.3)[0]
    p_res = plate_model(frame, conf=0.25, imgsz=1280)[0]
    print(f"Frame 1 test: found {len(v_res.boxes)} vehicles, {len(p_res.boxes)} plates")
    for box in p_res.boxes:
        print("  Plate box:", box.xyxy[0].tolist(), "conf:", float(box.conf[0]))
