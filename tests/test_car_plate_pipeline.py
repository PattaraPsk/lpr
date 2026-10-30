import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from tools.DataManipulator.app.plate_reader import PlateReader

veh_model = YOLO('models/detector/yolov8s.pt').to('cuda:0')
plate_model = YOLO('models/detector/vehicle_plate_yolov11n.pt').to('cuda:0')
ocr_reader = PlateReader.get_instance('models/reader/thai_plate_reader.pth')

cap = cv2.VideoCapture('dataset/raw/27-Oct-2026/05_SaKhu_C1-SKU-SB-01.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 500)

tracks = {}
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    if i % 5 != 0:
        continue
    
    h, w = frame.shape[:2]
    res = veh_model.track(frame, classes=[2, 3, 5, 7], conf=0.35, persist=True, verbose=False, tracker='bytetrack.yaml')[0]
    if res.boxes.id is None:
        continue
    
    boxes = res.boxes.xyxy.cpu().numpy()
    ids = res.boxes.id.int().cpu().numpy()
    clss = res.boxes.cls.int().cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    
    for box, trk_id, cls_id, conf in zip(boxes, ids, clss, confs):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0:
            continue
        
        # Detect plate inside car crop
        p_res = plate_model(car_crop, conf=0.2, verbose=False)[0]
        best_p_box, best_p_conf, best_orient = None, 0.0, 'unknown'
        for pb in p_res.boxes:
            pc = int(pb.cls[0])
            p_name = plate_model.names[pc]
            if p_name == 'LicensePlate':
                if float(pb.conf[0]) > best_p_conf:
                    best_p_conf = float(pb.conf[0])
                    best_p_box = pb.xyxy[0].cpu().numpy()
            elif p_name in ['Front', 'Rear']:
                best_orient = p_name
        
        ocr_text = ''
        plate_crop = None
        if best_p_box is not None:
            px1, py1, px2, py2 = map(int, best_p_box)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(car_crop.shape[1], px2), min(car_crop.shape[0], py2)
            plate_crop = car_crop[py1:py2, px1:px2]
            if plate_crop.size > 0:
                ocr_out = ocr_reader.predict(Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)))
                ocr_text = ocr_out.get('full_plate', '')
                
        if trk_id not in tracks:
            tracks[trk_id] = {
                'class': veh_model.names[cls_id],
                'frames': 1,
                'best_car_crop': car_crop,
                'best_plate_crop': plate_crop,
                'ocr_text': ocr_text,
                'orient': best_orient
            }
        else:
            tracks[trk_id]['frames'] += 1
            if plate_crop is not None and tracks[trk_id]['best_plate_crop'] is None:
                tracks[trk_id]['best_plate_crop'] = plate_crop
                tracks[trk_id]['best_car_crop'] = car_crop
                tracks[trk_id]['ocr_text'] = ocr_text
                tracks[trk_id]['orient'] = best_orient

cap.release()
print(f'Detected tracks: {len(tracks)}')
for tid, d in tracks.items():
    print(f"Track {tid}: {d['class']}, frames={d['frames']}, orient={d['orient']}, plate='{d['ocr_text']}'")
