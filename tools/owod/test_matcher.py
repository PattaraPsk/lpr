import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

vehicle_model = YOLO("yolov8s.pt")
plate_model = YOLO("yolov11x_license_plate.pt")

CLASS_NAMES = ["car_plate", "motorcycle_plate", "truck_plate", "bus_plate", "license_plate"]
COCO_VEHICLE_MAP = {2: 0, 3: 1, 7: 2, 5: 3} # car->0, motorcycle->1, truck->2, bus->3

def match_plates_to_vehicles(v_boxes, p_boxes, img_shape):
    """
    v_boxes: list of (box_xyxy, cls_id, conf)
    p_boxes: list of (box_xyxy, conf)
    returns: list of (plate_xyxy, class_id, conf)
    """
    matched = []
    for p_box, p_conf in p_boxes:
        px1, py1, px2, py2 = p_box
        p_cx = (px1 + px2) / 2.0
        p_cy = (py1 + py2) / 2.0
        p_area = (px2 - px1) * (py2 - py1)
        if p_area <= 0:
            continue
            
        best_v = None
        best_score = -1.0
        
        for v_box, v_cls, v_conf in v_boxes:
            vx1, vy1, vx2, vy2 = v_box
            # Check overlap
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            inter_area = iw * ih
            overlap_ratio = inter_area / p_area
            
            # Check if plate center is inside vehicle box
            is_center_inside = (vx1 <= p_cx <= vx2) and (vy1 <= p_cy <= vy2)
            
            if is_center_inside or overlap_ratio > 0.4:
                # Prioritize smaller vehicle bounding box if multiple match (e.g. motorcycle in front of truck)
                v_area = (vx2 - vx1) * (vy2 - vy1)
                score = overlap_ratio + (1.0 / (v_area + 1e-5)) * 1000.0
                if score > best_score:
                    best_score = score
                    best_v = v_cls
                    
        if best_v is not None and best_v in COCO_VEHICLE_MAP:
            target_cls = COCO_VEHICLE_MAP[best_v]
            matched.append((p_box, target_cls, p_conf))
        elif p_conf >= 0.5: # plate detected with high confidence even if vehicle not recognized
            matched.append((p_box, 4, p_conf)) # class 4: general license_plate
            
    return matched

print("Matching function defined successfully.")
