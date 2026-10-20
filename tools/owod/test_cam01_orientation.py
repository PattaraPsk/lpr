import sys
import cv2
from ultralytics import YOLO

raw_path = sys.argv[1] if len(sys.argv) > 1 else 'cam01_rtsp_sample.jpg'
rot_path = sys.argv[2] if len(sys.argv) > 2 else 'cam01_rtsp_rotated.jpg'

raw = cv2.imread(raw_path)
rot = cv2.imread(rot_path)

if raw is None or rot is None:
    print(f"[INFO] Test image '{raw_path}' or '{rot_path}' not found.")
    print("Usage: python test_cam01_orientation.py [raw_image_path] [rotated_image_path]")
    sys.exit(0)

m_plate = YOLO('yolov11x_license_plate.pt')

r_p_raw = m_plate(raw, conf=0.15, verbose=False)[0]
r_v_raw = m_veh(raw, classes=[2, 3, 5, 7], conf=0.15, verbose=False)[0]
print('RAW (upside down): Plates:', len(r_p_raw.boxes), 'Vehicles:', len(r_v_raw.boxes))

r_p_rot = m_plate(rot, conf=0.15, verbose=False)[0]
r_v_rot = m_veh(rot, classes=[2, 3, 5, 7], conf=0.15, verbose=False)[0]
print('ROTATED (upright): Plates:', len(r_p_rot.boxes), 'Vehicles:', len(r_v_rot.boxes))
for b in r_p_rot.boxes:
    print('  Rotated Plate conf:', float(b.conf[0]), 'box:', [round(x, 1) for x in b.xyxy[0].tolist()])
for b in r_v_rot.boxes:
    print('  Rotated Vehicle class:', m_veh.names[int(b.cls[0])], 'conf:', float(b.conf[0]))
