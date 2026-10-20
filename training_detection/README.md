# License Plate Detection Model Training

Custom training and ONNX deployment suite for vehicle license plate detection models (YOLOv8 / YOLOv11 / YOLO-World) optimized for Thai plate aspect ratios and edge runtime performance.

```
training_detection/
├── datasets/                       # Raw YOLO format dataset archives (.zip)
│   ├── motorcycle license plate thai.v12i.yolov11.zip
│   ├── thai license plate.v2i.yolov11.zip
│   ├── Thai license plate.v6-thai0.0.2.yolov11.zip
│   ├── thai-license-plate-character-detect.v1i.yolov11.zip
│   └── Thailand-License-Plate-Recognition.v1i.yolov11.zip
├── configs/
│   └── yolo_detection.yaml         # Training hyperparameters (AdamW, augmentations, loss gains)
├── prepare_dataset.py              # Unpacks datasets and generates data.yaml
├── train.py                        # Ultralytics detector training engine
├── export_onnx.py                  # Exports PyTorch .pt -> ONNX with dynamic batching
├── evaluate.py                     # Evaluates mAP50, mAP50-95, precision, and recall
├── train_detector.bat              # 1-Click interactive launcher for training & export
└── README.md                       # This documentation
```

---

## 1. Quickstart: 3 Steps to Train & Deploy

### Step 1: Unpack Datasets
Extract one or all YOLO archives into standard `train`/`valid`/`test` folders:
```powershell
python prepare_dataset.py --extract-all
```

### Step 2: Run Detector Training
Train on your GPU (e.g. RTX 4090):
```powershell
python train.py --model yolov8s.pt --epochs 100 --batch 16 --imgsz 640 --device 0 --name thai_plate_yolov8s
```
*Tip: For edge CPU deployment, train `yolo11n.pt` for maximum speed with minimal accuracy loss.*

### Step 3: Export to ONNX & Deploy
Export the best weights to ONNX format and automatically deploy into `../pipeline/models/`:
```powershell
python export_onnx.py --weights runs/detect/thai_plate_yolov8s/weights/best.pt --deploy-to-pipeline
```

---

## 2. Interactive Windows Launcher
Run `train_detector.bat` to launch the menu-driven training and export assistant.
