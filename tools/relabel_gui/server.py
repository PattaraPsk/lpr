#!/usr/bin/env python3
"""
Thai LPR Dataset Relabeling Studio - Unified Backend Server
Supports:
1. Detection Relabeler: YOLO bounding box annotation editor (.txt) with optional YOLO auto-detect.
2. OCR Relabeler: Thai license plate transcription, 77-province selection, metadata/renaming.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import re
import urllib.parse
import mimetypes
import shutil
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import datetime

# Canonical 77 Thai Provinces
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา", "ชลบุรี", "ชัยนาท",
    "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง",
    "ตราด", "ตาก", "นครนายก", "นครปฐม", "นครพนม",
    "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส",
    "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์",
    "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พะเยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน",
    "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง",
    "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน", "เลย",
    "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี",
    "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์", "หนองคาย",
    "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์",
    "อุทัยธานี", "อุบลราชธานี"
]

TOP_PROVINCES = [
    "กรุงเทพมหานคร", "เชียงใหม่", "ชลบุรี", "นนทบุรี", "ปทุมธานี",
    "ขอนแก่น", "นครราชสีมา", "สงขลา", "ภูเก็ต", "สมุทรปราการ"
]

THAI_CONSONANTS = [
    "ก", "ข", "ค", "ฆ", "ง", "จ", "ฉ", "ช", "ฌ", "ญ",
    "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท",
    "ธ", "น", "บ", "ป", "ผ", "พ", "ฟ", "ภ", "ม", "ย",
    "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ROOT_LPR = Path("D:/Projects-anti/lpr").resolve()

# Cache for YOLO model if Ultralytics is available
YOLO_MODEL = None
YOLO_INITIALIZED = False


def get_yolo_model():
    global YOLO_MODEL, YOLO_INITIALIZED
    if YOLO_INITIALIZED:
        return YOLO_MODEL
    YOLO_INITIALIZED = True
    try:
        from ultralytics import YOLO
        # Search for available weights
        candidates = [
            ROOT_LPR / "yolov8n.pt",
            ROOT_LPR / "pipeline" / "models" / "yolov8n.pt",
            Path("yolov8n.pt")
        ]
        for c in candidates:
            if c.exists():
                print(f"[AI] Loading YOLO detector from {c}...")
                YOLO_MODEL = YOLO(str(c))
                print("[AI] YOLO detector loaded successfully.")
                return YOLO_MODEL
        # Fallback to load default yolov8n
        print("[AI] Loading default yolov8n.pt...")
        YOLO_MODEL = YOLO("yolov8n.pt")
        return YOLO_MODEL
    except Exception as e:
        print(f"[AI] YOLO not loaded or ultralytics not installed: {e}")
        YOLO_MODEL = None
        return None


def sanitize_filename(name: str) -> str:
    """Windows safe filename sanitizer."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = re.sub(r'\s+', '_', sanitized.strip())
    return sanitized


def find_label_file_for_image(img_path: Path) -> Path:
    """Finds or determines YOLO label file path for given image path."""
    # Pattern 1: Same directory (img.jpg -> img.txt)
    same_dir_txt = img_path.with_suffix(".txt")
    if same_dir_txt.exists():
        return same_dir_txt
    
    # Pattern 2: YOLO standard dataset layout images/... -> labels/...
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        label_candidate = Path(*parts).with_suffix(".txt")
        return label_candidate

    return same_dir_txt


def read_yolo_labels(label_path: Path):
    """Parses YOLO label text file into list of bounding boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        boxes.append({
                            "class_id": cls_id,
                            "x_center": max(0.0, min(1.0, xc)),
                            "y_center": max(0.0, min(1.0, yc)),
                            "width": max(0.0, min(1.0, w)),
                            "height": max(0.0, min(1.0, h))
                        })
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[WARN] Error reading label file {label_path}: {e}")
    return boxes


def write_yolo_labels(label_path: Path, boxes):
    """Writes boxes to YOLO format text file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        cls_id = int(b.get("class_id", 0))
        xc = float(b.get("x_center", 0.0))
        yc = float(b.get("y_center", 0.0))
        w = float(b.get("width", 0.0))
        h = float(b.get("height", 0.0))
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def parse_ocr_from_filename(stem: str):
    """Tries to extract plate number and province from filename stem."""
    plate_number = ""
    province = ""
    
    # 1. Check if any Thai province matches in filename
    for prov in THAI_PROVINCES:
        if prov in stem:
            province = prov
            stem = stem.replace(prov, "")
            break

    # 2. Split by underscore or hyphen
    raw_tokens = [t.strip() for t in re.split(r'[_+\-\s]+', stem) if t.strip()]
    
    # Filter out common video crop metadata & generic placeholders:
    # dates (20260520), times (115924), frame (f0012), score (s49), camera (YDXJ1232), generic (thai, plate, num, img)
    clean_tokens = []
    for t in raw_tokens:
        if re.match(r'^(f\d+|s\d+|\d{5,}|\d{1,4}|ydxj\d+|frame\d+|img\d+|crop\d+|thai|plate|num|image|sample|test|dataset)$', t, re.IGNORECASE):
            continue
        clean_tokens.append(t)

    # If any token has Thai characters or letters+digits combo
    for t in clean_tokens:
        if re.search(r'[\u0e00-\u0e7f]', t):
            plate_number = t
            break
    
    if not plate_number and clean_tokens:
        # Candidate could be like "1กข8920" or "8920"
        candidate = " ".join(clean_tokens[:2])
        if re.search(r'[\u0e00-\u0e7f]', candidate) or (re.search(r'[a-zA-Z]', candidate) and re.search(r'\d', candidate)):
            plate_number = candidate

    return plate_number, province


def load_metadata_manifest(folder: Path):
    """Loads metadata.jsonl or labels.csv if present."""
    manifest = {}
    jsonl_path = folder / "metadata.jsonl"
    if jsonl_path.exists():
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        fn = entry.get("filename")
                        if fn:
                            manifest[fn] = entry
        except Exception as e:
            print(f"[WARN] Failed parsing {jsonl_path}: {e}")

    csv_path = folder / "labels.csv"
    if csv_path.exists() and not manifest:
        try:
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fn = row.get("filename")
                    if fn:
                        manifest[fn] = row
        except Exception as e:
            print(f"[WARN] Failed parsing {csv_path}: {e}")

    return manifest


class RelabelRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable aggressive caching for API
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=400):
        self.send_json({"success": False, "error": message}, status=status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # Static assets
        if path == "/" or path == "/index.html":
            self.serve_static_file("index.html", "text/html; charset=utf-8")
            return
        elif path.startswith("/static/"):
            rel_file = path[len("/static/"):]
            mime_type, _ = mimetypes.guess_type(rel_file)
            self.serve_static_file(rel_file, mime_type or "application/octet-stream")
            return

        # APIs
        if path == "/api/quick_dirs":
            self.handle_quick_dirs()
        elif path == "/api/provinces":
            self.send_json({
                "provinces": THAI_PROVINCES,
                "top_provinces": TOP_PROVINCES,
                "consonants": THAI_CONSONANTS
            })
        elif path == "/api/browse":
            folder_str = query.get("dir", [""])[0]
            self.handle_browse(folder_str)
        elif path == "/api/dataset_items":
            folder_str = query.get("dir", [""])[0]
            mode = query.get("mode", ["detection"])[0]
            self.handle_dataset_items(folder_str, mode)
        elif path == "/api/image":
            img_path_str = query.get("path", [""])[0]
            self.handle_serve_image(img_path_str)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(post_body) if post_body else {}
        except Exception:
            payload = {}

        if path == "/api/detection/save":
            self.handle_detection_save(payload)
        elif path == "/api/detection/auto_detect":
            self.handle_detection_auto_detect(payload)
        elif path == "/api/ocr/save":
            self.handle_ocr_save(payload)
        elif path == "/api/file/delete":
            self.handle_file_delete(payload)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def serve_static_file(self, filename: str, content_type: str):
        static_dir = Path(__file__).resolve().parent / "static"
        file_path = static_dir / filename
        if not file_path.exists():
            self.send_response(404)
            self.end_headers()
            self.wfile.write(f"File not found: {filename}".encode("utf-8"))
            return
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(e).encode("utf-8"))

    def handle_serve_image(self, img_path_str: str):
        if not img_path_str:
            self.send_error_json("Missing image path", 400)
            return
        p = Path(img_path_str)
        if not p.exists() or not p.is_file():
            self.send_error_json(f"Image not found: {img_path_str}", 404)
            return

        mime_type, _ = mimetypes.guess_type(str(p))
        if not mime_type:
            mime_type = "image/jpeg"

        try:
            with open(p, "rb") as f:
                img_data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(img_data)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(img_data)
        except Exception as e:
            self.send_error_json(f"Error serving image: {e}", 500)

    def handle_quick_dirs(self):
        """Returns list of curated directories in projects-anti/lpr."""
        curated = [
            {
                "name": "Raw Video Plate Crops (Today's Crops)",
                "path": str(ROOT_LPR / "dataset" / "raw_video_crops"),
                "recommended_mode": "ocr",
                "desc": "Fresh crops extracted directly from dashcam videos"
            },
            {
                "name": "Verified Thai License Plates Storage",
                "path": str(ROOT_LPR / "dataset" / "storage" / "verified"),
                "recommended_mode": "ocr",
                "desc": "Clean verified plate crops"
            },
            {
                "name": "Thai Plates Dataset (Wikimedia / Storage)",
                "path": str(ROOT_LPR / "dataset" / "storage" / "thai_license_plate_dataset"),
                "recommended_mode": "ocr",
                "desc": "Downloaded authentic plate photos with metadata.jsonl"
            },
            {
                "name": "VLM Sorter Real Plates",
                "path": str(ROOT_LPR / "dataset" / "labeling" / "vlm_sorter" / "real_plates"),
                "recommended_mode": "ocr",
                "desc": "Filtered real physical plates"
            },
            {
                "name": "VLM Sorter FB Plates (Unfiltered Input)",
                "path": str(ROOT_LPR / "dataset" / "labeling" / "vlm_sorter" / "fb_plates"),
                "recommended_mode": "ocr",
                "desc": "Raw Facebook harvested vehicle photos"
            },
            {
                "name": "YOLO Detection Datasets Root",
                "path": str(ROOT_LPR / "training_detection" / "datasets"),
                "recommended_mode": "detection",
                "desc": "YOLOv11 vehicle/plate detection archives & folders"
            },
            {
                "name": "OCR Synthetic Augmented Plates",
                "path": str(ROOT_LPR / "training_ocr" / "synthetic_data" / "augmented_plates"),
                "recommended_mode": "ocr",
                "desc": "Synthetically generated plates across 77 provinces"
            },
            {
                "name": "OCR Character Bank (Isolated Glyphs)",
                "path": str(ROOT_LPR / "training_ocr" / "character_bank" / "char_bank"),
                "recommended_mode": "ocr",
                "desc": "Individual character crops"
            }
        ]

        # Verify which ones exist
        results = []
        for c in curated:
            p = Path(c["path"])
            exists = p.exists()
            img_count = 0
            if exists:
                try:
                    img_count = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
                except Exception:
                    img_count = 0
            c["exists"] = exists
            c["image_count"] = img_count
            results.append(c)

        self.send_json({"quick_dirs": results})

    def handle_browse(self, folder_str: str):
        target = Path(folder_str) if folder_str else ROOT_LPR
        if not target.exists():
            target = ROOT_LPR
        
        parent = str(target.parent) if target.parent != target else ""
        subdirs = []
        try:
            for item in sorted(target.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    # Check images inside
                    has_imgs = any(f.suffix.lower() in SUPPORTED_EXTENSIONS for f in item.iterdir() if f.is_file())
                    subdirs.append({
                        "name": item.name,
                        "path": str(item.resolve()),
                        "has_images": has_imgs
                    })
        except Exception as e:
            print(f"[WARN] Browse directory error: {e}")

        self.send_json({
            "current_dir": str(target.resolve()),
            "parent_dir": parent,
            "subdirs": subdirs
        })

    def handle_dataset_items(self, folder_str: str, mode: str):
        folder = Path(folder_str)
        if not folder.exists() or not folder.is_dir():
            self.send_error_json(f"Folder not found: {folder_str}", 404)
            return

        # Find classes file for detection mode if available
        classes = ["license_plate"]
        classes_file = folder / "classes.txt"
        if not classes_file.exists() and folder.parent:
            classes_file = folder.parent / "classes.txt"
        if classes_file.exists():
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    c_lines = [l.strip() for l in f if l.strip()]
                    if c_lines:
                        classes = c_lines
            except Exception:
                pass

        # Load metadata manifest if OCR mode
        metadata_map = load_metadata_manifest(folder)

        # Collect image files
        items = []
        try:
            image_files = sorted(
                [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS],
                key=lambda x: x.name
            )
        except Exception as e:
            self.send_error_json(f"Cannot read directory: {e}", 500)
            return

        for img in image_files:
            item_data = {
                "name": img.name,
                "path": str(img.resolve()),
                "ext": img.suffix.lower(),
                "size_kb": round(img.stat().st_size / 1024, 1)
            }

            if mode == "detection":
                label_path = find_label_file_for_image(img)
                has_label = label_path.exists()
                boxes = read_yolo_labels(label_path) if has_label else []
                item_data["has_label"] = has_label and len(boxes) > 0
                item_data["label_path"] = str(label_path.resolve())
                item_data["boxes"] = boxes
            else:
                # OCR Mode
                meta = metadata_map.get(img.name, {})
                guess_num, guess_prov = parse_ocr_from_filename(img.stem)

                plate_num = meta.get("plate_number") or guess_num
                province = meta.get("province") or guess_prov
                category = meta.get("category", "real_physical")
                is_verified = meta.get("verified", False) or bool(plate_num and province)

                item_data["has_label"] = bool(plate_num)
                item_data["plate_number"] = plate_num
                item_data["province"] = province
                item_data["category"] = category
                item_data["is_verified"] = is_verified

            items.append(item_data)

        self.send_json({
            "folder": str(folder.resolve()),
            "count": len(items),
            "classes": classes,
            "items": items
        })

    def handle_detection_save(self, payload: dict):
        img_path_str = payload.get("image_path")
        boxes = payload.get("boxes", [])
        if not img_path_str:
            self.send_error_json("Missing image_path")
            return

        img_path = Path(img_path_str)
        label_path = find_label_file_for_image(img_path)
        try:
            write_yolo_labels(label_path, boxes)
            self.send_json({
                "success": True,
                "message": f"Saved {len(boxes)} boxes to {label_path.name}",
                "label_path": str(label_path.resolve()),
                "box_count": len(boxes)
            })
        except Exception as e:
            self.send_error_json(f"Failed to write label: {e}", 500)

    def handle_detection_auto_detect(self, payload: dict):
        img_path_str = payload.get("image_path")
        conf_thresh = float(payload.get("confidence", 0.25))
        if not img_path_str:
            self.send_error_json("Missing image_path")
            return

        p = Path(img_path_str)
        if not p.exists():
            self.send_error_json("Image does not exist", 404)
            return

        model = get_yolo_model()
        if model is None:
            self.send_error_json("YOLO model not available. Please install ultralytics or place yolov8n.pt.", 501)
            return

        try:
            from PIL import Image
            with Image.open(p) as pil_img:
                img_w, img_h = pil_img.size

            results = model.predict(str(p), conf=conf_thresh, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    # xywh normalized: box.xywhn
                    xywhn = box.xywhn[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    boxes.append({
                        "class_id": 0, # Map to plate class 0 by default
                        "x_center": round(xywhn[0], 6),
                        "y_center": round(xywhn[1], 6),
                        "width": round(xywhn[2], 6),
                        "height": round(xywhn[3], 6),
                        "confidence": round(conf, 3)
                    })
            self.send_json({
                "success": True,
                "boxes": boxes,
                "count": len(boxes)
            })
        except Exception as e:
            self.send_error_json(f"Auto detection failed: {e}", 500)

    def handle_ocr_save(self, payload: dict):
        img_path_str = payload.get("image_path")
        plate_number = (payload.get("plate_number") or "").strip()
        province = (payload.get("province") or "").strip()
        category = payload.get("category", "real_physical")
        save_mode = payload.get("save_mode", "both")  # 'rename', 'metadata', or 'both'
        move_to_subfolder = payload.get("move_to_subfolder", False)

        if not img_path_str:
            self.send_error_json("Missing image_path")
            return

        img_path = Path(img_path_str)
        if not img_path.exists():
            self.send_error_json("Image not found", 404)
            return

        folder = img_path.parent
        orig_ext = img_path.suffix.lower()
        new_path = img_path

        # 1. Direct file rename if requested
        if save_mode in ("rename", "both") and (plate_number or province):
            base_tokens = []
            if plate_number:
                base_tokens.append(sanitize_filename(plate_number))
            if province:
                base_tokens.append(sanitize_filename(province))
            
            # Keep original unique id or timestamp suffix if already present
            stem_tokens = img_path.stem.split("_")
            last_token = stem_tokens[-1] if len(stem_tokens) > 1 and (stem_tokens[-1].isdigit() or stem_tokens[-1].startswith("f") or stem_tokens[-1].startswith("s")) else ""
            if last_token:
                base_tokens.append(last_token)

            new_stem = "_".join(base_tokens)
            if not new_stem:
                new_stem = img_path.stem

            candidate = folder / f"{new_stem}{orig_ext}"
            counter = 1
            while candidate.exists() and candidate.resolve() != img_path.resolve():
                candidate = folder / f"{new_stem}_{counter}{orig_ext}"
                counter += 1

            if candidate.resolve() != img_path.resolve():
                try:
                    img_path.rename(candidate)
                    new_path = candidate
                except Exception as e:
                    print(f"[WARN] Rename failed: {e}")

        # 2. Move to category subfolder if enabled
        if move_to_subfolder and category:
            cat_dir = folder / sanitize_filename(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            target_moved = cat_dir / new_path.name
            try:
                shutil.move(str(new_path), str(target_moved))
                new_path = target_moved
            except Exception as e:
                print(f"[WARN] Move to folder failed: {e}")

        # 3. Update metadata.jsonl
        if save_mode in ("metadata", "both"):
            jsonl_file = folder / "metadata.jsonl"
            existing_records = []
            updated = False
            if jsonl_file.exists():
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            if rec.get("filename") in (img_path.name, new_path.name):
                                rec["filename"] = new_path.name
                                rec["plate_number"] = plate_number
                                rec["province"] = province
                                rec["category"] = category
                                rec["verified"] = True
                                rec["updated_at"] = datetime.datetime.now().isoformat()
                                updated = True
                            existing_records.append(rec)
                except Exception as e:
                    print(f"[WARN] Error reading metadata.jsonl: {e}")

            if not updated:
                existing_records.append({
                    "filename": new_path.name,
                    "plate_number": plate_number,
                    "province": province,
                    "category": category,
                    "verified": True,
                    "updated_at": datetime.datetime.now().isoformat()
                })

            try:
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for rec in existing_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] Error saving metadata.jsonl: {e}")

        self.send_json({
            "success": True,
            "new_path": str(new_path.resolve()),
            "new_filename": new_path.name,
            "plate_number": plate_number,
            "province": province,
            "category": category
        })

    def handle_file_delete(self, payload: dict):
        img_path_str = payload.get("image_path")
        if not img_path_str:
            self.send_error_json("Missing image_path")
            return
        p = Path(img_path_str)
        if not p.exists():
            self.send_error_json("File not found", 404)
            return

        # Move to trash subfolder instead of permanent destruction for safety
        trash_dir = p.parent / "_discarded"
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / p.name
        try:
            shutil.move(str(p), str(target))
            # Also move associated label if exists
            label_p = find_label_file_for_image(p)
            if label_p.exists():
                shutil.move(str(label_p), str(trash_dir / label_p.name))
            self.send_json({"success": True, "message": f"Moved to {trash_dir.name}"})
        except Exception as e:
            self.send_error_json(f"Failed to delete file: {e}", 500)


def run_server(port=8501):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, RelabelRequestHandler)
    print("=" * 65)
    print("  [OK] Thai LPR Dataset Relabeling Studio - Active")
    print(f"  [URL] http://127.0.0.1:{port}")
    print("  Press Ctrl+C to stop the server")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    import webbrowser
    port = 8501
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    
    # Open browser automatically if not specified --no-browser
    if "--no-browser" not in sys.argv:
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass
    run_server(port)
