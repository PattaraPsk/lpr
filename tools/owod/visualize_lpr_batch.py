"""
Visualizer Generator for LPR-20261019T045913Z-1-001 dataset samples.
Renders annotated bounding boxes, crops, montage, and interactive HTML viewer.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure UTF-8 output
if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

def parse_yolo_labels(lbl_path: Path, img_w: int, img_h: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:5]]
                x1 = int((xc - w / 2) * img_w)
                y1 = int((yc - h / 2) * img_h)
                x2 = int((xc + w / 2) * img_w)
                y2 = int((yc + h / 2) * img_h)
                boxes.append({
                    'class_id': cls_id,
                    'box': (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)),
                    'norm': (xc, yc, w, h)
                })
    return boxes

def generate_visualizations(
    dataset_dir: str = "dataset/yolo_owod_labeled_dataset",
    output_dir: str = "dataset/yolo_owod_labeled_dataset/visualizations_lpr",
    artifact_dir: str = ""
):
    ds_path = Path(dataset_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    art_path = Path(artifact_dir).resolve() if artifact_dir else None
    if art_path:
        art_path.mkdir(parents=True, exist_ok=True)
        
    print(f"🔍 Searching for LPR-20261019T045913Z-1-001 samples in {ds_path}...")
    
    # Locate all images
    img_files = sorted(list((ds_path / "images").rglob("*LPR-20261019T045913Z-1-001*")))
    print(f"📸 Found {len(img_files)} images.")
    
    vis_records = []
    montage_frames = []
    
    for idx, img_p in enumerate(img_files):
        # Determine split (train or val)
        split = img_p.parent.name
        lbl_p = ds_path / "labels" / split / f"{img_p.stem}.txt"
        
        # Load image
        data = np.fromfile(str(img_p), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
            
        h, w = img_bgr.shape[:2]
        boxes = parse_yolo_labels(lbl_p, w, h)
        
        # Draw bounding boxes
        annotated = img_bgr.copy()
        crops_list = []
        
        for b_idx, b in enumerate(boxes):
            x1, y1, x2, y2 = b['box']
            # Draw glowing green bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
            cv2.rectangle(annotated, (max(0, x1-1), max(0, y1-1)), (min(w, x2+1), min(h, y2+1)), (255, 255, 255), 1)
            
            # Badge
            label_text = f"plate #{b_idx+1}"
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 6, max(0, y1)), (34, 197, 94), -1)
            cv2.putText(annotated, label_text, (x1 + 3, max(th, y1 - baseline - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Extract crop
            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                crops_list.append(crop)
                
        # Save visualization image
        vis_filename = f"{img_p.stem}_annotated.jpg"
        save_p = out_path / vis_filename
        success, buf = cv2.imencode(".jpg", annotated)
        if success:
            with open(str(save_p), "wb") as f:
                f.write(buf)
                
        # Also copy to artifact dir if provided
        if art_path:
            with open(str(art_path / vis_filename), "wb") as f:
                f.write(buf)
                
        vis_records.append({
            'stem': img_p.stem,
            'split': split,
            'resolution': f"{w}x{h}",
            'boxes_count': len(boxes),
            'vis_file': vis_filename,
            'boxes': boxes,
            'crops_count': len(crops_list)
        })
        
        # Prepare for montage (first 6 samples)
        if len(montage_frames) < 6:
            resized = cv2.resize(annotated, (480, 270))
            montage_frames.append(resized)
            
    # Create a 2x3 or 3x2 montage
    if len(montage_frames) >= 4:
        r1 = np.hstack(montage_frames[:3]) if len(montage_frames) >= 3 else np.hstack(montage_frames[:2])
        r2 = np.hstack(montage_frames[3:6]) if len(montage_frames) >= 6 else (np.hstack(montage_frames[2:4]) if len(montage_frames) >= 4 else r1)
        # Ensure equal widths
        if r1.shape[1] == r2.shape[1]:
            montage = np.vstack([r1, r2])
            montage_p = out_path / "lpr_detections_montage.jpg"
            cv2.imwrite(str(montage_p), montage)
            if art_path:
                cv2.imwrite(str(art_path / "lpr_detections_montage.jpg"), montage)
            print(f"🖼️ Generated Detection Montage: {montage_p}")
            
    # Generate standalone interactive HTML viewer
    html_p = out_path / "lpr_visualizer.html"
    generate_html_viewer(vis_records, html_p)
    if art_path:
        generate_html_viewer(vis_records, art_path / "lpr_visualizer.html")
    print(f"🌐 Generated Interactive HTML Visualizer: {html_p}")
    
    return vis_records

def generate_html_viewer(records, html_file: Path):
    cards_html = ""
    for r in records:
        cards_html += f"""
        <div class="card">
            <div class="card-header">
                <span class="badge {r['split']}">{r['split'].upper()}</span>
                <span class="title">{r['stem']}</span>
                <span class="count">{r['boxes_count']} Plates Found</span>
            </div>
            <div class="img-container">
                <img src="{r['vis_file']}" alt="{r['stem']}" loading="lazy" />
            </div>
            <div class="meta">
                Resolution: {r['resolution']} | Bounding Boxes: {r['boxes_count']}
            </div>
        </div>
        """
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LPR-20261019T045913Z-1-001 Dataset Visualizer</title>
    <style>
        body {{
            background-color: #0f172a;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            color: #38bdf8;
            margin-bottom: 6px;
        }}
        .subtitle {{
            color: #94a3b8;
            margin-bottom: 24px;
            font-size: 14px;
        }}
        .stats-bar {{
            display: flex;
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-badge {{
            background-color: #1e293b;
            padding: 10px 18px;
            border-radius: 8px;
            border: 1px solid #334155;
            font-weight: bold;
        }}
        .stat-badge span {{
            color: #34d399;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
            gap: 20px;
        }}
        .card {{
            background-color: #1e293b;
            border-radius: 10px;
            border: 1px solid #334155;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            display: flex;
            align-items: center;
            padding: 10px 14px;
            background-color: #141f36;
            gap: 10px;
            font-size: 13px;
        }}
        .badge.train {{
            background-color: #059669;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
        }}
        .badge.val {{
            background-color: #6366f1;
            color: white;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
        }}
        .title {{
            font-weight: 600;
            color: #e2e8f0;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .count {{
            color: #38bdf8;
            font-weight: bold;
            font-size: 12px;
        }}
        .img-container {{
            width: 100%;
            background-color: #000;
        }}
        .img-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .meta {{
            padding: 8px 14px;
            font-size: 11px;
            color: #94a3b8;
            background-color: #141f36;
        }}
    </style>
</head>
<body>
    <h1>⚡ LPR-20261019T045913Z-1-001 Detection Visualizer</h1>
    <div class="subtitle">OWOD Zero-Shot Auto-Labeled Samples in <code>dataset/yolo_owod_labeled_dataset</code></div>
    <div class="stats-bar">
        <div class="stat-badge">Total Frames: <span>{len(records)}</span></div>
        <div class="stat-badge">Total Plates: <span>{sum(r['boxes_count'] for r in records)}</span></div>
        <div class="stat-badge">Class: <span>license_plate (0)</span></div>
        <div class="stat-badge">Model: <span>yolov8s-worldv2 (CUDA 12.4 GPU)</span></div>
    </div>
    <div class="grid">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="")
    args = parser.parse_args()
    generate_visualizations(artifact_dir=args.artifact_dir)
