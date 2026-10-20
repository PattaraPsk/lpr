"""
ISO 25012 & ISO 3166-2:TH Compliant Annotation Export & Sync Tool.
Synchronizes DataManipulator labeling sessions between dataset/ and annotations/
for Git version control and cross-workstation portability.
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# Ensure repository root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tools.DataManipulator.app.provinces import normalize_thai_script, get_iso_code

ANNOTATIONS_DIR = ROOT_DIR / "annotations"
DATASET_PLATES_SESSION = ROOT_DIR / "dataset" / "curated_lpr_dataset" / "plates" / ".lpr_session.json"
DATASET_ROOT_SESSION = ROOT_DIR / "dataset" / "curated_lpr_dataset" / ".lpr_session.json"

TARGET_STANDARDS = [
    "ISO 8601:2019",
    "ISO 3166-2:TH",
    "ISO/IEC 10646",
    "ISO/IEC 25012:2008",
    "ISO/IEC 5259-2:2023"
]

def upgrade_and_standardize(session_data: Dict[str, Any]) -> Dict[str, Any]:
    session_data["version"] = "1.1"
    session_data["standards"] = TARGET_STANDARDS
    session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    
    items = session_data.get("items", {})
    for item in items.values():
        prov = item.get("province", "")
        if prov:
            norm_prov = normalize_thai_script(prov)
            item["province"] = norm_prov
            item["iso_3166_2"] = get_iso_code(norm_prov)
        else:
            item["iso_3166_2"] = None
            
        d3 = item.get("d3", "")
        if d3:
            item["d3"] = normalize_thai_script(d3)
            
    return session_data

def export_all():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATASET_PLATES_SESSION.exists():
        print(f"[ERROR] Source session not found at: {DATASET_PLATES_SESSION}")
        return False
        
    print(f"[INFO] Reading session from {DATASET_PLATES_SESSION}...")
    with open(DATASET_PLATES_SESSION, "r", encoding="utf-8") as f:
        plates_data = json.load(f)
        
    plates_data = upgrade_and_standardize(plates_data)
    
    # Save back standardized version to source
    with open(DATASET_PLATES_SESSION, "w", encoding="utf-8") as f:
        json.dump(plates_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Standardized {DATASET_PLATES_SESSION}")
    
    # 1. Export plates_session.json
    out_plates_session = ANNOTATIONS_DIR / "plates_session.json"
    with open(out_plates_session, "w", encoding="utf-8") as f:
        json.dump(plates_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {out_plates_session}")
    
    # 2. Export root curated_lpr_session.json if exists
    if DATASET_ROOT_SESSION.exists():
        with open(DATASET_ROOT_SESSION, "r", encoding="utf-8") as f:
            root_data = json.load(f)
        root_data = upgrade_and_standardize(root_data)
        with open(DATASET_ROOT_SESSION, "w", encoding="utf-8") as f:
            json.dump(root_data, f, ensure_ascii=False, indent=2)
        out_root_session = ANNOTATIONS_DIR / "curated_lpr_session.json"
        with open(out_root_session, "w", encoding="utf-8") as f:
            json.dump(root_data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Exported {out_root_session}")
        
    # 3. Export approved_plates.json and approved_plates.csv
    items = plates_data.get("items", {})
    approved_items = [v for v in items.values() if v.get("status") == "approved"]
    
    out_approved_json = ANNOTATIONS_DIR / "approved_plates.json"
    with open(out_approved_json, "w", encoding="utf-8") as f:
        json.dump({
            "standards": TARGET_STANDARDS,
            "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "total_approved": len(approved_items),
            "approved_plates": approved_items
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {out_approved_json} ({len(approved_items)} records)")
    
    out_approved_csv = ANNOTATIONS_DIR / "approved_plates.csv"
    with open(out_approved_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "name", "plate_type", "d3", "d4", "province", "iso_3166_2", "status"])
        for it in approved_items:
            writer.writerow([
                it.get("filename", ""),
                it.get("name", ""),
                it.get("plate_type", ""),
                it.get("d3", ""),
                it.get("d4", ""),
                it.get("province", ""),
                it.get("iso_3166_2", ""),
                it.get("status", "")
            ])
    print(f"[OK] Exported {out_approved_csv}")
    
    # 4. Generate Markdown Summary
    status_counts = {}
    for it in items.values():
        s = it.get("status", "pending")
        status_counts[s] = status_counts.get(s, 0) + 1
        
    province_counts = {}
    type_counts = {}
    for it in approved_items:
        p = f"{it.get('province')} ({it.get('iso_3166_2')})"
        province_counts[p] = province_counts.get(p, 0) + 1
        t = it.get("plate_type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        
    summary_md = ANNOTATIONS_DIR / "LABELING_PROGRESS.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Thai LPR Dataset Labeling Progress Report\n\n")
        f.write(f"- **Updated (UTC):** {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- **Standards:** {', '.join(TARGET_STANDARDS)}\n")
        f.write(f"- **Current Index Position:** {plates_data.get('current_index', 0)} / {len(items)}\n\n")
        f.write("## 1. Review Status Overview\n\n")
        f.write("| Status | Count | Percentage |\n")
        f.write("| :--- | :---: | :---: |\n")
        total = max(1, len(items))
        for st in ["approved", "cannot_read", "not_plate", "pending"]:
            cnt = status_counts.get(st, 0)
            f.write(f"| **{st.capitalize()}** | {cnt:,} | {cnt/total*100:.1f}% |\n")
        f.write(f"| **Total Images** | {total:,} | 100.0% |\n\n")
        
        f.write("## 2. Approved Plates by Vehicle / Plate Type\n\n")
        f.write("| Plate Type | Count | Percentage |\n")
        f.write("| :--- | :---: | :---: |\n")
        for pt, cnt in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {pt} | {cnt} | {cnt/max(1, len(approved_items))*100:.1f}% |\n")
            
        f.write("\n## 3. Approved Plates by Province (ISO 3166-2:TH)\n\n")
        f.write("| Province (ISO 3166-2:TH) | Count |\n")
        f.write("| :--- | :---: |\n")
        for prov, cnt in sorted(province_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"| {prov} | {cnt} |\n")
            
    print(f"[OK] Generated {summary_md}")
    return True

def restore_all():
    in_plates_session = ANNOTATIONS_DIR / "plates_session.json"
    if not in_plates_session.exists():
        print(f"[ERROR] No annotations found at {in_plates_session}")
        return False
        
    DATASET_PLATES_SESSION.parent.mkdir(parents=True, exist_ok=True)
    with open(in_plates_session, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(DATASET_PLATES_SESSION, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Restored session to {DATASET_PLATES_SESSION}")
    
    in_root_session = ANNOTATIONS_DIR / "curated_lpr_session.json"
    if in_root_session.exists():
        DATASET_ROOT_SESSION.parent.mkdir(parents=True, exist_ok=True)
        with open(in_root_session, "r", encoding="utf-8") as f:
            rdata = json.load(f)
        with open(DATASET_ROOT_SESSION, "w", encoding="utf-8") as f:
            json.dump(rdata, f, ensure_ascii=False, indent=2)
        print(f"[OK] Restored session to {DATASET_ROOT_SESSION}")
        
    return True

def main():
    parser = argparse.ArgumentParser(description="Export or restore LPR annotations.")
    parser.add_argument("--restore", action="store_true", help="Restore annotations into dataset directory")
    args = parser.parse_args()
    
    if args.restore:
        restore_all()
    else:
        export_all()

if __name__ == "__main__":
    main()
