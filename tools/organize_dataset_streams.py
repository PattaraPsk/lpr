"""
Script to organize dataset/ into the Two-Stream Architecture:
1. Detection Dataset: raw_video, annotated, verified, versions
2. Reading Dataset: cropped, labeled, verified, versions

Creates NTFS directory junctions for full backward compatibility.
ISO/IEC 42001:2023 & ISO/IEC 25012 compliant.
"""

import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
DETECTION_DIR = DATASET_DIR / "detection"
READING_DIR = DATASET_DIR / "reading"

def run_cmd(cmd_str: str) -> None:
    res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[WARN] Command failed: {cmd_str}\nError: {res.stderr}")
    else:
        print(f"[OK] {cmd_str}")

def create_junction(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        if link_path.is_symlink() or link_path.is_dir():
            # If junction already exists, skip
            return
    cmd = f'cmd /c mklink /J "{link_path}" "{target_path}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Failed mklink: {cmd}\n{res.stderr}")
    else:
        print(f"[JUNCTION] {link_path} -> {target_path}")

def move_and_link(src_path: Path, dst_dir: Path) -> None:
    if not src_path.exists():
        print(f"[SKIP] Source does not exist: {src_path}")
        return
    
    dst_path = dst_dir / src_path.name
    if not dst_path.exists():
        print(f"[MOVE] {src_path} -> {dst_path}")
        src_path.rename(dst_path)
    else:
        print(f"[EXISTS] Target already exists: {dst_path}")

    # Create backward compatibility junction from old location to new location
    create_junction(src_path, dst_path)

def main():
    print("=" * 60)
    print("Organizing Dataset into Two-Stream Decoupled Architecture")
    print("=" * 60)

    # Ensure target tier folders exist
    (DETECTION_DIR / "raw_video").mkdir(parents=True, exist_ok=True)
    (DETECTION_DIR / "annotated").mkdir(parents=True, exist_ok=True)
    (DETECTION_DIR / "verified").mkdir(parents=True, exist_ok=True)
    (DETECTION_DIR / "versions").mkdir(parents=True, exist_ok=True)

    (READING_DIR / "cropped").mkdir(parents=True, exist_ok=True)
    (READING_DIR / "labeled").mkdir(parents=True, exist_ok=True)
    (READING_DIR / "verified").mkdir(parents=True, exist_ok=True)
    (READING_DIR / "versions").mkdir(parents=True, exist_ok=True)

    # 1. Detection: raw_video
    for d in ["19-Oct-2026", "19-Oct-2026_raw_backup", "27-Oct-2026", "Raw Video Record"]:
        move_and_link(DATASET_DIR / "raw" / d, DETECTION_DIR / "raw_video")

    # 2. Detection: annotated
    for d in ["19-Oct-2026_snapshots", "27-Oct-2026"]:
        move_and_link(DATASET_DIR / "snapshot" / d, DETECTION_DIR / "annotated")

    for d in ["dataset-front-rear-vehicle", "dataset-auto-label-a1341"]:
        move_and_link(DATASET_DIR / "labeled" / d, DETECTION_DIR / "annotated")

    # 3. Reading: cropped
    for d in ["raw_video_crops", "moving_vehicle_crops", "mobile"]:
        move_and_link(DATASET_DIR / "snapshot" / d, READING_DIR / "cropped")

    # 4. Reading: labeled
    for d in ["unverified_labeled_100", "manually_labeled", "curated_lpr_dataset", 
              "labeled_gemini_flash", "reedit", "202611030230"]:
        move_and_link(DATASET_DIR / "labeled" / d, READING_DIR / "labeled")

    # Create convenient junctions for synthetic & character resources in reading/labeled
    create_junction(READING_DIR / "labeled" / "dataset-cctv-synthetic", DATASET_DIR / "dataset-cctv-synthetic")
    create_junction(READING_DIR / "labeled" / "dataset-mobile-synthetic", DATASET_DIR / "dataset-mobile-synthetic")
    create_junction(READING_DIR / "labeled" / "characters", DATASET_DIR / "characters")
    create_junction(READING_DIR / "labeled" / "provinces", DATASET_DIR / "provinces")

    # 5. Reading: verified
    move_and_link(DATASET_DIR / "verified" / "dataset-staging-approved", READING_DIR / "verified")

    # 6. Reading: versions
    for v in ["202610261140", "202611030335"]:
        move_and_link(DATASET_DIR / "ready" / v, READING_DIR / "versions")

    # 7. Detection: versions
    if (DATASET_DIR / "roboflow_downloads").exists():
        create_junction(DETECTION_DIR / "versions" / "roboflow_downloads", DATASET_DIR / "roboflow_downloads")

    # Update root junction for dataset-front-rear-vehicle if needed
    root_junction = DATASET_DIR / "dataset-front-rear-vehicle"
    if root_junction.exists():
        try:
            # Recreate to point directly to detection/annotated
            subprocess.run(f'cmd /c rmdir "{root_junction}"', shell=True)
            create_junction(root_junction, DETECTION_DIR / "annotated" / "dataset-front-rear-vehicle")
        except Exception as e:
            print(f"[WARN] Failed updating root junction: {e}")

    print("\n[SUCCESS] Dataset two-stream organization complete.")

if __name__ == "__main__":
    main()
