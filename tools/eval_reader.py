import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

import torch
from tools.DataManipulator.app.plate_reader import PlateReader

def parse_gt(p: Path):
    parts = p.stem.split("_")
    if len(parts) >= 5 and parts[-1] and parts[-2] and parts[-3]:
        d3 = parts[-3]
        d4 = parts[-2]
        prov = parts[-1]
    else:
        folder_parts = p.parent.name.split("_")
        if len(folder_parts) >= 4:
            d3 = folder_parts[-3]
            d4 = folder_parts[-2]
            prov = folder_parts[-1]
        elif len(folder_parts) == 3:
            d3 = folder_parts[0]
            d4 = folder_parts[1]
            prov = folder_parts[2]
        else:
            d3 = parts[-3] if len(parts) >= 3 else ""
            d4 = parts[-2] if len(parts) >= 2 else ""
            prov = parts[-1] if len(parts) >= 1 else ""

    if d3 in ["+", "-"]: d3 = ""
    if d4 in ["+", "-"]: d4 = ""
    if prov in ["+", "-"]: prov = ""

    # Consonant clean
    d3 = d3.replace("ฃ", "ข").replace("ฅ", "ค")
    return d3, d4, prov

def evaluate():
    new_weights = "runs/train/reader/best_reader.pth"
    base_weights = "models/reader/thai_plate_reader/best_norm_ED.pth"

    print("=" * 75)
    print("        THAI LICENSE PLATE OCR EVALUATION (ACCURACY & GENERALIZATION)")
    print(f"Baseline Checkpoint: {base_weights}")
    print(f"New Checkpoint     : {new_weights}")
    print("=" * 75)

    reader_base = PlateReader(weights_path=base_weights)
    reader_new = PlateReader(weights_path=new_weights)

    # 1. Ready Ground Truth Samples
    ready_dir = Path("dataset/ready/202610261140")
    all_ready = [
        p for p in ready_dir.rglob("*.jpg")
        if "+_+" not in p.as_posix() and "-_-" not in p.as_posix()
    ]
    # Pick a distinct sample of 20 plates across folders
    ready_samples = []
    seen_dirs = set()
    for p in all_ready:
        if p.parent.name not in seen_dirs:
            ready_samples.append(p)
            seen_dirs.add(p.parent.name)
        if len(ready_samples) >= 15:
            break

    print(f"\n--- [EVAL 1] Ready Ground Truth Verification ({len(ready_samples)} distinct real plates) ---")
    base_correct = 0
    new_correct = 0
    for p in ready_samples:
        gt_d3, gt_d4, gt_prov = parse_gt(p)
        gt_full = f"{gt_d3} {gt_d4} {gt_prov}".strip()

        res_b = reader_base.predict(str(p), use_cache=False)
        res_n = reader_new.predict(str(p), use_cache=False)

        ocr_b = res_b.get("full_plate", "")
        ocr_n = res_n.get("full_plate", "")

        is_b = (res_b.get("d3") == gt_d3 and res_b.get("d4") == gt_d4 and res_b.get("province") == gt_prov)
        is_n = (res_n.get("d3") == gt_d3 and res_n.get("d4") == gt_d4 and res_n.get("province") == gt_prov)
        if is_b: base_correct += 1
        if is_n: new_correct += 1

        print(f"Sample: {p.parent.name[:28]:<28}")
        print(f"  GT   : [{gt_full}]")
        print(f"  Base : [{ocr_b:<25}] -> {'MATCH' if is_b else 'DIFF'}")
        print(f"  New  : [{ocr_n:<25}] -> {'MATCH' if is_n else 'DIFF'}")

    print(f"\n>> Real Ready Accuracy: Baseline = {base_correct}/{len(ready_samples)} ({base_correct/len(ready_samples)*100:.1f}%) | "
          f"New = {new_correct}/{len(ready_samples)} ({new_correct/len(ready_samples)*100:.1f}%)")

    # 2. CCTV Synthetic Challenging Samples (Perspective warp, blur, dirty, motorcycle, graphic)
    synth_dir = Path("dataset/dataset-cctv-synthetic/plates")
    all_synth = list(synth_dir.glob("*.jpg"))
    # Pick 20 diverse synthetic plates (sampling with stride)
    stride = max(1, len(all_synth) // 20)
    synth_samples = all_synth[::stride][:20]

    print(f"\n--- [EVAL 2] CCTV Synthetic Edge Benchmark ({len(synth_samples)} edge samples) ---")
    base_synth_correct = 0
    new_synth_correct = 0
    for p in synth_samples:
        gt_d3, gt_d4, gt_prov = parse_gt(p)
        gt_full = f"{gt_d3} {gt_d4} {gt_prov}".strip()

        res_b = reader_base.predict(str(p), use_cache=False)
        res_n = reader_new.predict(str(p), use_cache=False)

        ocr_b = res_b.get("full_plate", "")
        ocr_n = res_n.get("full_plate", "")

        is_b = (res_b.get("d3") == gt_d3 and res_b.get("d4") == gt_d4 and res_b.get("province") == gt_prov)
        is_n = (res_n.get("d3") == gt_d3 and res_n.get("d4") == gt_d4 and res_n.get("province") == gt_prov)
        if is_b: base_synth_correct += 1
        if is_n: new_synth_correct += 1

        print(f"Sample: {p.stem[:32]:<32}")
        print(f"  GT   : [{gt_full}]")
        print(f"  Base : [{ocr_b:<25}] -> {'MATCH' if is_b else 'DIFF'}")
        print(f"  New  : [{ocr_n:<25}] -> {'MATCH' if is_n else 'DIFF'}")

    print(f"\n>> CCTV Synth Accuracy: Baseline = {base_synth_correct}/{len(synth_samples)} ({base_synth_correct/len(synth_samples)*100:.1f}%) | "
          f"New = {new_synth_correct}/{len(synth_samples)} ({new_synth_correct/len(synth_samples)*100:.1f}%)")

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY:")
    print(f"  Total Verified Ready Real Plates Sampled: {len(ready_samples)}")
    print(f"    - Baseline Exact Match: {base_correct}/{len(ready_samples)}")
    print(f"    - New Best Exact Match: {new_correct}/{len(ready_samples)}")
    print(f"  Total CCTV Synthetic Benchmark Sampled: {len(synth_samples)}")
    print(f"    - Baseline Exact Match: {base_synth_correct}/{len(synth_samples)}")
    print(f"    - New Best Exact Match: {new_synth_correct}/{len(synth_samples)}")
    print("=" * 75)

if __name__ == "__main__":
    evaluate()
