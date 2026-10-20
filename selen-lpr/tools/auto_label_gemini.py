"""
Auto-Label Thai License Plates using Gemini Flash Multimodal API.

This tool inspects Thai license plate images and extracts structured,
DLT-compliant ground-truth annotations (prefix, digits, province, vehicle type, plate color).
Zero third-party library dependencies (uses Python standard library: urllib, json, base64).
"""

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List

# Official 77 Provinces of Thailand according to DLT regulations
VALID_PROVINCES = {
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
}

PROMPT_TEMPLATE = """You are an expert Thai Department of Land Transport (DLT) license plate verification specialist.
Analyze this Thai vehicle license plate image with maximum visual precision.

Return ONLY a valid JSON object with the following schema:
{
  "is_valid_plate": true/false,
  "3digit": "Thai prefix consonants or alphanumeric characters (e.g., 'กต', '1กณ', 'ขข', '71', 'บท', 'ฮ')",
  "4digit": "License digits (1 to 9999, e.g. '910', '1899', '27', '0638')",
  "province": "Exact official Thai province name in Thai script (e.g. 'น่าน', 'เชียงใหม่', 'กรุงเทพมหานคร')",
  "vehicle_type": "sedan | pickup | van | motorcycle | truck | bus | suv",
  "plate_color": "white_black | white_green | white_blue | yellow_black | red_black",
  "category": "private_car | private_pickup | private_van | commercial_transport | temporary_plate | motorcycle",
  "rejection_reason": "none or describe why if not a real plate (e.g. price ad, parking sign)"
}

Important Rules:
- Distinguish easily confused Thai characters: ฉ vs ษ, ณ vs ม, ฐ vs ส, ฎ/ฏ vs ช.
- Ensure the province matches one of the 77 official Thai provinces.
- If it is not a genuine vehicle plate (e.g., digital ad graphic, parking sign, price tag), set "is_valid_plate": false.
"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def label_plate_image(image_path: str, api_key: str, model: str = "gemini-2.5-flash") -> Optional[Dict[str, Any]]:
    """Calls Gemini REST API using standard urllib to annotate a plate image."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    img_b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": PROMPT_TEMPLATE},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_b64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.0
        }
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidate = data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(candidate)
            return result
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"[ERROR] API HTTP error {e.code} for {image_path}: {err_msg}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path}: {e}", file=sys.stderr)
        return None


def run_batch_labeling(input_dir: str, output_csv: str, detailed_csv: Optional[str] = None, api_key: Optional[str] = None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY environment variable or --api-key argument is required.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg")))

    if not image_files:
        print(f"[WARNING] No images found in {input_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images in {input_dir}. Beginning annotation...")

    records = []
    detailed_records = []

    for idx, img_p in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Annotating {img_p.name}...", end=" ", flush=True)
        res = label_plate_image(str(img_p), api_key)
        if res and res.get("is_valid_plate", True):
            d3 = res.get("3digit", "").strip()
            d4 = res.get("4digit", "").strip()
            prov = res.get("province", "").strip()
            v_type = res.get("vehicle_type", "sedan")
            p_color = res.get("plate_color", "white_black")
            cat = res.get("category", "private_car")

            print(f"OK -> '{d3} {d4} {prov}' ({cat})")
            records.append((img_p.name, d3, d4, prov))
            if detailed_csv:
                full_plate = f"{d3} {d4} {prov}".strip()
                detailed_records.append((img_p.name, d3, d4, prov, full_plate, v_type, p_color, "standard_rectangular", cat, "gemini_flash_auto"))
        else:
            reason = res.get("rejection_reason", "Invalid/Rejected") if res else "API Failure"
            print(f"REJECTED: {reason}")
        
        time.sleep(0.5)

    # Write standard benchmark CSV
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("IMAGE_NAME,3DIGIT,4DIGIT,PROVINCE\n")
        for row in records:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]}\n")

    print(f"[SUCCESS] Saved {len(records)} valid annotations to {output_csv}")

    # Write detailed CSV if requested
    if detailed_csv and detailed_records:
        with open(detailed_csv, "w", encoding="utf-8") as f:
            f.write("IMAGE_NAME,3DIGIT,4DIGIT,PROVINCE,FULL_PLATE,VEHICLE_TYPE,PLATE_COLOR,PLATE_DESIGN,CATEGORY,SOURCE\n")
            for row in detailed_records:
                f.write(",".join(row) + "\n")
        print(f"[SUCCESS] Saved detailed metadata to {detailed_csv}")


def main():
    parser = argparse.ArgumentParser(description="Auto-Label Thai License Plates using Gemini Flash")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing plate images")
    parser.add_argument("--output-csv", type=str, default="labels.csv", help="Output path for standard benchmark CSV")
    parser.add_argument("--detailed-csv", type=str, default=None, help="Output path for detailed metadata CSV")
    parser.add_argument("--api-key", type=str, default=None, help="Gemini API Key (or set GEMINI_API_KEY env)")
    args = parser.parse_args()

    run_batch_labeling(args.input_dir, args.output_csv, args.detailed_csv, args.api_key)


if __name__ == "__main__":
    main()
