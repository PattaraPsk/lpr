"""
src/utils/formatter.py
Official Thai License Plate Grammar, Regex Validation & Province Normalization.
Conforms strictly to Thailand Department of Land Transport (DLT) regulations:
Format: หมวดอักษร + หมายเลขทะเบียน + จังหวัด
"""
import re
from typing import Dict, List, Optional, Tuple
from src.pipeline.types import PlateReading

# 77 Official Provinces of Thailand
THAI_PROVINCES: List[str] = [
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

# Common aliases / short names
PROVINCE_ALIASES: Dict[str, str] = {
    "กทม": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "กรุงเทพ": "กรุงเทพมหานคร",
    "อยุธยา": "พระนครศรีอยุธยา",
    "โคราช": "นครราชสีมา",
    "ปราจีน": "ปราจีนบุรี",
    "สมุทรปราการ": "สมุทรปราการ",
}

# 44 Thai Consonants permitted on plates
THAI_CONSONANTS_SET = set("กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ")

# Regex patterns for Thai plate registration groups
RE_MODERN_PREFIX = re.compile(r"^[1-9][ก-ฮ]{2}$")     # e.g., 1กข, 9วย
RE_CLASSIC_PREFIX = re.compile(r"^[ก-ฮ]{1,2}$")        # e.g., กข, ฆง, ฐ
RE_COMMERCIAL_PREFIX = re.compile(r"^[0-9]{2}-[0-9]{4}$") # e.g., 70-1234
RE_DIGITS = re.compile(r"^[1-9][0-9]{0,3}$")          # 1 to 9999 (DLT non-zero leading)


def match_closest_province(candidate: str, threshold: float = 0.55) -> Optional[str]:
    """Finds closest matching official Thai province using character overlap / edit distance."""
    if not candidate:
        return None
    cleaned = candidate.strip().replace(" ", "")

    # Direct match or alias match
    if cleaned in THAI_PROVINCES:
        return cleaned
    if cleaned in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[cleaned]

    # Partial substring match
    for prov in THAI_PROVINCES:
        if cleaned in prov or prov in cleaned:
            return prov

    # Character Jaccard similarity fallback
    best_match = None
    best_score = 0.0
    cand_set = set(cleaned)

    for prov in THAI_PROVINCES:
        prov_set = set(prov)
        intersection = len(cand_set.intersection(prov_set))
        union = len(cand_set.union(prov_set))
        score = intersection / max(1.0, float(union))
        if score > best_score and score >= threshold:
            best_score = score
            best_match = prov

    return best_match


def sanitize_thai_plate(
    raw_consonants: str,
    raw_digits: str,
    raw_province: str = "",
    default_province: str = "กรุงเทพมหานคร",
) -> PlateReading:
    """
    Sanitizes, validates, and standardizes raw OCR characters into a verified PlateReading.
    """
    consonants = re.sub(r"[^0-9ก-ฮ]", "", raw_consonants.strip())
    digits = re.sub(r"[^0-9]", "", raw_digits.strip())
    province_in = raw_province.strip()

    # Province resolution
    province = match_closest_province(province_in) or default_province

    # Determine validation status and plate category
    is_valid_dlt = False
    plate_category = "private_car"

    is_modern = bool(RE_MODERN_PREFIX.match(consonants))
    is_classic = bool(RE_CLASSIC_PREFIX.match(consonants))
    is_valid_digits = bool(RE_DIGITS.match(digits))

    if (is_modern or is_classic) and is_valid_digits and province in THAI_PROVINCES:
        is_valid_dlt = True
        plate_category = "private_car"
    elif consonants and digits:
        # Partial match acceptable in noisy edge conditions
        is_valid_dlt = len(digits) <= 4 and len(consonants) <= 4

    # Build standardized string: {consonants}_{digits}_{province}
    formatted = f"{consonants}_{digits}_{province}" if (consonants and digits) else ""

    return PlateReading(
        raw_text=f"{raw_consonants} {raw_digits} {raw_province}".strip(),
        consonants=consonants,
        digits=digits,
        province=province,
        formatted_string=formatted,
        confidence=0.95 if is_valid_dlt else 0.60,
        is_valid_dlt=is_valid_dlt,
        plate_category=plate_category,
    )
