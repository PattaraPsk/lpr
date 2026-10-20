"""
List of 77 provinces of Thailand and special annotation codes.
Standardized according to ISO 3166-2:TH and Unicode NFC (ISO/IEC 10646).
"""

import unicodedata
from typing import Optional, Tuple, List, Dict

# Special codes as requested:
# + for not license plate
# - for cannot read
SPECIAL_OPTIONS = [
    ("+", "+ (Not License Plate / ไม่ใช่ป้ายทะเบียน)"),
    ("-", "- (Cannot Read / อ่านไม่ออก)"),
]

# 77 Thailand Provinces with English romanization and ISO 3166-2:TH codes
# Format: (Thai Name, English Romanization, ISO 3166-2:TH Code)
PROVINCES_ISO_DATA: List[Tuple[str, str, str]] = [
    ("กรุงเทพมหานคร", "Bangkok", "TH-10"),
    ("สมุทรปราการ", "Samut Prakan", "TH-11"),
    ("นนทบุรี", "Nonthaburi", "TH-12"),
    ("ปทุมธานี", "Pathum Thani", "TH-13"),
    ("พระนครศรีอยุธยา", "Phra Nakhon Si Ayutthaya", "TH-14"),
    ("อ่างทอง", "Ang Thong", "TH-15"),
    ("ลพบุรี", "Lopburi", "TH-16"),
    ("สิงห์บุรี", "Sing Buri", "TH-17"),
    ("ชัยนาท", "Chai Nat", "TH-18"),
    ("สระบุรี", "Saraburi", "TH-19"),
    ("ชลบุรี", "Chonburi", "TH-20"),
    ("ระยอง", "Rayong", "TH-21"),
    ("จันทบุรี", "Chanthaburi", "TH-22"),
    ("ตราด", "Trat", "TH-23"),
    ("ฉะเชิงเทรา", "Chachoengsao", "TH-24"),
    ("ปราจีนบุรี", "Prachinburi", "TH-25"),
    ("นครนายก", "Nakhon Nayok", "TH-26"),
    ("สระแก้ว", "Sa Kaeo", "TH-27"),
    ("นครราชสีมา", "Nakhon Ratchasima", "TH-30"),
    ("บุรีรัมย์", "Buriram", "TH-31"),
    ("สุรินทร์", "Surin", "TH-32"),
    ("ศรีสะเกษ", "Sisaket", "TH-33"),
    ("อุบลราชธานี", "Ubon Ratchathani", "TH-34"),
    ("ยโสธร", "Yasothon", "TH-35"),
    ("ชัยภูมิ", "Chaiyaphum", "TH-36"),
    ("อำนาจเจริญ", "Amnat Charoen", "TH-37"),
    ("บึงกาฬ", "Bueng Kan", "TH-38"),
    ("หนองบัวลำภู", "Nong Bua Lamphu", "TH-39"),
    ("ขอนแก่น", "Khon Kaen", "TH-40"),
    ("อุดรธานี", "Udon Thani", "TH-41"),
    ("เลย", "Loei", "TH-42"),
    ("หนองคาย", "Nong Khai", "TH-43"),
    ("มหาสารคาม", "Maha Sarakham", "TH-44"),
    ("ร้อยเอ็ด", "Roi Et", "TH-45"),
    ("กาฬสินธุ์", "Kalasin", "TH-46"),
    ("สกลนคร", "Sakon Nakhon", "TH-47"),
    ("นครพนม", "Nakhon Phanom", "TH-48"),
    ("มุกดาหาร", "Mukdahan", "TH-49"),
    ("เชียงใหม่", "Chiang Mai", "TH-50"),
    ("ลำพูน", "Lamphun", "TH-51"),
    ("ลำปาง", "Lampang", "TH-52"),
    ("อุตรดิตถ์", "Uttaradit", "TH-53"),
    ("แพร่", "Phrae", "TH-54"),
    ("น่าน", "Nan", "TH-55"),
    ("พะเยา", "Phayao", "TH-56"),
    ("เชียงราย", "Chiang Rai", "TH-57"),
    ("แม่ฮ่องสอน", "Mae Hong Son", "TH-58"),
    ("นครสวรรค์", "Nakhon Sawan", "TH-60"),
    ("อุทัยธานี", "Uthai Thani", "TH-61"),
    ("กำแพงเพชร", "Kamphaeng Phet", "TH-62"),
    ("ตาก", "Tak", "TH-63"),
    ("สุโขทัย", "Sukhothai", "TH-64"),
    ("พิษณุโลก", "Phitsanulok", "TH-65"),
    ("พิจิตร", "Phichit", "TH-66"),
    ("เพชรบูรณ์", "Phetchabun", "TH-67"),
    ("ราชบุรี", "Ratchaburi", "TH-70"),
    ("กาญจนบุรี", "Kanchanaburi", "TH-71"),
    ("สุพรรณบุรี", "Suphan Buri", "TH-72"),
    ("นครปฐม", "Nakhon Pathom", "TH-73"),
    ("สมุทรสาคร", "Samut Sakhon", "TH-74"),
    ("สมุทรสงคราม", "Samut Songkhram", "TH-75"),
    ("เพชรบุรี", "Phetchaburi", "TH-76"),
    ("ประจวบคีรีขันธ์", "Prachuap Khiri Khan", "TH-77"),
    ("นครศรีธรรมราช", "Nakhon Si Thammarat", "TH-80"),
    ("กระบี่", "Krabi", "TH-81"),
    ("พังงา", "Phang Nga", "TH-82"),
    ("ภูเก็ต", "Phuket", "TH-83"),
    ("สุราษฎร์ธานี", "Surat Thani", "TH-84"),
    ("ระนอง", "Ranong", "TH-85"),
    ("ชุมพร", "Chumphon", "TH-86"),
    ("สงขลา", "Songkhla", "TH-90"),
    ("สตูล", "Satun", "TH-91"),
    ("ตรัง", "Trang", "TH-92"),
    ("พัทลุง", "Phatthalung", "TH-93"),
    ("ปัตตานี", "Pattani", "TH-94"),
    ("ยะลา", "Yala", "TH-95"),
    ("นราธิวาส", "Narathiwat", "TH-96"),
]

# Backward compatibility 2-tuple list
PROVINCES_DATA: List[Tuple[str, str]] = [
    (th, en) for th, en, _ in PROVINCES_ISO_DATA
]

# Fast lookups
ISO_TO_PROVINCE: Dict[str, Tuple[str, str]] = {
    iso: (th, en) for th, en, iso in PROVINCES_ISO_DATA
}

NAME_TO_ISO: Dict[str, str] = {}
for th, en, iso in PROVINCES_ISO_DATA:
    NAME_TO_ISO[unicodedata.normalize("NFC", th)] = iso
    NAME_TO_ISO[en.lower()] = iso


def normalize_thai_script(text: str) -> str:
    """
    Normalizes Thai text into Canonical Composition (Unicode NFC, ISO/IEC 10646).
    Eliminates decomposed tone marks, vowel ordering variations, and trailing whitespaces.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", str(text).strip())


def get_province_entries() -> List[Tuple[str, str]]:
    """
    Returns list of tuples: (actual_value, display_text)
    Special options are placed at the very top.
    Includes ISO 3166-2:TH code in display for global clarity.
    """
    entries = []
    # Special entries
    for val, disp in SPECIAL_OPTIONS:
        entries.append((val, disp))

    # Standard provinces with ISO codes
    for th, en, iso in PROVINCES_ISO_DATA:
        entries.append((th, f"{th} ({en}) [{iso}]"))

    return entries


def get_iso_code(province_name: str) -> Optional[str]:
    """
    Returns the ISO 3166-2:TH code for a given Thai or English province name.
    """
    norm = normalize_thai_script(province_name)
    if norm in NAME_TO_ISO:
        return NAME_TO_ISO[norm]
    if norm.lower() in NAME_TO_ISO:
        return NAME_TO_ISO[norm.lower()]
    return None


def get_province_by_iso(code: str) -> Optional[Tuple[str, str]]:
    """
    Returns (Thai Name, English Name) for a given ISO 3166-2:TH code (e.g. 'TH-10').
    """
    code_upper = code.strip().upper()
    return ISO_TO_PROVINCE.get(code_upper)


def match_province_value(text: str) -> str:
    """
    Given an input string (from filename or user input),
    returns the best matching canonical Thai province name.
    Normalized to Unicode NFC.
    """
    if not text:
        return ""
    text_clean = normalize_thai_script(text)

    if text_clean == "+" or "not" in text_clean.lower():
        return "+"
    if text_clean == "-" or "cannot" in text_clean.lower():
        return "-"

    # Check direct ISO code match (e.g. user typed TH-10)
    iso_match = get_province_by_iso(text_clean)
    if iso_match:
        return iso_match[0]

    for th, en, _ in PROVINCES_ISO_DATA:
        th_norm = normalize_thai_script(th)
        if text_clean.lower() == th_norm.lower() or text_clean.lower() == en.lower():
            return th_norm
        if th_norm in text_clean:
            return th_norm

    return text_clean
