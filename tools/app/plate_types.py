"""
Thailand License Plate Types definition, codes, and descriptions.
Uses Uppercase separation (PascalCase) for plate type IDs (e.g. WhiteBlackCar, YellowBlue).
"""

# Special codes
SPECIAL_PLATE_TYPES = [
    ("+", "+ (Not License Plate)"),
    ("-", "- (Cannot Read)"),
]

# Standard Thai Plate Types based on Department of Land Transport (DLT)
# Uses Uppercase separation (PascalCase)
PLATE_TYPES_DATA = [
    {
        "id": "WhiteBlackCar",
        "aliases": ["white_black_car", "wb_car", "whiteblackcar", "white-black-car", "car", "passenger_car", "white_black", "whiteblack", "wb", "1"],
        "display": "White / Black - Car (รถยนต์นั่งส่วนบุคคล ≤ 7 ที่นั่ง)",
        "desc": "White Background with Black Letters: Private passenger cars (7 seats or fewer)",
        "badge_text": "CAR",
        "bg_color": "#ffffff",
        "text_color": "#000000",
        "border_color": "#000000"
    },
    {
        "id": "WhiteBlackMotor",
        "aliases": ["white_black_motor", "wb_motor", "whiteblackmotor", "white-black-motor", "white_black_motorcycle", "motor", "motorcycle", "bike"],
        "display": "White / Black - Motor (รถจักรยานยนต์ส่วนบุคคล)",
        "desc": "White Background with Black Letters: Standard private motorcycles",
        "badge_text": "MC",
        "bg_color": "#ffffff",
        "text_color": "#000000",
        "border_color": "#000000"
    },
    {
        "id": "WhiteBlue",
        "aliases": ["white_blue", "whiteblue", "wbl", "white-blue", "white_bg_blue_text", "2"],
        "display": "White / Blue (รถยนต์นั่งส่วนบุคคลเกิน 7 ที่นั่ง / รถตู้)",
        "desc": "White Background with Blue Letters: Small buses, vans, or passenger vehicles carrying > 7 seats",
        "badge_text": "VAN",
        "bg_color": "#ffffff",
        "text_color": "#1d4ed8",
        "border_color": "#1d4ed8"
    },
    {
        "id": "WhiteGreen",
        "aliases": ["white_green", "whitegreen", "wg", "white-green", "white_bg_green_text", "3"],
        "display": "White / Green (รถกระบะบรรทุกส่วนบุคคล)",
        "desc": "White Background with Green Letters: Private pickup trucks or registered commercial personal trucks",
        "badge_text": "TRK",
        "bg_color": "#ffffff",
        "text_color": "#15803d",
        "border_color": "#15803d"
    },
    {
        "id": "YellowBlack",
        "aliases": ["yellow_black", "yellowblack", "yb", "yellow-black", "yellow_bg_black_text", "4"],
        "display": "Yellow / Black (แท็กซี่ / มอเตอร์ไซค์รับจ้าง)",
        "desc": "Yellow Background with Black Letters: Commercial hire vehicles, urban taxis, motorcycle taxis",
        "badge_text": "TAXI",
        "bg_color": "#facc15",
        "text_color": "#000000",
        "border_color": "#000000"
    },
    {
        "id": "YellowBlue",
        "aliases": ["yellow_blue", "yellowblue", "ybl", "yellow-blue", "yellow_bg_blue_text", "5"],
        "display": "Yellow / Blue (รถสองแถว / รถโดยสารประจำทาง)",
        "desc": "Yellow Background with Blue Letters: Fixed-route local public buses or songthaews (shared trucks)",
        "badge_text": "BUS",
        "bg_color": "#facc15",
        "text_color": "#1d4ed8",
        "border_color": "#1d4ed8"
    },
    {
        "id": "YellowGreen",
        "aliases": ["yellow_green", "yellowgreen", "yg", "yellow-green", "yellow_bg_green_text", "6"],
        "display": "Yellow / Green (รถสามล้อรับจ้าง / ตุ๊กตุ๊ก)",
        "desc": "Yellow Background with Green Letters: Public tuk-tuks",
        "badge_text": "TUK",
        "bg_color": "#facc15",
        "text_color": "#15803d",
        "border_color": "#15803d"
    },
    {
        "id": "GreenWhite",
        "aliases": ["green_white", "greenwhite", "gw", "green-white", "green_bg_white_text", "7"],
        "display": "Green / White (รถบริการธุรกิจ / รถทัศนาจร / ลีมูซีน)",
        "desc": "Green Background with White Letters: Tour, hotel, airport, or limousine services",
        "badge_text": "LIMO",
        "bg_color": "#15803d",
        "text_color": "#ffffff",
        "border_color": "#ffffff"
    },
    {
        "id": "RedBlack",
        "aliases": ["red_black", "redblack", "rb", "red-black", "red_bg_black_text", "red_white", "8"],
        "display": "Red / Black-White (ป้ายแดง / รถใหม่ระหว่างจดทะเบียน)",
        "desc": "Red Background with Black/White Letters: Temporary or dealer registration plates",
        "badge_text": "RED",
        "bg_color": "#dc2626",
        "text_color": "#ffffff",
        "border_color": "#000000"
    },
    {
        "id": "OrangeBlack",
        "aliases": ["orange_black", "orangeblack", "ob", "orange-black", "orange_bg_black_text", "9"],
        "display": "Orange / Black (รถแทรกเตอร์ / เครื่องจักรกลการเกษตร)",
        "desc": "Orange Background with Black Letters: Agricultural vehicles, tractors, and heavy construction rollers",
        "badge_text": "AGR",
        "bg_color": "#ea580c",
        "text_color": "#000000",
        "border_color": "#000000"
    },
]

def get_plate_type_entries():
    """Returns list of (id, display_label) tuples including + and -"""
    entries = []
    for val, disp in SPECIAL_PLATE_TYPES:
        entries.append((val, disp))
    for pt in PLATE_TYPES_DATA:
        entries.append((pt["id"], pt["display"]))
    return entries

def match_plate_type(text: str) -> str:
    """Matches text from filename to standard PascalCase plate type id, or returns input as-is."""
    if not text:
        return ""
    t = text.strip()
    if t == "+":
        return "+"
    if t == "-":
        return "-"
    
    # Normalize input by removing underscores, dashes, spaces for flexible matching
    t_clean = t.lower().replace("_", "").replace("-", "").replace(" ", "")
    
    # 1. Exact match against id (case-insensitive without separator)
    for pt in PLATE_TYPES_DATA:
        if t_clean == pt["id"].lower():
            return pt["id"]
            
    # 2. Check aliases
    for pt in PLATE_TYPES_DATA:
        for alias in pt["aliases"]:
            alias_clean = alias.lower().replace("_", "").replace("-", "").replace(" ", "")
            if t_clean == alias_clean:
                return pt["id"]
                
    # 3. Substring match
    for pt in PLATE_TYPES_DATA:
        pt_clean = pt["id"].lower()
        if pt_clean in t_clean or t_clean in pt_clean:
            return pt["id"]
            
    return t
