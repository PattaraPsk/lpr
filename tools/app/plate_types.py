"""
Thailand License Plate Types definition, codes, and descriptions.
"""

# Special codes
SPECIAL_PLATE_TYPES = [
    ("+", "+ (Not License Plate)"),
    ("-", "- (Cannot Read)"),
]

# Standard Thai Plate Types based on Department of Land Transport (DLT)
PLATE_TYPES_DATA = [
    {
        "id": "white_black",
        "aliases": ["wb", "white_black", "white-black", "white_bg_black_text", "1"],
        "display": "White / Black (รถยนต์นั่งส่วนบุคคล ≤ 7 ที่นั่ง / มอเตอร์ไซค์)",
        "desc": "White Background with Black Letters: Private passenger cars (≤ 7 seats) & standard motorcycles",
        "bg_color": "#ffffff",
        "text_color": "#000000",
        "border_color": "#000000"
    },
    {
        "id": "white_blue",
        "aliases": ["wbl", "white_blue", "white-blue", "white_bg_blue_text", "2"],
        "display": "White / Blue (รถยนต์นั่งส่วนบุคคลเกิน 7 ที่นั่ง / รถตู้)",
        "desc": "White Background with Blue Letters: Small buses, vans, or passenger vehicles carrying > 7 seats",
        "bg_color": "#ffffff",
        "text_color": "#1d4ed8",
        "border_color": "#1d4ed8"
    },
    {
        "id": "white_green",
        "aliases": ["wg", "white_green", "white-green", "white_bg_green_text", "3"],
        "display": "White / Green (รถกระบะบรรทุกส่วนบุคคล)",
        "desc": "White Background with Green Letters: Private pickup trucks or registered commercial personal trucks",
        "bg_color": "#ffffff",
        "text_color": "#15803d",
        "border_color": "#15803d"
    },
    {
        "id": "yellow_black",
        "aliases": ["yb", "yellow_black", "yellow-black", "yellow_bg_black_text", "4"],
        "display": "Yellow / Black (แท็กซี่ / มอเตอร์ไซค์รับจ้าง)",
        "desc": "Yellow Background with Black Letters: Commercial hire vehicles, urban taxis, motorcycle taxis",
        "bg_color": "#facc15",
        "text_color": "#000000",
        "border_color": "#000000"
    },
    {
        "id": "yellow_blue",
        "aliases": ["ybl", "yellow_blue", "yellow-blue", "yellow_bg_blue_text", "5"],
        "display": "Yellow / Blue (รถสองแถว / รถโดยสารประจำทาง)",
        "desc": "Yellow Background with Blue Letters: Fixed-route local public buses or songthaews (shared trucks)",
        "bg_color": "#facc15",
        "text_color": "#1d4ed8",
        "border_color": "#1d4ed8"
    },
    {
        "id": "yellow_green",
        "aliases": ["yg", "yellow_green", "yellow-green", "yellow_bg_green_text", "6"],
        "display": "Yellow / Green (รถสามล้อรับจ้าง / ตุ๊กตุ๊ก)",
        "desc": "Yellow Background with Green Letters: Public tuk-tuks",
        "bg_color": "#facc15",
        "text_color": "#15803d",
        "border_color": "#15803d"
    },
    {
        "id": "green_white",
        "aliases": ["gw", "green_white", "green-white", "green_bg_white_text", "7"],
        "display": "Green / White (รถบริการธุรกิจ / รถทัศนาจร / ลีมูซีน)",
        "desc": "Green Background with White Letters: Tour, hotel, airport, or limousine services",
        "bg_color": "#15803d",
        "text_color": "#ffffff",
        "border_color": "#ffffff"
    },
    {
        "id": "red_black",
        "aliases": ["rb", "red_black", "red-black", "red_bg_black_text", "red_white", "8"],
        "display": "Red / Black-White (ป้ายแดง / รถใหม่ระหว่างจดทะเบียน)",
        "desc": "Red Background with Black/White Letters: Temporary or dealer registration plates",
        "bg_color": "#dc2626",
        "text_color": "#ffffff",
        "border_color": "#000000"
    },
    {
        "id": "orange_black",
        "aliases": ["ob", "orange_black", "orange-black", "orange_bg_black_text", "9"],
        "display": "Orange / Black (รถแทรกเตอร์ / เครื่องจักรกลการเกษตร)",
        "desc": "Orange Background with Black Letters: Agricultural vehicles, tractors, and heavy construction rollers",
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
    """Matches text from filename to standard plate type id, or returns input as-is."""
    if not text:
        return ""
    t = text.strip()
    if t == "+":
        return "+"
    if t == "-":
        return "-"
    
    t_lower = t.lower().replace(" ", "_").replace("-", "_")
    for pt in PLATE_TYPES_DATA:
        if t_lower == pt["id"].lower():
            return pt["id"]
        for alias in pt["aliases"]:
            if t_lower == alias.lower():
                return pt["id"]
        if pt["id"] in t_lower or t_lower in pt["id"]:
            return pt["id"]
            
    return t
