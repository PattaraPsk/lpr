"""
Official Thai Department of Land Transport (DLT) Registration Rules & Mappings.
Provides legal mapping between plate color, consonant prefixes, and vehicle types.
"""

from typing import Tuple, Optional, Dict, Any

# 8 Category Definitions and their Legal Rules
DLT_RULES: Dict[str, Dict[str, Any]] = {
    "private_car": {
        "description": "Private Cars (Sedans, SUVs, Hatches up to 7 seats)",
        "bg_color": "white",
        "text_color": "black",
        "legal_consonants": set("กขงจฉชฌญฎฆธฐพภวศษส"),
        "aspect_ratio": "standard"
    },
    "private_pickup": {
        "description": "Private Pickups (2-door light commercial trucks)",
        "bg_color": "white",
        "text_color": "green",
        "legal_consonants": set("ฒณตถบปผยรล"),
        "aspect_ratio": "standard"
    },
    "private_van": {
        "description": "Private Vans / Microbuses (Vehicles with more than 7 seats)",
        "bg_color": "white",
        "text_color": "blue",
        "legal_consonants": set("นฬอฮ"),
        "aspect_ratio": "standard"
    },
    "public_taxi": {
        "description": "Public Taxis (Standard commercial & cooperative taxis)",
        "bg_color": "yellow",
        "text_color": "black",
        "legal_consonants": set("ทม"),
        "aspect_ratio": "standard"
    },
    "tuktuk": {
        "description": "Tuk-Tuks (Commercial three-wheelers)",
        "bg_color": "yellow",
        "text_color": "green",
        "legal_consonants": set("ศส"),
        "valid_prefixes": {"ศข", "สก", "สข"},
        "aspect_ratio": "standard"
    },
    "trailer_tractor": {
        "description": "Trailers & Tractors (Construction & Agricultural)",
        "bg_color": "orange",
        "text_color": "black",
        "legal_consonants": set("ซฆ"),
        "aspect_ratio": "standard"
    },
    "service_vehicle": {
        "description": "Service Vehicles (Airport / Tour / Hotel transport)",
        "bg_color": "green",
        "text_color": "white",
        "legal_consonants": set("ฌณ"),
        "aspect_ratio": "standard"
    },
    "motorcycle": {
        "description": "Motorcycles (Two-wheeled motor vehicles)",
        "bg_color": "white",
        "text_color": "black",
        "legal_consonants": set("กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ123456789"),
        "aspect_ratio": "motorcycle"
    }
}


def map_plate_to_vehicle_type(
    bg_color: str,
    text_color: str,
    d3_prefix: str,
    aspect_ratio: float
) -> Tuple[str, str]:
    """
    Fuses visual color, letter prefix, and geometry to determine vehicle type.
    Returns: (category_id, description)
    """
    bg = bg_color.lower().strip()
    text = text_color.lower().strip()
    prefix = d3_prefix.strip()

    # 1. Geometry Check: Motorcycle (Square 2-tier aspect ratio ~ 1.0 to 1.35)
    if aspect_ratio < 1.45:
        return "motorcycle", DLT_RULES["motorcycle"]["description"]

    # Extract Thai consonants from prefix (stripping leading digits like '1' in '1กข')
    consonants = "".join([ch for ch in prefix if "\u0e01" <= ch <= "\u0e2e"])

    # 2. Orange Background -> Trailer / Tractor
    if bg == "orange":
        return "trailer_tractor", DLT_RULES["trailer_tractor"]["description"]

    # 3. Green Background -> Service Vehicle (Airport / Tour)
    if bg == "green":
        return "service_vehicle", DLT_RULES["service_vehicle"]["description"]

    # 4. Yellow Background:
    if bg == "yellow":
        if text == "green" or prefix in DLT_RULES["tuktuk"].get("valid_prefixes", set()):
            return "tuktuk", DLT_RULES["tuktuk"]["description"]
        return "public_taxi", DLT_RULES["public_taxi"]["description"]

    # 5. White Background Family (Differentiated by Text Color & Consonant Rules)
    if bg == "white":
        # Check text color first
        if text == "green":
            return "private_pickup", DLT_RULES["private_pickup"]["description"]
        if text == "blue":
            return "private_van", DLT_RULES["private_van"]["description"]
        
        # If text color is ambiguous (e.g. shadowed or night lighting), use consonant rules:
        if consonants:
            first_c = consonants[0]
            if first_c in DLT_RULES["private_pickup"]["legal_consonants"]:
                return "private_pickup", DLT_RULES["private_pickup"]["description"]
            if first_c in DLT_RULES["private_van"]["legal_consonants"]:
                return "private_van", DLT_RULES["private_van"]["description"]
            if first_c in DLT_RULES["private_car"]["legal_consonants"]:
                return "private_car", DLT_RULES["private_car"]["description"]

        return "private_car", DLT_RULES["private_car"]["description"]

    # Fallback to private car
    return "private_car", DLT_RULES["private_car"]["description"]
