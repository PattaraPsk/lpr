"""
Structured Data Types and Schemas for selen-lpr.
Defines API contracts, normalized scale coordinates, and DLT vehicle classifications.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class NormalizedCoordinates:
    """
    Scale-normalized bounding box coordinates relative to the full original frame [0.0, 1.0].
    Formula:
        s_x = (x_roi + x_plate) / frame_width
        s_y = (y_roi + y_plate) / frame_height
        s_w = w_plate / frame_width
        s_h = h_plate / frame_height
    """
    s_x: float
    s_y: float
    s_w: float
    s_h: float

    def to_list(self) -> List[float]:
        return [round(self.s_x, 4), round(self.s_y, 4), round(self.s_w, 4), round(self.s_h, 4)]


@dataclass
class PlateDesign:
    """Visual design characteristics of the license plate."""
    background_color: str   # "white", "yellow", "orange", "green"
    text_color: str         # "black", "green", "blue", "white"
    layout: str             # "standard_single_row" or "motorcycle_double_row"


@dataclass
class VehicleType:
    """Classified vehicle type mapped from plate design and letter prefix."""
    category_id: str        # e.g., "private_car", "private_pickup", "public_taxi"
    description: str        # e.g., "Private Pickups (2-door light commercial trucks)"


@dataclass
class PlateReading:
    """Structured alphanumeric reading of the license plate."""
    d3: str                 # Front category (1 digit + 2 Thai consonants e.g. "1กข" or "กข")
    d4: str                 # Numeric registration digits (1 to 4 digits e.g. "8920")
    province: str           # Official Thai province name e.g. "กรุงเทพมหานคร"
    full_plate: str         # Formatted complete plate text e.g. "1กข 8920 กรุงเทพมหานคร"


@dataclass
class DetectionResult:
    """Master output entity for each detected and classified license plate."""
    normalized_coords: NormalizedCoordinates
    confidence: float
    plate_design: PlateDesign
    vehicle_type: VehicleType
    reading: PlateReading
    crop_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "s_x": round(self.normalized_coords.s_x, 4),
            "s_y": round(self.normalized_coords.s_y, 4),
            "s_w": round(self.normalized_coords.s_w, 4),
            "s_h": round(self.normalized_coords.s_h, 4),
            "d3": self.reading.d3,
            "d4": self.reading.d4,
            "province": self.reading.province,
            "vehicle_category": self.vehicle_type.category_id,
            "vehicle_description": self.vehicle_type.description,
            "background_color": self.plate_design.background_color,
            "text_color": self.plate_design.text_color,
            "confidence": round(self.confidence, 4),
            "crop_path": self.crop_path
        }
