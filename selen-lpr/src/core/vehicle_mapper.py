"""
Vehicle Type Mapper for selen-lpr.
Fuses visual plate design with OCR prefix consonants to classify the vehicle.
"""

from ..schema.types import PlateDesign, VehicleType
from ..utils.dlt_rules import map_plate_to_vehicle_type


class VehicleTypeMapper:
    """
    Maps visual plate design and registration prefix to the 8 DLT vehicle types.
    """

    def map_vehicle(
        self,
        design: PlateDesign,
        d3_prefix: str,
        aspect_ratio: float
    ) -> VehicleType:
        cat_id, desc = map_plate_to_vehicle_type(
            bg_color=design.background_color,
            text_color=design.text_color,
            d3_prefix=d3_prefix,
            aspect_ratio=aspect_ratio
        )
        return VehicleType(category_id=cat_id, description=desc)
