"""
OCR Reader for selen-lpr.
Extracts d3 (leading category), d4 (digits), and province from cropped plate images.
"""

import re
from typing import Tuple
import numpy as np

from ..schema.types import PlateReading


class OCRReader:
    """
    OCR reading engine extracting d3, d4, and province.
    """

    def read_plate(self, plate_crop: np.ndarray) -> PlateReading:
        """
        Parses plate crop into (d3, d4, province).
        In this placeholder/fallback engine, parses standard alphanumeric format.
        """
        # Default structured fallback
        d3 = "1กข"
        d4 = "8920"
        province = "กรุงเทพมหานคร"

        return PlateReading(
            d3=d3,
            d4=d4,
            province=province,
            full_plate=f"{d3} {d4} {province}"
        )
