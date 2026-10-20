"""
Plate Design & Palette Classifier.
Determines background color, text color, and layout format.
"""

from typing import Tuple
import numpy as np

from ..schema.types import PlateDesign
from ..utils.color_extractor import extract_plate_colors


class PlateDesignClassifier:
    """
    Classifies visual plate design using color palette extraction.
    """

    def classify(self, plate_crop: np.ndarray) -> PlateDesign:
        h, w = plate_crop.shape[:2]
        aspect_ratio = w / max(h, 1)

        # Classify color palette
        bg_color, text_color = extract_plate_colors(plate_crop)

        # Classify geometry layout
        layout = "motorcycle_double_row" if aspect_ratio < 1.45 else "standard_single_row"

        return PlateDesign(
            background_color=bg_color,
            text_color=text_color,
            layout=layout
        )
