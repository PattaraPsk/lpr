"""
Thai License Plate Recognition (ALPR/ANPR) Production Package
"""

__version__ = "2.0.0"
__author__ = "Senior Edge Computer Vision Architect"

from src.pipeline.types import BoundingBox, PlateReading, LPRResult
from src.pipeline.lpr_pipeline import LPRPipeline

__all__ = ["BoundingBox", "PlateReading", "LPRResult", "LPRPipeline"]
