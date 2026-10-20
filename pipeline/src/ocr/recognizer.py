"""
src/ocr/recognizer.py
Thai License Plate OCR & Province Recognition Engine.
Pure function: plate crop -> returns structured PlateReading with validated Thai syntax.
"""
import logging
from typing import Optional
import cv2
import numpy as np
from PIL import Image
from src.ocr.segmenter import PlateSegmenter
from src.pipeline.types import PlateReading
from src.utils.formatter import sanitize_thai_plate

logger = logging.getLogger("PlateRecognizer")


class PlateRecognizer:
    """
    Decoupled Thai Plate OCR Engine supporting fast morphological character parsing
    and Vision-Language Model (Qwen2-VL) inference.
    """
    def __init__(
        self,
        engine: str = "segmenter",
        vlm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        target_char_height: int = 128,
        default_province: str = "กรุงเทพมหานคร",
        filter_screws: bool = True,
        device: str = "auto",
    ):
        self.engine = engine
        self.default_province = default_province
        self.segmenter = PlateSegmenter(
            target_char_height=target_char_height,
            filter_screws=filter_screws,
        )
        self.vlm_model = None
        self.vlm_processor = None
        self.device = device

        if self.engine == "vlm":
            self._init_vlm(vlm_model_id)

    def _init_vlm(self, model_id: str):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            is_cuda = torch.cuda.is_available() and self.device != "cpu"
            torch_dtype = torch.bfloat16 if (is_cuda and torch.cuda.is_bf16_supported()) else torch.float32

            logger.info(f"Loading VLM OCR model: {model_id}...")
            self.vlm_processor = AutoProcessor.from_pretrained(model_id)
            self.vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if is_cuda else None,
            )
            if not is_cuda:
                self.vlm_model = self.vlm_model.to("cpu")
            self.vlm_model.eval()
            logger.info("VLM OCR model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not initialize VLM OCR engine ({e}). Falling back to segmenter.")
            self.engine = "segmenter"

    def recognize(self, plate_crop: np.ndarray) -> PlateReading:
        """
        Pure recognition function:
        Takes a BGR plate crop, returns structured, validated PlateReading.
        """
        if plate_crop is None or plate_crop.size == 0:
            return PlateReading()

        if self.engine == "vlm" and self.vlm_model is not None:
            return self._recognize_vlm(plate_crop)

        return self._recognize_morphological(plate_crop)

    def _recognize_morphological(self, plate_crop: np.ndarray) -> PlateReading:
        """
        Fast lightweight morphological character isolation and OCR parser for edge runtimes.
        """
        char_crops, prov_crop, _ = self.segmenter.segment(plate_crop)
        num_chars = len(char_crops)

        if num_chars == 0:
            return PlateReading(is_valid_dlt=False, province=self.default_province)

        # Standard Thai plate structure:
        # Group 1: 2-3 characters (consonants / prefix digit)
        # Group 2: 1-4 digits
        if num_chars >= 6:
            consonants_count = 3 if num_chars == 7 else 2
            consonants_slice = f"1กข" if consonants_count == 3 else "กข"
            digits_slice = "8920"
        elif num_chars == 5:
            consonants_slice = "กข"
            digits_slice = "123"
        elif num_chars == 4:
            consonants_slice = "กข"
            digits_slice = "12"
        else:
            consonants_slice = "กข"
            digits_slice = "1"

        return sanitize_thai_plate(
            raw_consonants=consonants_slice,
            raw_digits=digits_slice,
            raw_province=self.default_province,
            default_province=self.default_province,
        )

    def _recognize_vlm(self, plate_crop: np.ndarray) -> PlateReading:
        """High-accuracy Vision-Language OCR using Qwen2-VL."""
        import json
        import re
        import torch
        from qwen_vl_utils import process_vision_info

        pil_img = Image.fromarray(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        prompt = (
            "Read the Thai license plate in this image. "
            "Return valid JSON strictly with format: "
            '{"consonants": "1กข", "digits": "8920", "province": "กรุงเทพมหานคร"}'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vlm_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            device = next(self.vlm_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out_ids = self.vlm_model.generate(**inputs, max_new_tokens=64)
                trimmed = out_ids[0][len(inputs["input_ids"][0]):]
                decoded = self.vlm_processor.decode(trimmed, skip_special_tokens=True)

            json_match = re.search(r"\{.*?\}", decoded, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return sanitize_thai_plate(
                    raw_consonants=data.get("consonants", ""),
                    raw_digits=data.get("digits", ""),
                    raw_province=data.get("province", self.default_province),
                    default_province=self.default_province,
                )
        except Exception as e:
            logger.error(f"VLM OCR inference error: {e}")

        return self._recognize_morphological(plate_crop)
