import unittest
from pathlib import Path
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.vision_attention_ocr import VisionAttentionOCR
from src.models.vocab import decode_raw_prediction, DECODING_DICT


class TestPurePyTorchOCR(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.weights_path = Path(__file__).resolve().parent.parent / "models" / "ocr" / "thai_plate_attn_ocr.pt"
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = VisionAttentionOCR(num_classes=187)
        if cls.weights_path.exists():
            cls.model.load_state_dict(torch.load(str(cls.weights_path), map_location=cls.device))
        cls.model.to(cls.device)
        cls.model.eval()

    def test_forward_pass_dummy(self):
        # Input shape: [1, 1, 64, 600]
        dummy_input = torch.randn(1, 1, 64, 600).to(self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        # Output shape: [1, batch_max_length + 1, num_classes]
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[2], 187)

    def test_vocab_decoding(self):
        from src.models.vocab import CHARACTERS
        bkk_idx = CHARACTERS.index('¢') + 2
        test_pred_indices = [
            CHARACTERS.index('1') + 2,
            CHARACTERS.index('.') + 2,   # '.' -> 'ก'
            CHARACTERS.index('ü') + 2,   # 'ü' -> 'ข'
            60,                          # '_' delimiter
            CHARACTERS.index('8') + 2,
            CHARACTERS.index('9') + 2,
            CHARACTERS.index('2') + 2,
            CHARACTERS.index('0') + 2,
            60,                          # '_' delimiter
            bkk_idx,                     # '¢' -> 'กรุงเทพมหานคร'
            1,                           # [s] EOS token
        ]
        d3, d4, province, full_plate = decode_raw_prediction(test_pred_indices, vehicle_category="Private Car")
        self.assertTrue(len(d3) > 0)
        self.assertTrue(len(d4) > 0)
        self.assertEqual(province, "กรุงเทพมหานคร")


if __name__ == "__main__":
    unittest.main()
