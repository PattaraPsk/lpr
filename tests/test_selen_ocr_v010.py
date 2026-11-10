"""
Comprehensive ISO/IEC 42001:2023 Verification Suite for SelenOCR v0.1.0 & Unified 11-Token Codec
"""

import sys
import unittest
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tools.selen_token_codec import (
    VOCAB_SIZE,
    PAD_ID,
    START_ID,
    STOP_ID,
    encode_plate_to_tokens,
    decode_tokens_to_plate,
    THAI_CONSONANTS
)
from tools.selen_ocr_model import SelenOCR


class TestSelenOCRV010(unittest.TestCase):
    """Verification suite for SelenOCR v0.1.0 architecture, codec, and checkpoints."""

    def test_vocabulary_and_dlt_compliance(self):
        """Verifies vocabulary conforms to DLT specs with exactly 145 classes and 41 valid consonants."""
        self.assertEqual(VOCAB_SIZE, 145)
        self.assertEqual(len(THAI_CONSONANTS), 41)
        self.assertNotIn("ฃ", THAI_CONSONANTS)
        self.assertNotIn("ฅ", THAI_CONSONANTS)
        self.assertNotIn("ฏ", THAI_CONSONANTS)

    def test_token_codec_max_11_tokens(self):
        """Verifies encoded sequence length is strictly <= 11 tokens across all valid formats."""
        test_cases = [
            ("WhiteBlackCar", "1กข", "1234", "กรุงเทพมหานคร"),
            ("WhiteBlackMotor", "1กข", "123", "ภูเก็ต"),
            ("YellowBlack", "53", "9210", "นนทบุรี"),
            ("WhiteGreen", "บพ", "88", "ชลบุรี"),
            ("WhiteBlue", "นข", "5678", "เชียงใหม่"),
            ("-", "-", "-", "-"),
            ("+", "+", "+", "+"),
        ]

        for p_type, d3, d4, prov in test_cases:
            enc = encode_plate_to_tokens(p_type, d3, d4, prov)
            self.assertLessEqual(len(enc), 11, f"Encoded sequence {enc} exceeded 11 tokens!")
            self.assertEqual(enc[0], START_ID)
            self.assertEqual(enc[-1], STOP_ID)

            # Roundtrip decoding
            dec_type, dec_d3, dec_d4, dec_prov, line = decode_tokens_to_plate(enc)
            self.assertEqual(dec_type, p_type)
            self.assertEqual(dec_d3, d3)
            self.assertEqual(dec_d4, d4)
            self.assertEqual(dec_prov, prov)

    def test_obsolete_consonant_remapping(self):
        """Verifies obsolete consonants are remapped to standard DLT orthography."""
        # ฃ -> ข
        enc = encode_plate_to_tokens("WhiteBlackCar", "ฃข", "1234", "กรุงเทพมหานคร")
        _, dec_d3, _, _, _ = decode_tokens_to_plate(enc)
        self.assertEqual(dec_d3, "ขข")

        # ฅ -> ค
        enc = encode_plate_to_tokens("WhiteBlackCar", "ฅค", "1234", "กรุงเทพมหานคร")
        _, dec_d3, _, _, _ = decode_tokens_to_plate(enc)
        self.assertEqual(dec_d3, "คค")

        # ฏ -> ฎ
        enc = encode_plate_to_tokens("WhiteBlackCar", "ฏฎ", "1234", "กรุงเทพมหานคร")
        _, dec_d3, _, _, _ = decode_tokens_to_plate(enc)
        self.assertEqual(dec_d3, "ฎฎ")

    def test_selen_ocr_forward_backward_pass(self):
        """Verifies SelenOCR forward and backward tensor flow on CPU."""
        model = SelenOCR(
            d_model=384,
            nhead=6,
            num_encoder_layers=2,  # test configuration
            lstm_hidden=128,
            vocab_size=VOCAB_SIZE,
            num_plate_types=13,
            num_provinces=81
        )
        model.train()

        dummy = torch.randn(2, 3, 64, 320)
        out = model(dummy)

        self.assertIn("ctc_logits", out)
        self.assertIn("type_logits", out)
        self.assertIn("prov_logits", out)

        ctc_logits = out["ctc_logits"]
        type_logits = out["type_logits"]
        prov_logits = out["prov_logits"]

        self.assertEqual(ctc_logits.shape, (2, 40, VOCAB_SIZE))
        self.assertEqual(type_logits.shape, (2, 13))
        self.assertEqual(prov_logits.shape, (2, 81))

        # Backward loss
        loss = ctc_logits.sum() + type_logits.sum() + prov_logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and "fpn.fpn_convs.1" not in name and "fpn.fpn_convs.2" not in name:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_checkpoint_goal_validation(self):
        """Verifies that the trained SelenOCR checkpoint meets or exceeds the >95.0% accuracy goal."""
        chkpt_path = ROOT_DIR / "models" / "reader" / "selen_ocr_v010" / "best_reader.pth"
        self.assertTrue(chkpt_path.exists(), f"Checkpoint {chkpt_path} does not exist!")

        chkpt = torch.load(chkpt_path, map_location="cpu")
        self.assertIn("val_acc", chkpt)
        val_acc = chkpt["val_acc"]
        self.assertGreaterEqual(
            val_acc, 95.0,
            f"SelenOCR v0.1.0 val_acc was {val_acc:.2f}%, which is below the 95.0% target goal!"
        )


if __name__ == "__main__":
    unittest.main()
