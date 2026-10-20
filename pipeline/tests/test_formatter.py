"""
tests/test_formatter.py
Unit tests for Thai license plate syntax validation and province normalization.
"""
import unittest
from src.utils.formatter import sanitize_thai_plate, match_closest_province, THAI_PROVINCES


class TestThaiFormatter(unittest.TestCase):
    def test_modern_registration_format(self):
        reading = sanitize_thai_plate("1กข", "8920", "กรุงเทพมหานคร")
        self.assertTrue(reading.is_valid_dlt)
        self.assertEqual(reading.consonants, "1กข")
        self.assertEqual(reading.digits, "8920")
        self.assertEqual(reading.province, "กรุงเทพมหานคร")
        self.assertEqual(reading.formatted_string, "1กข_8920_กรุงเทพมหานคร")

    def test_classic_registration_format(self):
        reading = sanitize_thai_plate("กข", "1234", "เชียงใหม่")
        self.assertTrue(reading.is_valid_dlt)
        self.assertEqual(reading.consonants, "กข")
        self.assertEqual(reading.digits, "1234")
        self.assertEqual(reading.province, "เชียงใหม่")

    def test_province_fuzzy_matching(self):
        # Alias resolution
        self.assertEqual(match_closest_province("กทม."), "กรุงเทพมหานคร")
        self.assertEqual(match_closest_province("อยุธยา"), "พระนครศรีอยุธยา")
        self.assertEqual(match_closest_province("โคราช"), "นครราชสีมา")

        # Substring / noisy OCR resolution
        self.assertEqual(match_closest_province("เชียงใหม"), "เชียงใหม่")
        self.assertEqual(match_closest_province("ภูเก็ต"), "ภูเก็ต")

    def test_invalid_syntax_rejection(self):
        reading = sanitize_thai_plate("XYZ", "ABCD", "UnknownPlace")
        self.assertFalse(reading.is_valid_dlt)

    def test_77_provinces_count(self):
        self.assertEqual(len(THAI_PROVINCES), 77)


if __name__ == "__main__":
    unittest.main()
