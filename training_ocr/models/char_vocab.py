"""
Thai License Plate OCR Vocabulary Definition.
Includes Thai consonants, vowels, tone marks, digits, and special delimiters.
"""

# Thai Consonants (44 letters)
THAI_CONSONANTS = [
    "ก", "ข", "ฃ", "ค", "ฅ", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ", "ฎ", "ฏ",
    "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท", "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ",
    "ฟ", "ภ", "ม", "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
]

# Digits 0-9
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Vowels and tone marks commonly seen on Thai plates / provinces
THAI_VOWELS_TONES = [
    "ะ", "ั", "า", "ำ", "ิ", "ี", "ึ", "ื", "ุ", "ู", "ฺ",
    "เ", "แ", "โ", "ใ", "ไ", "ๅ", "็", "่", "้", "๊", "๋", "์"
]

# Special tokens
BLANK_TOKEN = "<blank>"   # For CTC loss
SPACE_TOKEN = " "
UNKNOWN_TOKEN = "<unk>"

# Complete Vocabulary List
VOCAB = [BLANK_TOKEN] + DIGITS + THAI_CONSONANTS + THAI_VOWELS_TONES + [SPACE_TOKEN, UNKNOWN_TOKEN]

CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(VOCAB)}
IDX_TO_CHAR = {idx: ch for idx, ch in enumerate(VOCAB)}


def encode_text(text: str) -> list[int]:
    """Converts a Thai plate string into a sequence of token indices."""
    return [CHAR_TO_IDX.get(ch, CHAR_TO_IDX[UNKNOWN_TOKEN]) for ch in text]


def decode_indices(indices: list[int], collapse_ctc: bool = True) -> str:
    """Decodes token indices back into a string with optional CTC duplicate collapse."""
    if not collapse_ctc:
        return "".join([IDX_TO_CHAR.get(idx, "") for idx in indices if idx != 0])
        
    chars = []
    prev_idx = -1
    for idx in indices:
        if idx == 0:  # Blank token in CTC
            prev_idx = -1
            continue
        if idx != prev_idx:
            char = IDX_TO_CHAR.get(idx, "")
            if char not in (BLANK_TOKEN, UNKNOWN_TOKEN):
                chars.append(char)
        prev_idx = idx
    return "".join(chars)


def get_vocab_size() -> int:
    return len(VOCAB)
