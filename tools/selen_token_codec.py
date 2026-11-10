#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Token Codec for Thai License Plate Recognition (ISO/IEC 42001:2023 & ISO/IEC 5259-2:2023)

Implements the single-token representation requested:
- Output line: "plateType d3 d4 province"
- Max sequence length: start + plateType + 3 + 4 + 1 + stop = 11 tokens
- Classes consist of:
  - start, stop, pad (blank)
  - digits '0'-'9' (10)
  - 41 Thai consonants
  - 11 plate types (encoded as single tokens)
  - 78 provinces (encoded as single tokens)
  - '+' (not plate at all)
  - '-' (cannot read / blur)
"""

import sys
import unicodedata
from typing import List, Tuple, Dict, Optional, Any

# Ensure UTF-8 standard output on Windows systems
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 1. 41 Valid Thai Consonants (excluding obsolete ฃ, ฅ, ฏ)
THAI_CONSONANTS = "กขคฆงจฉชซฌญฎฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ"

# 2. 11 Plate Types (Standard DLT)
PLATE_TYPES = [
    "WhiteBlackCar",
    "WhiteBlackMotor",
    "WhiteBlue",
    "WhiteGreen",
    "YellowBlack",
    "Other",
    "YellowBlue",
    "YellowGreen",
    "GreenWhite",
    "RedBlack",
    "OrangeBlack",
]
PLATE_TYPE_TO_IDX = {t: i for i, t in enumerate(PLATE_TYPES)}
IDX_TO_PLATE_TYPE = {i: t for i, t in enumerate(PLATE_TYPES)}

# 3. 78 Jurisdictions (77 provinces + Betong)
PROVINCES_78 = [
    "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา",
    "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี", "ชลบุรี", "ระยอง",
    "จันทบุรี", "ตราด", "ฉะเชิงเทรา", "ปราจีนบุรี", "นครนายก", "สระแก้ว",
    "นครราชสีมา", "บุรีรัมย์", "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร",
    "ชัยภูมิ", "อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี",
    "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์", "สกลนคร",
    "นครพนม", "มุกดาหาร", "เชียงใหม่", "ลำพูน", "ลำปาง", "อุตรดิตถ์", "แพร่",
    "น่าน", "พะเยา", "เชียงราย", "แม่ฮ่องสอน", "นครสวรรค์", "อุทัยธานี",
    "กำแพงเพชร", "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์", "ราชบุรี",
    "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม", "สมุทรสาคร", "สมุทรสงคราม", "เพชรบุรี",
    "ประจวบคีรีขันธ์", "นครศรีธรรมราช", "กระบี่", "พังงา", "ภูเก็ต", "สุราษฎร์ธานี",
    "ระนอง", "ชุมพร", "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี", "ยะลา",
    "นราธิวาส", "เบตง"
]
PROV_TO_IDX = {p: i for i, p in enumerate(PROVINCES_78)}
IDX_TO_PROV = {i: p for i, p in enumerate(PROVINCES_78)}

# Special tokens
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
STOP_TOKEN = "<stop>"
PLUS_TOKEN = "+"
MINUS_TOKEN = "-"

# Build Vocabulary:
# 0: <pad> (CTC blank)
# 1: <start>
# 2: <stop>
# 3: '+' (Not plate)
# 4: '-' (Cannot read)
# 5..14: '0'..'9' (10)
# 15..55: 41 Thai consonants
# 56..66: 11 Plate Types (as distinct tokens)
# 67..144: 78 Provinces (as distinct tokens)

VOCABULARY: List[str] = [
    PAD_TOKEN,
    START_TOKEN,
    STOP_TOKEN,
    PLUS_TOKEN,
    MINUS_TOKEN,
] + [str(d) for d in range(10)] + list(THAI_CONSONANTS) + [f"<type_{t}>" for t in PLATE_TYPES] + [f"<prov_{p}>" for p in PROVINCES_78]

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCABULARY)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)  # 145

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
START_ID = TOKEN_TO_ID[START_TOKEN]
STOP_ID = TOKEN_TO_ID[STOP_TOKEN]
PLUS_ID = TOKEN_TO_ID[PLUS_TOKEN]
MINUS_ID = TOKEN_TO_ID[MINUS_TOKEN]


OCR_REMAP = {"ฃ": "ข", "ฅ": "ค", "ฏ": "ฎ"}


def encode_plate_to_tokens(
    plate_type: str,
    d3: str,
    d4: str,
    province: str,
    include_start_stop: bool = True
) -> List[int]:
    """
    Encode plate attributes into a sequence of token IDs.
    Max output length: start + plateType + 3 + 4 + 1 + stop = 11 tokens.
    """
    tokens = []
    if include_start_stop:
        tokens.append(START_ID)

    # 1. Negative conditions
    if plate_type == "+" or d3 == "+" or d4 == "+" or province == "+":
        # Not a plate at all: '+' '+' '+' '+'
        tokens.extend([PLUS_ID, PLUS_ID, PLUS_ID, PLUS_ID])
        if include_start_stop:
            tokens.append(STOP_ID)
        return tokens

    if plate_type == "-" or d3 == "-" or d4 == "-" or province == "-":
        # Cannot read blur: '-' '-' '-' '-'
        tokens.extend([MINUS_ID, MINUS_ID, MINUS_ID, MINUS_ID])
        if include_start_stop:
            tokens.append(STOP_ID)
        return tokens

    # 2. Plate Type Token
    type_tok = f"<type_{plate_type}>"
    if type_tok in TOKEN_TO_ID:
        tokens.append(TOKEN_TO_ID[type_tok])
    else:
        tokens.append(TOKEN_TO_ID["<type_Other>"])

    # 3. d3 (max 3 characters)
    for ch in str(d3)[:3]:
        ch = OCR_REMAP.get(ch, ch)
        if ch in TOKEN_TO_ID:
            tokens.append(TOKEN_TO_ID[ch])

    # 4. d4 (max 4 characters)
    for ch in str(d4)[:4]:
        ch = OCR_REMAP.get(ch, ch)
        if ch in TOKEN_TO_ID:
            tokens.append(TOKEN_TO_ID[ch])

    # 5. Province Token (1 token)
    norm_prov = unicodedata.normalize("NFC", province.strip())
    prov_tok = f"<prov_{norm_prov}>"
    if prov_tok in TOKEN_TO_ID:
        tokens.append(TOKEN_TO_ID[prov_tok])
    else:
        # Default to Bangkok if unknown
        tokens.append(TOKEN_TO_ID["<prov_กรุงเทพมหานคร>"])

    if include_start_stop:
        tokens.append(STOP_ID)

    return tokens


def decode_tokens_to_plate(token_ids: List[int]) -> Tuple[str, str, str, str, str]:
    """
    Decode a sequence of token IDs into (plate_type, d3, d4, province, full_text_line).
    Output line: "plateType d3 d4 province"
    """
    # Strip pad, start, stop
    filtered = [
        tid for tid in token_ids
        if tid not in (PAD_ID, START_ID, STOP_ID)
    ]
    if not filtered:
        return "-", "-", "-", "-", "- - - -"

    # Check for '+' or '-'
    if any(tid == PLUS_ID for tid in filtered):
        return "+", "+", "+", "+", "+ + + +"
    if any(tid == MINUS_ID for tid in filtered):
        return "-", "-", "-", "-", "- - - -"

    plate_type = "WhiteBlackCar"
    d3_chars = []
    d4_chars = []
    province = "กรุงเทพมหานคร"

    # Identify tokens
    type_idx = -1
    prov_idx = -1

    for i, tid in enumerate(filtered):
        tok_str = ID_TO_TOKEN.get(tid, "")
        if tok_str.startswith("<type_"):
            plate_type = tok_str[6:-1]
            type_idx = i
            break

    for i in range(len(filtered) - 1, -1, -1):
        tok_str = ID_TO_TOKEN.get(filtered[i], "")
        if tok_str.startswith("<prov_"):
            province = tok_str[6:-1]
            prov_idx = i
            break

    # Middle tokens are alphanumeric d3 + d4
    mid_start = type_idx + 1 if type_idx != -1 else 0
    mid_end = prov_idx if prov_idx != -1 else len(filtered)
    alphanumeric_tokens = filtered[mid_start:mid_end]

    alphanumeric_str = "".join(
        ID_TO_TOKEN.get(tid, "") for tid in alphanumeric_tokens
        if not ID_TO_TOKEN.get(tid, "").startswith("<")
    )

    # Split alphanumeric_str into d3 and d4
    # Split at the last Thai consonant
    last_thai_idx = -1
    for i, ch in enumerate(alphanumeric_str):
        if "\u0e01" <= ch <= "\u0e2e":
            last_thai_idx = i

    if last_thai_idx != -1:
        d3 = alphanumeric_str[:last_thai_idx + 1][:3]
        d4 = alphanumeric_str[last_thai_idx + 1:][:4]
    else:
        # Commercial numeric plate (e.g. '70 1234')
        if len(alphanumeric_str) > 4:
            d3 = alphanumeric_str[:-4][:3]
            d4 = alphanumeric_str[-4:][:4]
        else:
            d3 = ""
            d4 = alphanumeric_str[:4]

    full_text_line = f"{plate_type} {d3} {d4} {province}".strip()
    return plate_type, d3, d4, province, full_text_line


if __name__ == "__main__":
    # Self-test
    test_cases = [
        ("WhiteBlackCar", "1กข", "1234", "กรุงเทพมหานคร"),
        ("WhiteBlackMotor", "1กข", "4160", "นครราชสีมา"),
        ("YellowBlack", "70", "8825", "ชลบุรี"),
        ("WhiteBlue", "ฮฮ", "9999", "เชียงใหม่"),
        ("WhiteGreen", "1ฒณ", "5500", "ภูเก็ต"),
        ("-", "-", "-", "-"),
        ("+", "+", "+", "+"),
    ]

    print(f"Vocabulary Size: {VOCAB_SIZE}")
    for p_type, d3, d4, prov in test_cases:
        encoded = encode_plate_to_tokens(p_type, d3, d4, prov)
        decoded_type, dec_d3, dec_d4, dec_prov, full_line = decode_tokens_to_plate(encoded)
        print(f"Original: ({p_type}, {d3}, {d4}, {prov})")
        print(f"Encoded ({len(encoded)} tokens, max 11): {encoded}")
        print(f"Decoded: '{full_line}' -> ({decoded_type}, {dec_d3}, {dec_d4}, {dec_prov})")
        assert len(encoded) <= 11, f"Exceeded 11 tokens: {len(encoded)}"
        assert decoded_type == p_type, f"Mismatch type: {decoded_type} vs {p_type}"
        assert dec_d3 == d3, f"Mismatch d3: {dec_d3} vs {d3}"
        assert dec_d4 == d4, f"Mismatch d4: {dec_d4} vs {d4}"
        assert dec_prov == prov, f"Mismatch prov: {dec_prov} vs {prov}"
    print("All token codec self-tests passed successfully!")
