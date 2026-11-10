"""
Evaluation Metrics & Distance Functions for License Plate Recognition.
Compliant with ISO/IEC 42001:2023 Verification Standards.
"""

from typing import List, Sequence, Tuple


DLT_REPLACEMENTS = {
    "ฃ": "ข",
    "ฅ": "ค",
    "ฏ": "ฎ",
}


def sanitize_thai_plate_text(text: str) -> str:
    """
    Sanitizes recognition text according to DLT specifications:
    - Normalizes obsolete Thai consonants: ฃ->ข, ฅ->ค, ฏ->ฎ
    - Removes whitespace and extraneous symbols
    """
    if not text:
        return ""
    text = text.strip().replace(" ", "").replace("-", "")
    for k, v in DLT_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        for j in range(len(v0)):
            v0[j] = v1[j]

    return v1[len(s2)]


def compute_normalized_edit_distance(preds: Sequence[str], targets: Sequence[str]) -> float:
    """
    Computes 1 - (Levenshtein Distance / max(len(pred), len(target))).
    Returns a score between 0.0 and 1.0 (higher is better, 1.0 = perfect match).
    """
    if not preds or not targets or len(preds) != len(targets):
        return 0.0

    total_score = 0.0
    for pred, target in zip(preds, targets):
        pred_clean = sanitize_thai_plate_text(pred)
        target_clean = sanitize_thai_plate_text(target)

        max_len = max(len(pred_clean), len(target_clean))
        if max_len == 0:
            total_score += 1.0
        else:
            dist = levenshtein_distance(pred_clean, target_clean)
            total_score += max(0.0, 1.0 - (dist / float(max_len)))

    return total_score / len(preds)


def compute_accuracy(preds: Sequence[str], targets: Sequence[str]) -> float:
    """
    Computes Exact Match Sequence Accuracy (0.0 to 1.0).
    """
    if not preds or not targets or len(preds) != len(targets):
        return 0.0

    correct = 0
    for pred, target in zip(preds, targets):
        if sanitize_thai_plate_text(pred) == sanitize_thai_plate_text(target):
            correct += 1

    return correct / len(preds)


def compute_character_error_rate(preds: Sequence[str], targets: Sequence[str]) -> float:
    """
    Computes Character Error Rate (CER): total edit distance / total target characters.
    """
    if not preds or not targets:
        return 0.0

    total_dist = 0
    total_chars = 0
    for pred, target in zip(preds, targets):
        pred_clean = sanitize_thai_plate_text(pred)
        target_clean = sanitize_thai_plate_text(target)

        total_dist += levenshtein_distance(pred_clean, target_clean)
        total_chars += len(target_clean)

    if total_chars == 0:
        return 0.0
    return total_dist / float(total_chars)
