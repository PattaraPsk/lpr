"""
Data processing module for dataset preparation and loading.
"""

from src.data.dataset import (
    BLANK_TOKEN,
    CHAR_TO_IDX,
    DLT_THAI_CONSONANTS,
    IDX_TO_CHAR,
    VOCAB_LIST,
    VOCAB_SIZE,
    LicensePlateDataset,
    ctc_collate_fn,
    decode_indices,
    encode_label,
    get_dataloader,
)
from src.data.prepare import create_splits, prepare_dataset, scan_dataset_directory

__all__ = [
    "BLANK_TOKEN",
    "CHAR_TO_IDX",
    "DLT_THAI_CONSONANTS",
    "IDX_TO_CHAR",
    "VOCAB_LIST",
    "VOCAB_SIZE",
    "LicensePlateDataset",
    "ctc_collate_fn",
    "decode_indices",
    "encode_label",
    "get_dataloader",
    "create_splits",
    "prepare_dataset",
    "scan_dataset_directory",
]
