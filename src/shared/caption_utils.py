import re
import json
import numpy as np
from typing import Optional


SPECIAL_TOKENS = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}


def clean_caption(caption: str) -> str:
    # TODO: implementasi (str.lower + re.sub)
    raise NotImplementedError


def build_vocabulary(captions: list[str], max_vocab_size: Optional[int] = None) -> dict:
    # TODO: implementasi
    raise NotImplementedError


def save_vocabulary(vocab: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocabulary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def encode_caption(caption: str, vocab: dict, max_len: int) -> np.ndarray:
    # TODO: implementasi
    raise NotImplementedError


def encode_all_captions(
    captions: list[str], vocab: dict, max_len: int
) -> np.ndarray:
    return np.stack([encode_caption(c, vocab, max_len) for c in captions])
