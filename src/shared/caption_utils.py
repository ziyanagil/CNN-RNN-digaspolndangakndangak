import re
import json
import numpy as np
from typing import Optional


SPECIAL_TOKENS = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}


def clean_caption(caption: str) -> str:
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", "", caption)
    return caption.strip()


def build_vocabulary(captions: list[str], max_vocab_size: Optional[int] = None) -> dict:
    vocab = dict(SPECIAL_TOKENS)
    counter = {}
    for c in captions:
        for word in c.split():
            counter[word] = counter.get(word, 0) + 1
    sorted_words = sorted(counter, key=counter.get, reverse=True)
    if max_vocab_size:
        sorted_words = sorted_words[:max_vocab_size - len(SPECIAL_TOKENS)]
    for i, word in enumerate(sorted_words, start=len(SPECIAL_TOKENS)):
        vocab[word] = i
    return vocab


def save_vocabulary(vocab: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(vocab, f, indent=2)


def load_vocabulary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def encode_caption(caption: str, vocab: dict, max_len: int) -> np.ndarray:
    tokens = ["<start>"] + caption.split() + ["<end>"]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


def encode_all_captions(
    captions: list[str], vocab: dict, max_len: int
) -> np.ndarray:
    return np.stack([encode_caption(c, vocab, max_len) for c in captions])
