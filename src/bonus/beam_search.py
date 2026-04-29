import numpy as np
from dataclasses import dataclass, field

@dataclass
class BeamHypothesis:
    tokens: list[int] = field(default_factory=list)
    score: float = 0.0          # log-probability sum
    hidden_states: object = None  # h (dan c untuk LSTM)


def beam_search_rnn(
    decoder,           # RNNDecoder atau LSTMDecoder from scratch
    feature_vec: np.ndarray,
    vocab: dict,
    idx2word: dict,
    beam_size: int = 5,
    max_len: int = 30,
) -> str:
    # TODO: implementasi beam search
    # 1. Inisialisasi beam dengan <start>
    # 2. Per timestep: expand setiap hypothesis, ambil top-k
    # 3. Stop jika semua hypothesis sudah <end> atau max_len tercapai
    raise NotImplementedError
