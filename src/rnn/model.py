import numpy as np
from src.shared.dense import Dense
from .layers.embedding import Embedding
from .layers.simple_rnn_cell import SimpleRNN


class RNNDecoder:
    def __init__(self):
        self.feature_projection: Dense = None   # project CNN feature ke embed_dim
        self.embedding: Embedding = None
        self.rnn: SimpleRNN = None
        self.output_dense: Dense = None         # (units -> vocab_size, softmax)
        self.vocab: dict = None                 # word -> index
        self.idx2word: dict = None              # index -> word

    def load_weights(self, keras_model, vocab: dict) -> None:
        # TODO: implementasi — load masing-masing layer
        raise NotImplementedError

    def generate_caption(
        self,
        feature_vec: np.ndarray,
        max_len: int = 30,
    ) -> str:
        # TODO: implementasi pre-inject greedy decoding
        # 1. project feature_vec -> x_{-1}
        # 2. inisialisasi h = zeros
        # 3. loop: feed x_t ke RNN, ambil argmax dari output, lanjut sampai <end>
        raise NotImplementedError
