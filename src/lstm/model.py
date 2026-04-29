import numpy as np
from src.shared.dense import Dense
from .layers.embedding import Embedding
from .layers.lstm_cell import LSTM

class LSTMDecoder:

    def __init__(self):
        self.feature_projection: Dense = None
        self.embedding: Embedding = None
        self.lstm: LSTM = None
        self.output_dense: Dense = None
        self.vocab: dict = None
        self.idx2word: dict = None

    def load_weights(self, keras_model, vocab: dict) -> None:
        # TODO: implementasi
        raise NotImplementedError

    def generate_caption(
        self,
        feature_vec: np.ndarray,
        max_len: int = 30,
    ) -> str:
        # TODO: implementasi pre-inject greedy decoding dengan LSTM
        # Bedanya dengan RNN: track h dan c per layer
        raise NotImplementedError
