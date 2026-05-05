import numpy as np
from src.shared.dense import Dense
from .layers.embedding import Embedding
from .layers.simple_rnn_cell import SimpleRNNCell, SimpleRNN
from .layers.activations import softmax


class RNNDecoder:
    def __init__(self):
        self.feature_projection: Dense = None   # project CNN feature ke embed_dim
        self.embedding: Embedding = None
        self.rnn: SimpleRNN = None
        self.output_dense: Dense = None         # (units -> vocab_size, softmax)
        self.vocab: dict = None                 # word -> index
        self.idx2word: dict = None              # index -> word

    def load_weights(self, keras_model, vocab: dict) -> None:
        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}

        self.feature_projection = Dense(activation="linear")
        self.embedding = Embedding()
        self.rnn = SimpleRNN()
        self.output_dense = Dense(activation="softmax")

        rnn_layers = []
        for layer in keras_model.layers:
            cls = type(layer).__name__
            name = layer.name

            if name == "feat_proj":
                self.feature_projection.load_weights(layer)
            elif cls == "Embedding":
                self.embedding.load_weights(layer)
            elif cls == "SimpleRNN":
                cell = SimpleRNNCell()
                cell.load_weights(layer)
                self.rnn.add_layer(cell)
                rnn_layers.append(layer)
            elif cls == "Dense" and name != "feat_proj":
                self.output_dense.load_weights(layer)

    def generate_caption(
        self,
        feature_vec: np.ndarray,
        max_len: int = 30,
    ) -> str:
        start_idx = self.vocab["<start>"]
        end_idx = self.vocab["<end>"]

        # pre-inject: project feature sebagai x_{-1}, feed sebelum <start>
        x_minus1 = self.feature_projection.forward(feature_vec)  # (embed_dim,)

        # inisialisasi hidden states per layer dengan zeros
        h_states = [np.zeros(cell.units, dtype=np.float32) for cell in self.rnn.cells]

        # step t=-1: feed x_{-1} ke semua layer RNN
        x_t = x_minus1
        for i, cell in enumerate(self.rnn.cells):
            h_states[i] = cell.forward(x_t, h_states[i])
            x_t = h_states[i]

        # greedy decoding mulai dari <start>
        token = start_idx
        words = []
        for _ in range(max_len):
            x_t = self.embedding.forward(np.array(token))
            for i, cell in enumerate(self.rnn.cells):
                h_states[i] = cell.forward(x_t, h_states[i])
                x_t = h_states[i]
            logits = self.output_dense.forward(x_t)
            token = int(np.argmax(logits))
            if token == end_idx:
                break
            words.append(self.idx2word.get(token, "<unk>"))

        return " ".join(words)
