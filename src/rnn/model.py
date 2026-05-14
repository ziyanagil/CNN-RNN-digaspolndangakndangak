import numpy as np
from typing import List
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

    def generate_captions_batch(self, features, max_len=30, batch_size=16) -> List[str]:
        N = features.shape[0]
        all_captions = []

        for start in range(0, N, batch_size):
            batch_feats = features[start: start + batch_size]
            B = batch_feats.shape[0]

            # project features as x_{-1}: (B, embed_dim)
            x_minus1 = self.feature_projection.forward(batch_feats)

            # initialize hidden states per layer: (B, units)
            h_states = [np.zeros((B, cell.units), dtype=np.float32)
                        for cell in self.rnn.cells]

            # step t=-1: feed x_{-1} through all RNN layers
            cur_input = x_minus1
            for i, cell in enumerate(self.rnn.cells):
                h_states[i] = cell.forward_batch(cur_input, h_states[i])
                cur_input = h_states[i]

            start_idx = self.vocab["<start>"]
            end_idx = self.vocab["<end>"]

            token_ids = np.full(B, start_idx, dtype=np.int32)
            caption_lists = [[] for _ in range(B)]
            finished = np.zeros(B, dtype=bool)

            for _ in range(max_len):
                # embed tokens: (B, embed_dim)
                token_embs = self.embedding.forward(token_ids)

                cur_input = token_embs
                for i, cell in enumerate(self.rnn.cells):
                    h_states[i] = cell.forward_batch(cur_input, h_states[i])
                    cur_input = h_states[i]

                logits = self.output_dense.forward(cur_input)  # (B, vocab_size)
                token_ids = np.argmax(logits, axis=-1).astype(np.int32)

                for b in range(B):
                    if finished[b]:
                        continue
                    tid = int(token_ids[b])
                    if tid == end_idx:
                        finished[b] = True
                    else:
                        word = self.idx2word.get(tid, "<unk>")
                        if word not in ("<start>", "<end>", "<pad>", "<unk>"):
                            caption_lists[b].append(word)

                if finished.all():
                    break

            for b in range(B):
                all_captions.append(" ".join(caption_lists[b]))

        return all_captions
