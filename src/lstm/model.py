from typing import List
import numpy as np
from src.shared.dense import Dense
from src.shared.caption_utils import SPECIAL_TOKENS
from .layers.embedding import Embedding
from .layers.lstm_cell import LSTM, LSTMCell
class LSTMDecoder:
    def __init__(self):
        self.feature_projection: Dense = None
        self.embedding: Embedding = None
        self.lstm: LSTM = None
        self.output_dense: Dense = None
        self.vocab: dict = None
        self.idx2word: dict = None

    def load_weights(self, keras_model, vocab: dict) -> None:
        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}

        dense_layers = []
        embedding_layers = []
        lstm_layers = []

        for layer in keras_model.layers:
            cls = type(layer).__name__
            if cls == "Embedding":
                embedding_layers.append(layer)
            elif cls == "LSTM":
                lstm_layers.append(layer)
            elif cls == "Dense":
                dense_layers.append(layer)

        self.feature_projection = Dense(activation="linear")
        self.feature_projection.load_weights(dense_layers[0])

        self.embedding = Embedding()
        self.embedding.load_weights(embedding_layers[0])

        self.lstm = LSTM()
        for keras_lstm in lstm_layers:
            cell = LSTMCell()
            cell.load_weights(keras_lstm)
            self.lstm.add_layer(cell)

        self.output_dense = Dense(activation="softmax")
        self.output_dense.load_weights(dense_layers[1])

    def generate_caption(
        self,
        feature_vec: np.ndarray,
        max_len: int = 30,
    ) -> str:
        x_minus1 = self.feature_projection.forward(feature_vec)

        num_layers = len(self.lstm.cells)
        h_states = [np.zeros(cell.units, dtype=np.float32)
                    for cell in self.lstm.cells]
        c_states = [np.zeros(cell.units, dtype=np.float32)
                    for cell in self.lstm.cells]

        cur_input = x_minus1
        for i, cell in enumerate(self.lstm.cells):
            h_states[i], c_states[i] = cell.forward(
                cur_input, h_states[i], c_states[i]
            )
            cur_input = h_states[i]

        start_idx = self.vocab.get("<start>", SPECIAL_TOKENS["<start>"])
        end_idx   = self.vocab.get("<end>",   SPECIAL_TOKENS["<end>"])
        pad_idx   = self.vocab.get("<pad>",   SPECIAL_TOKENS["<pad>"])

        token_idx = start_idx
        caption_tokens = []

        for _ in range(max_len):
            token_emb = self.embedding.forward(
                np.array(token_idx, dtype=np.int32)
            )

            cur_input = token_emb
            for i, cell in enumerate(self.lstm.cells):
                h_states[i], c_states[i] = cell.forward(
                    cur_input, h_states[i], c_states[i]
                )
                cur_input = h_states[i]

            logits = self.output_dense.forward(cur_input)
            token_idx = int(np.argmax(logits))

            if token_idx == end_idx or token_idx == pad_idx:
                break

            word = self.idx2word.get(token_idx, "<unk>")
            if word not in ("<start>", "<end>", "<pad>", "<unk>"):
                caption_tokens.append(word)

        return " ".join(caption_tokens)

    def generate_captions_batch(
        self,
        features: np.ndarray,
        max_len: int = 30,
        batch_size: int = 16,
    ) -> List[str]:
        N = features.shape[0]
        all_captions = []

        for start in range(0, N, batch_size):
            batch_feats = features[start: start + batch_size]
            B = batch_feats.shape[0]

            x_minus1 = self.feature_projection.forward(batch_feats)

            h_states = [np.zeros((B, cell.units), dtype=np.float32)
                        for cell in self.lstm.cells]
            c_states = [np.zeros((B, cell.units), dtype=np.float32)
                        for cell in self.lstm.cells]

            cur_input = x_minus1
            for i, cell in enumerate(self.lstm.cells):
                h_states[i], c_states[i] = cell.forward_batch(
                    cur_input, h_states[i], c_states[i]
                )
                cur_input = h_states[i]

            start_idx = self.vocab.get("<start>", SPECIAL_TOKENS["<start>"])
            end_idx   = self.vocab.get("<end>",   SPECIAL_TOKENS["<end>"])
            pad_idx   = self.vocab.get("<pad>",   SPECIAL_TOKENS["<pad>"])

            token_ids = np.full(B, start_idx, dtype=np.int32)
            caption_lists = [[] for _ in range(B)]
            finished = np.zeros(B, dtype=bool)

            for _ in range(max_len):
                token_embs = self.embedding.forward(token_ids)

                cur_input = token_embs
                for i, cell in enumerate(self.lstm.cells):
                    h_states[i], c_states[i] = cell.forward_batch(
                        cur_input, h_states[i], c_states[i]
                    )
                    cur_input = h_states[i]

                logits = self.output_dense.forward(cur_input)
                token_ids = np.argmax(logits, axis=-1).astype(np.int32)

                for b in range(B):
                    if finished[b]:
                        continue
                    tid = int(token_ids[b])
                    if tid == end_idx or tid == pad_idx:
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
