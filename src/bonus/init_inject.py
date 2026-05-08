import numpy as np
import tensorflow as tf
from tensorflow import keras


def _masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=loss.dtype)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)


def build_init_inject_rnn(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = 256,
    rnn_units: int = 256,
    num_layers: int = 1,
) -> keras.Model:
    feat_input = keras.Input(shape=(feature_dim,), name="feature")
    cap_input = keras.Input(shape=(None,), name="caption", dtype=tf.int32)

    feat_proj = keras.layers.Dense(rnn_units, name="feat_proj")(feat_input)
    feat_proj = keras.layers.Reshape((1, rnn_units), name="feat_expand")(feat_proj)

    cap_emb = keras.layers.Embedding(vocab_size, embed_dim, name="embedding")(cap_input)

    x = cap_emb
    for i in range(num_layers):
        x = keras.layers.SimpleRNN(rnn_units, return_sequences=True, name=f"simple_rnn_{i}")(x)

    merged = keras.layers.Add(name="merge")([x, feat_proj])
    output = keras.layers.Dense(vocab_size, activation="softmax", name="output")(merged)
    return keras.Model([feat_input, cap_input], output)


def build_init_inject_lstm(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = 256,
    lstm_units: int = 256,
    num_layers: int = 1,
) -> keras.Model:
    feat_input = keras.Input(shape=(feature_dim,), name="feature")
    cap_input = keras.Input(shape=(None,), name="caption", dtype=tf.int32)

    feat_proj = keras.layers.Dense(lstm_units, name="feat_proj")(feat_input)
    feat_proj = keras.layers.Reshape((1, lstm_units), name="feat_expand")(feat_proj)

    cap_emb = keras.layers.Embedding(vocab_size, embed_dim, name="embedding")(cap_input)

    x = cap_emb
    for i in range(num_layers):
        x = keras.layers.LSTM(lstm_units, return_sequences=True, name=f"lstm_{i}")(x)

    merged = keras.layers.Add(name="merge")([x, feat_proj])
    output = keras.layers.Dense(vocab_size, activation="softmax", name="output")(merged)
    return keras.Model([feat_input, cap_input], output)


def train_init_inject(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str,
    save_dir: str,
    epochs: int = 20,
) -> keras.callbacks.History:
    import os
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}.keras")

    model.compile(optimizer="adam", loss=_masked_loss)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
    ]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)


def keras_greedy_decode(model, feature_vec: np.ndarray, vocab: dict,
                        idx2word: dict, max_len: int = 30) -> str:
    start_idx = vocab["<start>"]
    end_idx = vocab["<end>"]

    tokens = [start_idx]
    words = []

    for _ in range(max_len):
        feat = feature_vec[None, :]
        cap = np.array([tokens], dtype=np.int32)
        out = model([feat, cap], training=False).numpy()
        next_tok = int(np.argmax(out[0, -1]))
        if next_tok == end_idx:
            break
        tokens.append(next_tok)
        word = idx2word.get(next_tok, "<unk>")
        if word not in ("<pad>", "<start>", "<unk>"):
            words.append(word)

    return " ".join(words)


class InitInjectRNNDecoder:
    def __init__(self):
        self.feature_projection = None
        self.embedding = None
        self.rnn = None
        self.output_dense = None
        self.vocab: dict = None
        self.idx2word: dict = None

    def load_weights(self, keras_model, vocab: dict) -> None:
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from shared.dense import Dense
        from rnn.layers.embedding import Embedding
        from rnn.layers.simple_rnn_cell import SimpleRNNCell, SimpleRNN

        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}
        self.feature_projection = Dense(activation="linear")
        self.embedding = Embedding()
        self.rnn = SimpleRNN()
        self.output_dense = Dense(activation="softmax")

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
            elif cls == "Dense" and name == "output":
                self.output_dense.load_weights(layer)

    def generate_caption(self, feature_vec: np.ndarray, max_len: int = 30) -> str:
        start_idx = self.vocab["<start>"]
        end_idx = self.vocab["<end>"]

        feat_proj = self.feature_projection.forward(feature_vec)
        h_states = [np.zeros(cell.units, dtype=np.float32) for cell in self.rnn.cells]

        token = start_idx
        words = []
        for _ in range(max_len):
            x_t = self.embedding.forward(np.array(token))
            for i, cell in enumerate(self.rnn.cells):
                h_states[i] = cell.forward(x_t, h_states[i])
                x_t = h_states[i]
            combined = x_t + feat_proj
            logits = self.output_dense.forward(combined)
            token = int(np.argmax(logits))
            if token == end_idx:
                break
            word = self.idx2word.get(token, "<unk>")
            if word not in ("<pad>", "<start>", "<unk>"):
                words.append(word)

        return " ".join(words)


class InitInjectLSTMDecoder:
    def __init__(self):
        self.feature_projection = None
        self.embedding = None
        self.lstm = None
        self.output_dense = None
        self.vocab: dict = None
        self.idx2word: dict = None

    def load_weights(self, keras_model, vocab: dict) -> None:
        import sys, os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from shared.dense import Dense
        from lstm.layers.embedding import Embedding
        from lstm.layers.lstm_cell import LSTMCell, LSTM

        self.vocab = vocab
        self.idx2word = {v: k for k, v in vocab.items()}
        self.feature_projection = Dense(activation="linear")
        self.embedding = Embedding()
        self.lstm = LSTM()
        self.output_dense = Dense(activation="softmax")

        for layer in keras_model.layers:
            cls = type(layer).__name__
            name = layer.name
            if name == "feat_proj":
                self.feature_projection.load_weights(layer)
            elif cls == "Embedding":
                self.embedding.load_weights(layer)
            elif cls == "LSTM":
                cell = LSTMCell()
                cell.load_weights(layer)
                self.lstm.add_layer(cell)
            elif cls == "Dense" and name == "output":
                self.output_dense.load_weights(layer)

    def generate_caption(self, feature_vec: np.ndarray, max_len: int = 30) -> str:
        start_idx = self.vocab["<start>"]
        end_idx = self.vocab["<end>"]

        feat_proj = self.feature_projection.forward(feature_vec)
        h_states = [np.zeros(cell.units, dtype=np.float32) for cell in self.lstm.cells]
        c_states = [np.zeros(cell.units, dtype=np.float32) for cell in self.lstm.cells]

        token = start_idx
        words = []
        for _ in range(max_len):
            x_t = self.embedding.forward(np.array(token))
            for i, cell in enumerate(self.lstm.cells):
                h_states[i], c_states[i] = cell.forward(x_t, h_states[i], c_states[i])
                x_t = h_states[i]
            combined = x_t + feat_proj
            logits = self.output_dense.forward(combined)
            token = int(np.argmax(logits))
            if token == end_idx:
                break
            word = self.idx2word.get(token, "<unk>")
            if word not in ("<pad>", "<start>", "<unk>"):
                words.append(word)

        return " ".join(words)
