import tensorflow as tf
from tensorflow import keras

def build_init_inject_rnn(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = 256,
    rnn_units: int = 256,
) -> keras.Model:
    # TODO: implementasi Functional API dengan Concatenate/Add layer
    raise NotImplementedError

def build_init_inject_lstm(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = 256,
    lstm_units: int = 256,
) -> keras.Model:
    # TODO: implementasi
    raise NotImplementedError
