import tensorflow as tf
from tensorflow import keras
import numpy as np


EMBED_DIM = 256
EPOCHS = 20
BATCH_SIZE = 64
LOSS = "sparse_categorical_crossentropy"
OPTIMIZER = "adam"

FEATURES_PATH = "features/flickr8k_features.npy"
VOCAB_PATH = "features/vocab.json"


def build_lstm_decoder(
    vocab_size: int,
    embed_dim: int = EMBED_DIM,
    lstm_units: int = 256,
    num_lstm_layers: int = 1,
) -> keras.Model:
    # TODO: implementasi Functional API
    raise NotImplementedError


def train(
    model: keras.Model,
    dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    epochs: int = EPOCHS,
    save_path: str = "models/lstm/",
) -> keras.callbacks.History:
    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    # TODO: ModelCheckpoint
    raise NotImplementedError
