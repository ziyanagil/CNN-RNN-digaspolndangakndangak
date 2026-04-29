import tensorflow as tf
from tensorflow import keras
import numpy as np


BATCH_SIZE = 32
EPOCHS = 20
IMG_SIZE = (150, 150)
NUM_CLASSES = 6
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"

DATA_DIR = "data/intel"
TRAIN_DIR = f"{DATA_DIR}/seg_train/seg_train"
VAL_DIR   = f"{DATA_DIR}/seg_test/seg_test"
TEST_DIR  = f"{DATA_DIR}/seg_pred"


def build_conv2d_model(
    num_conv_layers: int = 3,
    filters: list[int] = [32, 64, 128],
    kernel_sizes: list[int] = [3, 3, 3],
    pooling: str = "max",
) -> keras.Model:
    # TODO: implementasi Sequential model
    raise NotImplementedError


def build_locally_connected_model() -> keras.Model:
    # TODO: implementasi
    raise NotImplementedError


def get_data_loaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    # TODO: implementasi dengan keras.utils.image_dataset_from_directory
    raise NotImplementedError


def train(model: keras.Model, train_ds, val_ds, epochs=EPOCHS) -> keras.callbacks.History:
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["accuracy"])
    # TODO: tambahkan ModelCheckpoint callback
    raise NotImplementedError
