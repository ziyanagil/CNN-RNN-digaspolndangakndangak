import os
import tensorflow as tf
from tensorflow import keras
import numpy as np


BATCH_SIZE = 32
EPOCHS =int(os.environ.get('CNN_EPOCHS', 10)) 
IMG_SIZE = (150, 150)
NUM_CLASSES = 6
OPTIMIZER = "adam"
LOSS = "sparse_categorical_crossentropy"

DATA_DIR = os.environ.get('CNN_DATA_DIR', 'data/intel')

def find_subdir(root, target):
    for r, dirs, files in os.walk(root):
        if target in dirs:
            cand = os.path.join(r, target)
            if os.path.exists(os.path.join(cand, target)): return os.path.join(cand, target)
            return cand
    return os.path.join(root, target)

TRAIN_DIR = find_subdir(DATA_DIR, 'seg_train')
VAL_DIR   = find_subdir(DATA_DIR, 'seg_test')
TEST_DIR  = find_subdir(DATA_DIR, 'seg_pred')


def build_conv2d_model(
    num_conv_layers: int = 3,
    filters: list[int] = [32, 64, 128],
    kernel_sizes: list[int] = [3, 3, 3],
    pooling: str = "max",
) -> keras.Model:
    Pool = keras.layers.MaxPooling2D if pooling == 'max' else keras.layers.AveragePooling2D
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = inputs
    for i in range(num_conv_layers):
        x = keras.layers.Conv2D(filters[i], kernel_sizes[i],
                                activation='relu', padding='same')(x)
        x = Pool(pool_size=(2,2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def build_locally_connected_model() -> keras.Model:
    inputs = keras.Input(shape=(64, 64, 3))
    x = keras.layers.LocallyConnected2D(32, 3, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.LocallyConnected2D(64, 3, activation='relu')(x)
    x = keras.layers.GlobalMaxPooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(6, activation='softmax')(x)
    return keras.Model(inputs, outputs)


def get_data_loaders(img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=img_size, batch_size=batch_size, label_mode='int')
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=img_size, batch_size=batch_size, label_mode='int')
    
    norm = keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train(model: keras.Model, train_ds, val_ds, model_name="model", epochs=EPOCHS) -> keras.callbacks.History:
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=["accuracy"])
    
    os.makedirs("models/cnn", exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        f"models/cnn/{model_name}.keras", 
        save_best_only=True, 
        monitor="val_loss"
    )
    
    return model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=epochs, 
        callbacks=[checkpoint]
    )
