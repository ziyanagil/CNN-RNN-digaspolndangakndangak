import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

EMBED_DIM  = 256
EPOCHS     = 20
BATCH_SIZE = 64
LOSS       = "sparse_categorical_crossentropy"
OPTIMIZER  = "adam"

FEATURES_PATH = "features/flickr8k_features.npy"
VOCAB_PATH    = "features/vocab.json"

def build_lstm_decoder(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = EMBED_DIM,
    lstm_units: int = 256,
    num_lstm_layers: int = 1,
    max_len: int = 30,
) -> keras.Model:
    feat_input = keras.Input(shape=(feature_dim,), name="feature")
    x_minus1 = keras.layers.Dense(embed_dim, name="feat_proj")(feat_input)
    x_minus1 = tf.expand_dims(x_minus1, axis=1)

    cap_input = keras.Input(shape=(max_len,), name="caption", dtype=tf.int32)
    cap_emb = keras.layers.Embedding(
        vocab_size, embed_dim, name="token_embedding"
    )(cap_input)

    full_seq = keras.layers.Concatenate(axis=1, name="full_seq")(
        [x_minus1, cap_emb]
    )

    x = full_seq
    for i in range(num_lstm_layers):
        x = keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            name=f"lstm_{i}",
        )(x)

    output = keras.layers.Dense(
        vocab_size, activation="softmax", name="output_dense"
    )(x)

    model = keras.Model(inputs=[feat_input, cap_input], outputs=output)
    return model

def build_dataset(
    features: np.ndarray,
    image_ids: list[str],
    captions_per_image: dict[str, list[np.ndarray]],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> tf.data.Dataset:
    feat_list, cap_in_list, cap_tgt_list = [], [], []

    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    for img_id, captions in captions_per_image.items():
        if img_id not in id_to_idx:
            continue
        feat_vec = features[id_to_idx[img_id]]
        for cap in captions:
            feat_list.append(feat_vec)
            cap_in_list.append(cap[:-1])
            cap_tgt_list.append(cap[1:])

    feat_arr   = np.array(feat_list,   dtype=np.float32)
    cap_in_arr = np.array(cap_in_list, dtype=np.int32)
    cap_tgt_arr = np.array(cap_tgt_list, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices(
        ((feat_arr, cap_in_arr), cap_tgt_arr)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(feat_arr), seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train(
    model: keras.Model,
    dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    model_name: str = "lstm_model",
    epochs: int = EPOCHS,
    save_path: str = "models/lstm/",
) -> keras.callbacks.History:
    os.makedirs(save_path, exist_ok=True)
    ckpt_path = os.path.join(save_path, f"{model_name}.weights.h5")

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history