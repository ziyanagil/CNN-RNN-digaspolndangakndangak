import os
import tensorflow as tf
from tensorflow import keras


EMBED_DIM = 256
EPOCHS = 20
BATCH_SIZE = 64

FEATURES_PATH = "features/flickr8k_features.npy"
VOCAB_PATH = "features/vocab.json"
MODELS_DIR = "models/rnn"


def masked_loss(y_true, y_pred):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=loss.dtype)
    return tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask) + 1e-8)


def build_rnn_decoder(
    vocab_size: int,
    feature_dim: int,
    embed_dim: int = EMBED_DIM,
    rnn_units: int = 256,
    num_rnn_layers: int = 1,
) -> keras.Model:
    feat_input = keras.Input(shape=(feature_dim,), name="feature")
    x_minus1 = keras.layers.Dense(embed_dim, name="feat_proj")(feat_input)
    x_minus1 = keras.layers.Reshape((1, embed_dim))(x_minus1)  # (batch, 1, embed_dim)

    cap_input = keras.Input(shape=(None,), name="caption", dtype=tf.int32)
    cap_emb = keras.layers.Embedding(vocab_size, embed_dim)(cap_input)  # (batch, seq, embed_dim)

    full_seq = keras.layers.Concatenate(axis=1)([x_minus1, cap_emb])  # (batch, seq+1, embed_dim)

    x = full_seq
    for _ in range(num_rnn_layers):
        x = keras.layers.SimpleRNN(rnn_units, return_sequences=True)(x)

    output = keras.layers.Dense(vocab_size, activation="softmax")(x)
    return keras.Model([feat_input, cap_input], output)


def train(
    model: keras.Model,
    dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    model_name: str,
    epochs: int = EPOCHS,
    save_dir: str = MODELS_DIR,
) -> keras.callbacks.History:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{model_name}.keras")

    model.compile(
        optimizer="adam",
        loss=masked_loss,
    )

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

    history = model.fit(
        dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history
