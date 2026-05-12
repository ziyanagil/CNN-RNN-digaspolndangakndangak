import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def get_intermediate_feature_maps(
    keras_model,
    image: np.ndarray,
    layer_names: list[str],
) -> dict[str, np.ndarray]:
    outputs = [keras_model.get_layer(n).output for n in layer_names]
    intermediate_model = tf.keras.Model(inputs=keras_model.input, outputs=outputs)

    img_batch = np.expand_dims(image, axis=0).astype(np.float32)
    results = intermediate_model.predict(img_batch, verbose=0)

    if len(layer_names) == 1:
        results = [results]

    return {name: results[i][0] for i, name in enumerate(layer_names)}


def grad_cam(
    keras_model,
    image: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str,
) -> np.ndarray:
    grad_model = tf.keras.Model(
        inputs=keras_model.input,
        outputs=[keras_model.get_layer(last_conv_layer_name).output, keras_model.output],
    )

    img_batch = tf.cast(np.expand_dims(image, axis=0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_batch)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    heatmap = np.dot(conv_outputs, pooled_grads)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]

    heatmap_pil = Image.fromarray(np.uint8(heatmap * 255))
    heatmap_resized = np.array(heatmap_pil.resize((w, h), Image.BILINEAR)) / 255.0

    colormap = cm.get_cmap("jet")
    heatmap_rgb = colormap(heatmap_resized)[:, :, :3]

    if image.max() <= 1.0:
        img_float = image.astype(np.float32)
    else:
        img_float = image.astype(np.float32) / 255.0

    superimposed = (1 - alpha) * img_float + alpha * heatmap_rgb
    superimposed = np.clip(superimposed, 0, 1)

    return superimposed
