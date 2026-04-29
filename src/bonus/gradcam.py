import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_intermediate_feature_maps(
    keras_model,
    image: np.ndarray,
    layer_names: list[str],
) -> dict[str, np.ndarray]:
    # TODO: buat intermediate model dengan tf.keras.Model(inputs, [layer.output ...])
    raise NotImplementedError


def grad_cam(
    keras_model,
    image: np.ndarray,
    class_idx: int,
    last_conv_layer_name: str,
) -> np.ndarray:
    # TODO: implementasi Grad-CAM (Selvaraju et al., 2016)
    # 1. GradientTape untuk grad output[class_idx] terhadap last conv layer
    # 2. Global average pool gradients -> weights
    # 3. Weighted sum feature maps -> ReLU
    raise NotImplementedError


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # TODO: resize heatmap ke ukuran gambar, apply colormap, blend
    raise NotImplementedError
