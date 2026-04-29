import numpy as np
from .activations import relu, softmax


class LocallyConnected2D:
    def __init__(self, activation: str = "relu"):
        self.kernel: np.ndarray = None  # (out_rows*out_cols, kH*kW*C_in, C_out)
        self.bias: np.ndarray = None    # (out_rows*out_cols, C_out)
        self.activation = activation
        self.kernel_size = (3, 3)
        self.strides = (1, 1)
        self.output_shape_spatial = None  # (out_rows, out_cols)

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.kernel = weights[0]
        self.bias = weights[1]
        cfg = keras_layer.get_config()
        self.kernel_size = tuple(cfg["kernel_size"])
        self.strides = tuple(cfg["strides"])
        # inferensi output spatial shape dari bobot
        n_positions = self.kernel.shape[0]
        # TODO: set self.output_shape_spatial setelah load (perlu info input shape)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi
        # Untuk setiap posisi (i,j) -> index pos = i*W_out + j:
        #   patch = x[i*sH:i*sH+kH, j*sW:j*sW+kW, :].flatten()
        #   output[i,j,:] = kernel[pos].T @ patch + bias[pos]
        raise NotImplementedError

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu(z)
        if self.activation == "softmax":
            return softmax(z)
        return z
