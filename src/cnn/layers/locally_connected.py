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

    def forward(self, x: np.ndarray) -> np.ndarray:
        kH, kW = self.kernel_size
        sH, sW = self.strides
        H, W, C_in = x.shape
        
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1
        
        out = np.zeros((H_out, W_out, self.kernel.shape[-1]), dtype=np.float32)
        for i in range(H_out):
            for j in range(W_out):
                pos = i * W_out + j
                patch = x[i*sH:i*sH+kH, j*sW:j*sW+kW, :].flatten()
                out[i, j, :] = patch @ self.kernel[pos] + self.bias[pos]
        return self._apply_activation(out)

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C) — LC2D is inherently per-position, loop over batch
        return np.stack([self.forward(x) for x in X])

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu(z)
        if self.activation == "softmax":
            return softmax(z)
        return z
