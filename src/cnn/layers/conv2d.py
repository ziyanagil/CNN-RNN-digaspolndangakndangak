import numpy as np
from .activations import relu, softmax


class Conv2D:
    def __init__(self, activation: str = "relu"):
        self.kernel: np.ndarray = None  # (kH, kW, C_in, C_out)
        self.bias: np.ndarray = None    # (C_out,)
        self.activation = activation
        self.strides = (1, 1)
        self.padding = "valid"

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.kernel = weights[0]  # (kH, kW, C_in, C_out)
        self.bias = weights[1]    # (C_out,)
        cfg = keras_layer.get_config()
        self.strides = tuple(cfg["strides"])
        self.padding = cfg["padding"]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_pad = self._pad_input(x)
        H, W, _ = x_pad.shape
        kH, kW, C_in, C_out = self.kernel.shape
        sH, sW = self.strides
        
        H_out = (H - kH) // sH + 1
        W_out = (W - kW) // sW + 1
        out = np.zeros((H_out, W_out, C_out), dtype=np.float32)
        
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[i*sH:i*sH+kH, j*sW:j*sW+kW, :]
                out[i, j, :] = np.einsum('hwc,hwck->k', patch, self.kernel) + self.bias
        return self._apply_activation(out)

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "valid":
            return x
        kH, kW = self.kernel.shape[:2]
        pH = kH // 2
        pW = kW // 2
        return np.pad(x, ((pH, pH), (pW, pW), (0, 0)), mode="constant")

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return relu(z)
        if self.activation == "softmax":
            return softmax(z)
        return z
