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

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C)
        N = X.shape[0]
        kH, kW, C_in, C_out = self.kernel.shape
        sH, sW = self.strides

        # Pad all N inputs at once: pad axes 1,2 (H,W), not batch or channel
        if self.padding == "same":
            pH = kH // 2
            pW = kW // 2
            X_pad = np.pad(X, ((0, 0), (pH, pH), (pW, pW), (0, 0)), mode="constant")
        else:
            X_pad = X

        _, H_pad, W_pad, _ = X_pad.shape
        H_out = (H_pad - kH) // sH + 1
        W_out = (W_pad - kW) // sW + 1
        out = np.zeros((N, H_out, W_out, C_out), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                patch_batch = X_pad[:, i*sH:i*sH+kH, j*sW:j*sW+kW, :]  # (N, kH, kW, C_in)
                out[:, i, j, :] = np.einsum('nhwc,hwck->nk', patch_batch, self.kernel) + self.bias

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
