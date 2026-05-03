import numpy as np


class Dense:
    def __init__(self, activation: str = "linear"):
        self.W: np.ndarray = None
        self.b: np.ndarray = None
        self.activation = activation

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.W = weights[0]  # shape (in_dim, out_dim)
        self.b = weights[1]  # shape (out_dim,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._apply_activation(x @ self.W + self.b)

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "softmax":
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        return z  # linear
