import numpy as np


class CNNScratch:
    def __init__(self):
        self.layers: list = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def load_from_keras(self, keras_model) -> None:
        # TODO: iterasi self.layers dan panggil layer.load_weights(keras_layer)
        raise NotImplementedError

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            return self.forward(x)
        return np.stack([self.forward(xi) for xi in x])
