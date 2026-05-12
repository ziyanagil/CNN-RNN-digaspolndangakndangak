import numpy as np

class Embedding:
    def __init__(self):
        self.W: np.ndarray = None

    def load_weights(self, keras_layer) -> None:
        self.W = keras_layer.get_weights()[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.W[x]
