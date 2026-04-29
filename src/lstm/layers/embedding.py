import numpy as np

class Embedding:

    def __init__(self):
        self.W: np.ndarray = None  # (vocab_size, embed_dim)

    def load_weights(self, keras_layer) -> None:
        self.W = keras_layer.get_weights()[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi
        raise NotImplementedError
