import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def load_weights(self, keras_layer) -> None:
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi sliding window max
        raise NotImplementedError


class AveragePooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def load_weights(self, keras_layer) -> None:
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi sliding window average
        raise NotImplementedError


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi
        raise NotImplementedError


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        # TODO: implementasi
        raise NotImplementedError
