import numpy as np


class CNNScratch:
    def __init__(self):
        self.layers: list = []

    def add(self, layer) -> None:
        self.layers.append(layer)

    def load_from_keras(self, keras_model) -> None:
        scratch_idx = 0
        for keras_layer in keras_model.layers:
            cls = type(keras_layer).__name__
            if cls in ('InputLayer', 'Dropout', 'Rescaling'):
                continue
                
            if cls in ('Conv2D', 'LocallyConnected2D', 'MaxPooling2D', 'AveragePooling2D', 'Dense'):
                if scratch_idx < len(self.layers) and hasattr(self.layers[scratch_idx], 'load_weights'):
                    self.layers[scratch_idx].load_weights(keras_layer)
                scratch_idx += 1
            elif cls in ('Flatten', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D'):
                scratch_idx += 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            return self.forward(x)
        out = x
        for layer in self.layers:
            if hasattr(layer, 'forward_batch'):
                out = layer.forward_batch(out)
            else:
                out = np.stack([layer.forward(xi) for xi in out])
        return out
