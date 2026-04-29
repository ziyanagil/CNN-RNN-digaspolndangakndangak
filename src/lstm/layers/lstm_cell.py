import numpy as np
from .activations import sigmoid, tanh

class LSTMCell:

    def __init__(self):
        self.W_x: np.ndarray = None   # (input_dim, 4*units)
        self.W_h: np.ndarray = None   # (units, 4*units)
        self.b: np.ndarray = None     # (4*units,)
        self.units: int = None

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.W_x = weights[0]                # kernel
        self.W_h = weights[1]                # recurrent_kernel
        bias = weights[2]                    # shape (2, 4*units) atau (4*units,)
        self.b = bias[0] + bias[1] if bias.ndim == 2 else bias
        self.units = self.W_h.shape[0]

    def forward(
        self, x_t: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # TODO: implementasi 4 gate LSTM
        raise NotImplementedError


class LSTM:

    def __init__(self):
        self.cells: list[LSTMCell] = []

    def add_layer(self, cell: LSTMCell) -> None:
        self.cells.append(cell)

    def forward(self, x: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        # TODO: loop timestep, propagate melalui stacked cells dengan h dan c
        raise NotImplementedError
