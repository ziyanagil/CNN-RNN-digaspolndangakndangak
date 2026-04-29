import numpy as np
from .activations import tanh


class SimpleRNNCell:
    def __init__(self):
        self.W_x: np.ndarray = None   # (input_dim, units)
        self.W_h: np.ndarray = None   # (units, units)
        self.b: np.ndarray = None     # (units,)
        self.units: int = None

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.W_x = weights[0]  # kernel
        self.W_h = weights[1]  # recurrent_kernel
        self.b = weights[2]    # bias
        self.units = self.W_h.shape[0]

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        # TODO: h_t = tanh(x_t @ W_x + h_prev @ W_h + b)
        raise NotImplementedError


class SimpleRNN:
    def __init__(self):
        self.cells: list[SimpleRNNCell] = []

    def add_layer(self, cell: SimpleRNNCell) -> None:
        self.cells.append(cell)

    def forward(self, x: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        # TODO: loop timestep, propagate melalui stacked cells
        raise NotImplementedError
