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
        self.W_x = weights[0]
        self.W_h = weights[1]
        self.b = weights[2]
        self.units = self.W_h.shape[0]

    def forward(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        return tanh(x_t @ self.W_x + h_prev @ self.W_h + self.b)


class SimpleRNN:
    def __init__(self):
        self.cells: list[SimpleRNNCell] = []

    def add_layer(self, cell: SimpleRNNCell) -> None:
        self.cells.append(cell)

    def forward(self, x: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        # x: (seq_len, input_dim)
        seq_len = x.shape[0]
        cur_input = x
        for cell in self.cells:
            h = np.zeros(cell.units, dtype=np.float32)
            layer_hs = []
            for t in range(seq_len):
                h = cell.forward(cur_input[t], h)
                layer_hs.append(h)
            cur_input = np.stack(layer_hs)  # (seq_len, units) — input for next layer
        return cur_input if return_sequences else h
