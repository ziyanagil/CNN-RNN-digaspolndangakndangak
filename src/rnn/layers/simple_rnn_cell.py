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

    def forward_batch(self, x_t: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        # x_t: (N, input_dim), h_prev: (N, units) -> h_t: (N, units)
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
            cur_input = np.stack(layer_hs)
        return cur_input if return_sequences else h

    def forward_batch(self, X: np.ndarray, return_sequences: bool = False) -> np.ndarray:
        # X: (N, seq_len, input_dim)
        N, seq_len, _ = X.shape
        cur_input = X  # (N, seq_len, input_dim)
        for cell in self.cells:
            h = np.zeros((N, cell.units), dtype=np.float32)
            layer_hs = []
            for t in range(seq_len):
                h = cell.forward_batch(cur_input[:, t, :], h)
                layer_hs.append(h)
            # layer_hs: list of (N, units) -> stack to (N, seq_len, units)
            cur_input = np.stack(layer_hs, axis=1)
        return cur_input if return_sequences else h

    def forward_with_cache(self, x: np.ndarray, return_sequences: bool = False):
        # x: (seq_len, input_dim)
        # returns (output, list_of_caches)
        seq_len = x.shape[0]
        cur_input = x
        caches = []
        for cell in self.cells:
            h = np.zeros(cell.units, dtype=np.float32)
            x_seq = []
            h_seq = [h.copy()]
            for t in range(seq_len):
                x_seq.append(cur_input[t])
                h = cell.forward(cur_input[t], h)
                h_seq.append(h)
            x_seq_arr = np.stack(x_seq)        # (seq_len, input_dim)
            h_seq_arr = np.stack(h_seq)        # (seq_len+1, units)
            cache = {
                "x_seq": x_seq_arr,
                "h_seq": h_seq_arr,
                "h_init": h_seq_arr[0],
                "W_x": cell.W_x,
                "W_h": cell.W_h,
            }
            caches.append(cache)
            cur_input = h_seq_arr[1:]          # (seq_len, units) for next layer
        output = cur_input if return_sequences else cur_input[-1]
        return output, caches
