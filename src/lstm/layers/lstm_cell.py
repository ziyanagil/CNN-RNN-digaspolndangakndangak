import numpy as np
from .activations import sigmoid, tanh

class LSTMCell:
    def __init__(self):
        self.W_x: np.ndarray = None
        self.W_h: np.ndarray = None
        self.b: np.ndarray   = None
        self.units: int      = None

    def load_weights(self, keras_layer) -> None:
        weights = keras_layer.get_weights()
        self.W_x = weights[0]
        self.W_h = weights[1]
        bias = weights[2]
        self.b = bias[0] + bias[1] if bias.ndim == 2 else bias
        self.units = self.W_h.shape[0]

    def forward(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u = self.units
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b

        i_gate = sigmoid(gates[0 * u: 1 * u])
        f_gate = sigmoid(gates[1 * u: 2 * u]) 
        g_gate = tanh   (gates[2 * u: 3 * u])
        o_gate = sigmoid(gates[3 * u: 4 * u])

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * tanh(c_t)
        return h_t, c_t

    def forward_with_cache(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        u = self.units
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b

        i_gate = sigmoid(gates[0 * u: 1 * u])
        f_gate = sigmoid(gates[1 * u: 2 * u])
        g_gate = tanh   (gates[2 * u: 3 * u])
        o_gate = sigmoid(gates[3 * u: 4 * u])

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * tanh(c_t)

        gates_dict = {"i": i_gate, "f": f_gate, "g": g_gate, "o": o_gate}
        return h_t, c_t, gates_dict

    def forward_batch(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        u = self.units
        gates = x_t @ self.W_x + h_prev @ self.W_h + self.b

        i_gate = sigmoid(gates[:, 0 * u: 1 * u])
        f_gate = sigmoid(gates[:, 1 * u: 2 * u])
        g_gate = tanh   (gates[:, 2 * u: 3 * u])
        o_gate = sigmoid(gates[:, 3 * u: 4 * u])

        c_t = f_gate * c_prev + i_gate * g_gate
        h_t = o_gate * tanh(c_t)
        return h_t, c_t

class LSTM:
    def __init__(self):
        self.cells: list[LSTMCell] = []

    def add_layer(self, cell: LSTMCell) -> None:
        self.cells.append(cell)

    def forward(
        self, x: np.ndarray, return_sequences: bool = False
    ) -> np.ndarray:
        seq_len = x.shape[0]
        cur_input = x

        for cell in self.cells:
            h = np.zeros(cell.units, dtype=np.float32)
            c = np.zeros(cell.units, dtype=np.float32)
            layer_hs = []
            for t in range(seq_len):
                h, c = cell.forward(cur_input[t], h, c)
                layer_hs.append(h)
            cur_input = np.stack(layer_hs)

        return cur_input if return_sequences else cur_input[-1]

    def forward_with_cache(
        self, x: np.ndarray, return_sequences: bool = False
    ) -> tuple[np.ndarray, list[dict]]:
        seq_len = x.shape[0]
        cur_input = x
        caches = []

        for cell in self.cells:
            h = np.zeros(cell.units, dtype=np.float32)
            c = np.zeros(cell.units, dtype=np.float32)
            h_init = h.copy()
            c_init = c.copy()

            x_list = []
            h_list, c_list = [], []
            i_list, f_list, g_list, o_list = [], [], [], []

            for t in range(seq_len):
                x_t = cur_input[t]
                h, c, gates_dict = cell.forward_with_cache(x_t, h, c)
                x_list.append(x_t)
                h_list.append(h)
                c_list.append(c)
                i_list.append(gates_dict["i"])
                f_list.append(gates_dict["f"])
                g_list.append(gates_dict["g"])
                o_list.append(gates_dict["o"])

            layer_cache = {
                "x_seq":  np.stack(x_list),
                "h_seq":  np.stack(h_list),
                "c_seq":  np.stack(c_list),
                "i_seq":  np.stack(i_list),
                "f_seq":  np.stack(f_list),
                "g_seq":  np.stack(g_list),
                "o_seq":  np.stack(o_list),
                "h_init": h_init,
                "c_init": c_init,
                "W_x":    cell.W_x,
                "W_h":    cell.W_h,
            }
            caches.append(layer_cache)
            cur_input = layer_cache["h_seq"]

        output = cur_input if return_sequences else cur_input[-1]
        return output, caches

    def forward_batch(
        self, X: np.ndarray, return_sequences: bool = False
    ) -> np.ndarray:
        N, seq_len, _ = X.shape
        cur_input = X

        for cell in self.cells:
            h = np.zeros((N, cell.units), dtype=np.float32)
            c = np.zeros((N, cell.units), dtype=np.float32)
            layer_hs = []
            for t in range(seq_len):
                h, c = cell.forward_batch(cur_input[:, t, :], h, c)
                layer_hs.append(h)
            cur_input = np.stack(layer_hs, axis=1)

        return cur_input if return_sequences else cur_input[:, -1, :]
