import numpy as np


def simple_rnn_backward(
    d_h_seq: np.ndarray,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_seq  = cache["x_seq"]
    h_seq  = cache["h_seq"]
    h_init = cache["h_init"]
    W_x    = cache["W_x"]
    W_h    = cache["W_h"]

    seq_len, input_dim = x_seq.shape
    units = h_seq.shape[1]

    d_x   = np.zeros_like(x_seq)
    d_W_x = np.zeros_like(W_x)
    d_W_h = np.zeros_like(W_h)
    d_b   = np.zeros(units)

    d_h_next = np.zeros(units)

    for t in reversed(range(seq_len)):
        h_t    = h_seq[t]
        h_prev = h_seq[t - 1] if t > 0 else h_init
        x_t    = x_seq[t]

        d_h_t = d_h_seq[t] + d_h_next

        d_z_t = d_h_t * (1.0 - h_t ** 2)

        d_W_x += np.outer(x_t, d_z_t)
        d_W_h += np.outer(h_prev, d_z_t)
        d_b   += d_z_t

        d_x[t] = d_z_t @ W_x.T

        d_h_next = d_z_t @ W_h.T

    return d_x, d_W_x, d_W_h, d_b
