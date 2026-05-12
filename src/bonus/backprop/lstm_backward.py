import numpy as np

def lstm_backward(
    d_h_seq: np.ndarray,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_seq  = cache["x_seq"]
    h_seq  = cache["h_seq"]
    c_seq  = cache["c_seq"]
    i_seq  = cache["i_seq"]
    f_seq  = cache["f_seq"]
    g_seq  = cache["g_seq"]
    o_seq  = cache["o_seq"]
    h_init = cache["h_init"]
    c_init = cache["c_init"]
    W_x    = cache["W_x"]
    W_h    = cache["W_h"]

    seq_len, input_dim = x_seq.shape
    units = h_seq.shape[1]

    d_x   = np.zeros_like(x_seq)
    d_W_x = np.zeros_like(W_x)
    d_W_h = np.zeros_like(W_h)
    d_b   = np.zeros(4 * units)

    d_h_next = np.zeros(units)
    d_c_next = np.zeros(units)

    for t in reversed(range(seq_len)):
        x_t    = x_seq[t]
        h_t    = h_seq[t]
        c_t    = c_seq[t]
        i_t    = i_seq[t]
        f_t    = f_seq[t]
        g_t    = g_seq[t]
        o_t    = o_seq[t]

        h_prev = h_seq[t - 1] if t > 0 else h_init
        c_prev = c_seq[t - 1] if t > 0 else c_init

        f_next = f_seq[t + 1] if t < seq_len - 1 else np.zeros(units)

        d_h_t = d_h_seq[t] + d_h_next             
        tanh_c_t = np.tanh(c_t)
        d_c_t = (d_h_t * o_t * (1.0 - tanh_c_t ** 2)
                 + d_c_next * f_next)               

        d_o_t = d_h_t * tanh_c_t
        d_f_t = d_c_t * c_prev
        d_i_t = d_c_t * g_t
        d_g_t = d_c_t * i_t

        d_i_raw = d_i_t * i_t * (1.0 - i_t)
        d_f_raw = d_f_t * f_t * (1.0 - f_t)
        d_g_raw = d_g_t * (1.0 - g_t ** 2)
        d_o_raw = d_o_t * o_t * (1.0 - o_t)

        d_gates = np.concatenate([d_i_raw, d_f_raw, d_g_raw, d_o_raw])

        d_W_x += np.outer(x_t, d_gates)
        d_W_h += np.outer(h_prev, d_gates)
        d_b   += d_gates

        d_x[t] = d_gates @ W_x.T

        d_h_next = d_gates @ W_h.T
        d_c_next = d_c_t * f_t

    return d_x, d_W_x, d_W_h, d_b
