import numpy as np

def conv2d_backward(
    d_out: np.ndarray,
    x: np.ndarray,
    kernel: np.ndarray,
    strides: tuple = (1, 1),
    padding: str = "valid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kH, kW, C_in, C_out = kernel.shape
    sH, sW = strides
    H_out, W_out, _ = d_out.shape

    if padding == "same":
        pH = max((H_out - 1) * sH + kH - x.shape[0], 0)
        pW = max((W_out - 1) * sW + kW - x.shape[1], 0)
        pad_top    = pH // 2
        pad_bottom = pH - pad_top
        pad_left   = pW // 2
        pad_right  = pW - pad_left
        x_pad = np.pad(
            x,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )
    else:
        x_pad = x
        pad_top = pad_left = 0

    d_x_pad  = np.zeros_like(x_pad, dtype=np.float64)
    d_kernel = np.zeros_like(kernel, dtype=np.float64)
    d_bias   = d_out.sum(axis=(0, 1))

    for i in range(H_out):
        for j in range(W_out):
            patch = x_pad[i * sH: i * sH + kH,
                          j * sW: j * sW + kW, :]
            d_kernel += np.einsum("hwc,k->hwck", patch, d_out[i, j, :])
            d_x_pad[i * sH: i * sH + kH,
                    j * sW: j * sW + kW, :] += np.einsum(
                "hwck,k->hwc", kernel, d_out[i, j, :]
            )

    H, W = x.shape[:2]
    if padding == "same" and (pad_top + pad_bottom + pad_left + pad_right) > 0:
        d_x = d_x_pad[pad_top: pad_top + H, pad_left: pad_left + W, :]
    else:
        d_x = d_x_pad

    return d_x.astype(np.float32), d_kernel.astype(np.float32), d_bias.astype(np.float32)
