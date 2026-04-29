import numpy as np

def conv2d_backward(
    d_out: np.ndarray,
    x: np.ndarray,
    kernel: np.ndarray,
    strides: tuple = (1, 1),
    padding: str = "valid",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # TODO: implementasi backprop conv2d
    raise NotImplementedError
