import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
