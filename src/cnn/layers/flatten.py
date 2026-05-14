import numpy as np

class Flatten:

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            return x.reshape(-1, order='C')
        return x.reshape(x.shape[0], -1, order='C')

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C) or any (N, ...) -> (N, -1)
        return X.reshape(X.shape[0], -1, order='C')
