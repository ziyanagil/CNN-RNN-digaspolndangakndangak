import numpy as np

class MaxPooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def load_weights(self, keras_layer) -> None:
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        pH, pW = self.pool_size
        sH, sW = self.strides
        H, W, C = x.shape
        
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        out = np.zeros((H_out, W_out, C), dtype=np.float32)
        
        for i in range(H_out):
            for j in range(W_out):
                out[i, j, :] = x[i*sH:i*sH+pH, j*sW:j*sW+pW, :].max(axis=(0, 1))
        return out


    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C)
        pH, pW = self.pool_size
        sH, sW = self.strides
        N, H, W, C = X.shape

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        out = np.zeros((N, H_out, W_out, C), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                out[:, i, j, :] = X[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :].max(axis=(1, 2))
        return out


class AveragePooling2D:
    def __init__(self, pool_size=(2, 2), strides=None):
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size

    def load_weights(self, keras_layer) -> None:
        cfg = keras_layer.get_config()
        self.pool_size = tuple(cfg["pool_size"])
        self.strides = tuple(cfg["strides"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        pH, pW = self.pool_size
        sH, sW = self.strides
        H, W, C = x.shape
        
        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        out = np.zeros((H_out, W_out, C), dtype=np.float32)
        
        for i in range(H_out):
            for j in range(W_out):
                out[i, j, :] = x[i*sH:i*sH+pH, j*sW:j*sW+pW, :].mean(axis=(0, 1))
        return out


    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C)
        pH, pW = self.pool_size
        sH, sW = self.strides
        N, H, W, C = X.shape

        H_out = (H - pH) // sH + 1
        W_out = (W - pW) // sW + 1
        out = np.zeros((N, H_out, W_out, C), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                out[:, i, j, :] = X[:, i*sH:i*sH+pH, j*sW:j*sW+pW, :].mean(axis=(1, 2))
        return out


class GlobalMaxPooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.max(axis=(0, 1))

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C) -> (N, C)
        return X.max(axis=(1, 2))


class GlobalAveragePooling2D:
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.mean(axis=(0, 1))

    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        # X: (N, H, W, C) -> (N, C)
        return X.mean(axis=(1, 2))
