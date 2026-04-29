import numpy as np
from src.shared.metrics import macro_f1


def evaluate_keras(model, test_ds) -> dict:
    # TODO: implementasi
    raise NotImplementedError


def evaluate_scratch(scratch_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    # TODO: implementasi
    raise NotImplementedError


def compare_outputs(keras_preds: np.ndarray, scratch_preds: np.ndarray) -> dict:
    # TODO: implementasi
    raise NotImplementedError
