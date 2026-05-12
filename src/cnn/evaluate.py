import numpy as np
from sklearn.metrics import f1_score


def evaluate_keras(model, test_ds) -> dict:
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(y_batch.numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_scratch(scratch_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = scratch_model.predict(X_test)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return {
        "y_true": y_test,
        "y_pred": y_pred,
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    }


def compare_outputs(keras_preds: np.ndarray, scratch_preds: np.ndarray) -> dict:
    if keras_preds.ndim > 1:
        keras_preds = np.argmax(keras_preds, axis=1)
    if scratch_preds.ndim > 1:
        scratch_preds = np.argmax(scratch_preds, axis=1)
    match = np.sum(keras_preds == scratch_preds)
    return {
        "total": len(keras_preds),
        "match": int(match),
        "mismatch": int(len(keras_preds) - match),
        "agreement_rate": float(match / len(keras_preds)),
    }
