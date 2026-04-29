import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def bleu4(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    # TODO: implementasi dengan nltk.translate.bleu_score
    raise NotImplementedError


def meteor(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    # TODO: implementasi dengan nltk.translate.meteor_score
    raise NotImplementedError
