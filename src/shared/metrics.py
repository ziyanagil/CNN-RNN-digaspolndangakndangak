import numpy as np
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")


def bleu4(references: list[list[list[str]]], hypotheses: list[list[str]]) -> float:
    smoothie = SmoothingFunction().method4
    return corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )


def meteor(references: list[list[list[str]]], hypotheses: list[list[str]]) -> float:
    scores = [
        meteor_score(refs, hyp)
        for refs, hyp in zip(references, hypotheses)
    ]
    return float(np.mean(scores))
