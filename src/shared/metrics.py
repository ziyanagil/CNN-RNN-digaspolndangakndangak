import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import f1_score

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro")

def bleu4(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    refs_nltk = []
    for r in references:
        if isinstance(r[0], str):
            refs_nltk.append([r])
        else:
            refs_nltk.append(r)

    smoothie = SmoothingFunction().method4
    return corpus_bleu(
        refs_nltk,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie,
    )

def meteor(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    from nltk.translate.meteor_score import meteor_score

    scores = [
        meteor_score([ref], hyp)
        for ref, hyp in zip(references, hypotheses)
    ]
    return float(np.mean(scores))
