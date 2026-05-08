import numpy as np
from dataclasses import dataclass, field
import copy


@dataclass
class BeamHypothesis:
    tokens: list = field(default_factory=list)
    score: float = 0.0
    h_states: list = field(default_factory=list)
    c_states: list = field(default_factory=list)


def _rnn_step(decoder, token: int, h_states: list) -> tuple:
    x_t = decoder.embedding.forward(np.array(token))
    new_h = []
    for i, cell in enumerate(decoder.rnn.cells):
        h = cell.forward(x_t, h_states[i])
        new_h.append(h)
        x_t = h
    logits = decoder.output_dense.forward(x_t)
    return logits, new_h


def _lstm_step(decoder, token: int, h_states: list, c_states: list) -> tuple:
    x_t = decoder.embedding.forward(np.array(token))
    new_h, new_c = [], []
    for i, cell in enumerate(decoder.lstm.cells):
        h, c = cell.forward(x_t, h_states[i], c_states[i])
        new_h.append(h)
        new_c.append(c)
        x_t = h
    logits = decoder.output_dense.forward(x_t)
    return logits, new_h, new_c


def beam_search_rnn(
    decoder,
    feature_vec: np.ndarray,
    vocab: dict,
    idx2word: dict,
    beam_size: int = 5,
    max_len: int = 30,
) -> str:
    start_idx = vocab["<start>"]
    end_idx = vocab["<end>"]
    is_lstm = hasattr(decoder, "lstm")

    x_minus1 = decoder.feature_projection.forward(feature_vec)

    if is_lstm:
        h_init = [np.zeros(cell.units, dtype=np.float32) for cell in decoder.lstm.cells]
        c_init = [np.zeros(cell.units, dtype=np.float32) for cell in decoder.lstm.cells]
        x_t = x_minus1
        for i, cell in enumerate(decoder.lstm.cells):
            h, c = cell.forward(x_t, h_init[i], c_init[i])
            h_init[i] = h
            c_init[i] = c
            x_t = h
        init_hyp = BeamHypothesis(
            tokens=[start_idx],
            score=0.0,
            h_states=copy.deepcopy(h_init),
            c_states=copy.deepcopy(c_init),
        )
    else:
        h_init = [np.zeros(cell.units, dtype=np.float32) for cell in decoder.rnn.cells]
        x_t = x_minus1
        for i, cell in enumerate(decoder.rnn.cells):
            h_init[i] = cell.forward(x_t, h_init[i])
            x_t = h_init[i]
        init_hyp = BeamHypothesis(
            tokens=[start_idx],
            score=0.0,
            h_states=copy.deepcopy(h_init),
        )

    active = [init_hyp]
    completed = []

    for _ in range(max_len):
        if not active:
            break

        candidates = []
        for hyp in active:
            last_token = hyp.tokens[-1]

            if is_lstm:
                logits, new_h, new_c = _lstm_step(decoder, last_token, hyp.h_states, hyp.c_states)
            else:
                logits, new_h = _rnn_step(decoder, last_token, hyp.h_states)

            log_probs = np.log(np.clip(logits, 1e-10, 1.0))
            top_k_idx = np.argpartition(log_probs, -beam_size)[-beam_size:]

            for tok in top_k_idx:
                tok = int(tok)
                new_score = hyp.score + log_probs[tok]
                new_tokens = hyp.tokens + [tok]

                if is_lstm:
                    cand = BeamHypothesis(
                        tokens=new_tokens,
                        score=new_score,
                        h_states=copy.deepcopy(new_h),
                        c_states=copy.deepcopy(new_c),
                    )
                else:
                    cand = BeamHypothesis(
                        tokens=new_tokens,
                        score=new_score,
                        h_states=copy.deepcopy(new_h),
                    )
                candidates.append(cand)

        candidates.sort(key=lambda h: h.score, reverse=True)
        candidates = candidates[:beam_size]

        active = []
        for cand in candidates:
            if cand.tokens[-1] == end_idx:
                completed.append(cand)
            else:
                active.append(cand)

    completed.extend(active)
    if not completed:
        return ""

    best = max(completed, key=lambda h: h.score / max(len(h.tokens) - 1, 1))

    words = []
    for tok in best.tokens[1:]:
        if tok == end_idx:
            break
        word = idx2word.get(tok, "<unk>")
        if word not in ("<pad>", "<start>", "<unk>"):
            words.append(word)
    return " ".join(words)
