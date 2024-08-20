import numpy as np

from binarizer import Bin, Binarizer


def our_metric(truth: np.ndarray[bool], estimate: np.ndarray[bool]) -> float:
    # trunk-ignore(bandit/B101)
    assert (
        truth.shape == estimate.shape
    ), "Classification and ground truth have different shape"

    if truth.dtype != bool:
        print("assuming truth values are 0, 1")
        truth = truth == 1
    if estimate.dtype != bool:
        print("assuming estimate values are 0, 1")
        estimate = estimate == 1

    n_pos = np.sum(truth)
    n_neg = truth.shape[0] - n_pos
    errors = truth != estimate
    return 1 - (np.sum(errors[truth]) / n_pos) - (np.sum(errors[~truth]) / n_neg)


def accuracy(truth: np.ndarray[bool], estimate: np.ndarray[bool]) -> float:
    # trunk-ignore(bandit/B101)
    assert (
        truth.shape == estimate.shape
    ), "Classification and ground truth have different shape"

    if truth.dtype != bool:
        print("assuming truth values are 0, 1")
        truth = truth == 1
    if estimate.dtype != bool:
        print("assuming estimate values are 0, 1")
        estimate = estimate == 1

    return np.mean(truth == estimate)


def balance_datasets(
    y: np.ndarray[bool], datasets: list[np.ndarray], seed: int
) -> list[np.ndarray]:
    pos = np.sum(y)
    neg = y.shape[0] - pos
    np.random.seed(seed)
    if pos > neg:
        to_drop = np.random.choice(np.where(y)[0], size=(pos - neg,), replace=False)
    elif neg > pos:
        to_drop = np.random.choice(np.where(~y)[0], size=(neg - pos,), replace=False)
    else:
        to_drop = []
    keep_mask = np.ones_like(y, dtype=bool)
    keep_mask[to_drop] = False

    pruned_datasets = [d[keep_mask] for d in datasets]
    return pruned_datasets


def _eval_term(term: list[Bin], binarizer: Binarizer, X_test: np.ndarray[bool]):
    mask = np.ones((X_test.shape[0],), dtype=bool)
    bin_feats = [str(f) for f in binarizer.get_bin_encodings(include_negations=True)]
    for feat in term:
        feat_i = bin_feats.index(str(feat))
        mask &= X_test[:, feat_i]
    return mask


def eval_terms(dnf: list[list[Bin]], binarizer: Binarizer, X_test: np.ndarray[bool]):
    masks = [_eval_term(term, binarizer, X_test) for term in dnf]
    return masks


def _format_term(term, eval):
    return "(" + " AND ".join(map(str, term)) + f")    --- (our objective: {eval})"


def print_dnf(
    dnf: list[list[Bin]], binarizer: Binarizer, term_evals: np.ndarray[float]
):
    positive, negative = binarizer.target_name()
    # term_strs = [_format_term(term, eval) for term, eval in zip(dnf, term_evals)]
    term_strs = ["(" + " AND ".join(map(str, term)) + ")" for term in dnf]
    maxlen = max(map(len, term_strs))
    term_strs = [
        term + " " * (maxlen - len(term)) + f" <-- (term's our objective: {eval})"
        for term, eval in zip(term_strs, term_evals)
    ]

    print(
        "IF \n    " + "\n OR ".join(term_strs) + f"\nTHEN\n {positive} ELSE {negative}",
    )
