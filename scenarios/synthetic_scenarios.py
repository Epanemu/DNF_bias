# import pandas as pd
import numpy as np

from binarizer import Binarizer
from data_handler import DataHandler

SCENARIOS = [
    "smallest_subclass",
    "linear_dependence",
    "constant_subclass",
]


def nums_to_bin(vals: np.ndarray[int], dim: int) -> np.ndarray[bool]:
    binvecs = np.zeros((vals.shape[0], dim), dtype=bool)
    for i in reversed(range(dim)):
        binvecs[:, i] = vals % 2 == 1
        vals = vals // 2
    return binvecs


def sample_with_fixed_zeros(
    rho: float, dimension: int, fixed_zeros: int, n_samples: int, seed: int
):
    np.random.seed(seed)
    d = dimension
    k = fixed_zeros
    assert d >= k, "Cannot fix more features than there are dimensions."
    n_mu = n_samples // 2
    n_shifted = 2 ** (d - k)

    mu_probs = np.full((2**d,), (1 - rho / (2 ** (k - 1))) / (2**d - n_shifted))
    mu_probs[:n_shifted] = rho / (2 ** (d - 1))
    mu_samples = np.random.choice(np.arange(2**d), n_mu, replace=True, p=mu_probs)

    nu_probs = np.full((2**d,), (1 - (1 - rho) / (2 ** (k - 1))) / (2**d - n_shifted))
    nu_probs[:n_shifted] = (1 - rho) / (2 ** (d - 1))
    nu_samples = np.random.choice(
        np.arange(2**d), n_samples - n_mu, replace=True, p=nu_probs
    )

    # TODO logger
    print(f"The true theoretical sup(\\mu - \\nu) = {(2*rho - 1) / (2 ** (k-1))}")
    objective = np.mean(mu_samples < n_shifted) - np.mean(nu_samples < n_shifted)
    print(
        f"The correct rule on sampled data has \\hat{{\\mu}} - \\hat{{\\nu}} = {objective}"
    )

    mu_data = nums_to_bin(mu_samples, dimension)
    nu_data = nums_to_bin(nu_samples, dimension)
    input_data = np.vstack([mu_data, nu_data])

    target_data = np.vstack([np.ones((n_mu, 1)), np.zeros((n_samples - n_mu, 1))])

    colnames = [f"x{i}" for i in range(dimension)]
    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map={c: [0, 1] for c in colnames},
        feature_names=colnames,
    )

    binarizer = Binarizer(dhandler)
    return binarizer, input_data, target_data


def sample_intersectional(
    rho: float, dimension: int, fixed_dims: int, n_samples: int, seed: int
):
    np.random.seed(seed)
    fix_vector = np.random.random_integers(0, 1, fixed_dims)
    X = np.random.random_integers(0, 1, (n_samples, dimension))
    positive = X[:, :fixed_dims] == fix_vector
    negative = X[:, :fixed_dims] == 1 - fix_vector
    y = np.random.choice([0, 1], (n_samples, 1), p=[0.5, 0.5])
    y[positive] = np.random.choice([0, 1], (np.sum(positive), 1), p=[1 - rho, rho])
    y[negative] = np.random.choice([0, 1], (np.sum(negative), 1), p=[rho, 1 - rho])

    colnames = [f"x{i}" for i in range(dimension)]
    dhandler = DataHandler.from_data(
        X,
        y,
        categ_map={c: [0, 1] for c in colnames},
        feature_names=colnames,
    )

    binarizer = Binarizer(dhandler)
    return binarizer, X, y


def sample_cummulative(
    rho: float, dimension: int, fixed_dims: int, n_samples: int, seed: int
):
    np.random.seed(seed)
    # TODO nonuniform variant?
    fix_vector = np.random.random_integers(0, 1, fixed_dims)
    X = np.random.random_integers(0, 1, (n_samples, dimension))
    n_correct_vals = np.zeros((n_samples,))
    for d in range(fixed_dims):
        n_correct_vals += X[:, d] == fix_vector[d]
    y = np.empty_like(n_correct_vals)
    for same_dims in range(fixed_dims + 1):
        mask = n_correct_vals == same_dims
        sigma = 1 - rho  # TODO alternatives?
        p_one = sigma + (same_dims * (rho - sigma)) / fixed_dims
        y[mask] = np.random.choice([0, 1], (np.sum(mask), 1), p=[1 - p_one, p_one])

    colnames = [f"x{i}" for i in range(dimension)]
    dhandler = DataHandler.from_data(
        X,
        y,
        categ_map={c: [0, 1] for c in colnames},
        feature_names=colnames,
    )

    binarizer = Binarizer(dhandler)
    return binarizer, X, y


def sample_scenario(name, dimension, n_samples, seed, **kwargs):
    rho = 0.8
    if "rho" in kwargs:
        rho = kwargs["rho"]
    if "k" in kwargs:
        k = kwargs["k"]
    if name in ["smallest_subclass", "linear_dependence", "constant_subclass"]:
        if name == "smallest_subclass":
            k = dimension
        elif name == "linear_dependence":
            k = np.round(np.log2(dimension)).astype(int)
        elif name == "constant_subclass" and "k" not in kwargs:
            k = min(3, dimension)
        binarizer, input_data, target_data = sample_with_fixed_zeros(
            rho, dimension, k, n_samples, seed
        )
        true_term = [b.negate_self() for b in binarizer.get_bin_encodings()[:k]]
    elif name == "intersectional":
        binarizer, input_data, target_data = sample_intersectional(
            rho, dimension, k, n_samples, seed
        )
    elif name == "cummulative":
        binarizer, input_data, target_data = sample_cummulative(
            rho, dimension, k, n_samples, seed
        )
    else:
        raise AttributeError(f"name must be one of {SCENARIOS}")
    # TODO permute the columns so that there is no advantage
    return (binarizer, input_data, target_data, true_term)
