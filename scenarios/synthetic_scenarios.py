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


def sample_scenario(name, dimension, n_samples, seed, **kwargs):
    rho = 0.8
    if "rho" in kwargs:
        rho = kwargs["rho"]
    if name == "smallest_subclass":
        k = dimension
    elif name == "linear_dependence":
        k = np.round(np.log2(dimension)).astype(int)
    elif name == "constant_subclass":
        k = min(3, dimension)
        if "rho" in kwargs:
            k = kwargs["k"]
    else:
        raise AttributeError(f"name must be one of {SCENARIOS}")
    binarizer, input_data, target_data = sample_with_fixed_zeros(
        rho, dimension, k, n_samples, seed
    )
    true_term = [b.negate_self() for b in binarizer.get_bin_encodings()[:k]]
    return (binarizer, input_data, target_data, true_term)
