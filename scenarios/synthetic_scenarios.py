# import pandas as pd
import numpy as np

from binarizer import Binarizer
from data_handler import DataHandler

SCENARIOS = [
    "smallest_subclass",
]


def nums_to_bin(vals: np.ndarray[int], dim: int) -> np.ndarray[bool]:
    binvecs = np.zeros((vals.shape[0], dim), dtype=bool)
    for i in reversed(range(dim)):
        binvecs[:, i] = vals % 2 == 1
        vals = vals // 2
    return binvecs


def smallest_subclass(rho: float, dimension: int, n_samples: int, seed: int):
    np.random.seed(seed)
    d = dimension
    n_mu = n_samples // 2
    mu_probs = np.full((2**d,), (1 - rho / (2 ** (d - 1))) / (2**d - 1))
    mu_probs[0] = rho / (2 ** (d - 1))
    mu_samples = np.random.choice(np.arange(2**d), n_mu, replace=True, p=mu_probs)

    nu_probs = np.full((2**d,), (1 - (1 - rho) / (2 ** (d - 1))) / (2**d - 1))
    nu_probs[0] = (1 - rho) / (2 ** (d - 1))
    nu_samples = np.random.choice(
        np.arange(2**d), n_samples - n_mu, replace=True, p=nu_probs
    )

    print(f"The true sup(\\mu - \\nu) = {mu_probs[0]-nu_probs[0]}")
    print(
        f"The correct rule has \\hat{{\\mu}} - \\hat{{\\nu}} = {np.mean(mu_samples==0) - np.mean(nu_samples==0)}"
    )

    mu_data = nums_to_bin(mu_samples, dimension)
    nu_data = nums_to_bin(nu_samples, dimension)
    input_data = np.vstack([mu_data, nu_data])

    target_data = np.vstack([np.ones((n_mu, 1)), np.zeros((n_samples - n_mu, 1))])

    colnames = [f"x{i}" for i in range(dimension)]
    dhandler = DataHandler.from_data(
        input_data,
        target_data,
        categ_map={c: [] for c in colnames},
        feature_names=colnames,
    )

    binarizer = Binarizer(dhandler)
    return binarizer, input_data, target_data


def sample_scenario(name, dimension, n_samples, seed, **kwargs):
    if name == "smallest_subclass":
        rho = 0.8
        if "rho" in kwargs:
            rho = kwargs["rho"]
        return smallest_subclass(rho, dimension, n_samples, seed)
