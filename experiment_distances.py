import logging
import os
import subprocess
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig

from methods import test_BRCG, test_RIPPER
from one_rule import OneRule
from scenarios.folktables_scenarios import load_scenario
from utils import (
    TV_binarized,
    balance_datasets,
    eval_terms,
    our_metric,
    wasserstein_distance,
)

gitcommit = ""

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="conf", config_name="distances")
def run_experiment(cfg: DictConfig):
    binarizer, dhandler, X_orig, y_orig, binarizer_protected, X_prot_orig = (
        load_scenario(cfg.scenario, cfg.seed, cfg.n_samples, state=cfg.state)
    )

    # cfg needs to have
    # state, scenario, seed, n_samples ~ total n of samples, train_samples, model

    n_samples = X_orig.shape[0]
    n_train_samples = cfg.train_samples
    while n_train_samples > n_samples:
        n_train_samples = n_train_samples // 2
    train_i = np.random.choice(n_samples, size=n_train_samples, replace=False)
    train_mask = np.zeros((n_samples,), dtype=bool)
    train_mask[train_i] = True

    X = binarizer.encode(X_orig[train_mask], include_negations=False)
    X_prot = binarizer_protected.encode(
        X_prot_orig[train_mask], include_negations=False, include_binary_negations=False
    )
    # for the MIO we need a binary feature for each possible value
    X_prot_full = binarizer_protected.encode(
        X_prot_orig[train_mask], include_negations=False, include_binary_negations=True
    )
    # for evaluating Ripper, we need all negations
    X_prot_ripper_eval = binarizer_protected.encode(
        X_prot_orig[train_mask], include_negations=True
    )
    y = binarizer.encode_y(y_orig[train_mask])
    # X_enc = dhandler.encode(X_orig[train_mask])

    n_samples, d = X.shape
    d_prot = X_prot.shape[1]

    t_start = time.time()

    opt = True
    if cfg.model == "OneRule":
        onerule = OneRule()
        term, opt = onerule.find_rule(
            X_prot_full,
            y,
            verbose=True,
            time_limit=cfg.time_limit,
            return_opt_flag=True,
        )
        y_hat = np.ones_like(y, dtype=bool)
        for conj in term:
            y_hat &= X_prot_full[:, conj]
        dist = our_metric(y, y_hat)
        d = X_prot_full.shape[1]
    elif cfg.model == "Ripper":
        y, X_prot, X_prot_ripper_eval = balance_datasets(
            y, [y, X_prot, X_prot_ripper_eval], seed=cfg.seed
        )
        n_samples, d = X_prot.shape
        y_hat, dnf = test_RIPPER(X_prot, y, X_prot, binarizer_protected)
        y_hat_true = eval_terms(dnf, binarizer_protected, X_prot_ripper_eval)[0]
        if not (np.array(y_hat) == y_hat_true).all():
            logger.warning("There is an issue in the RIPPER changes")
        dist = our_metric(y, y_hat_true)
    elif cfg.model == "BRCG":
        y, X_prot_full = balance_datasets(y, [y, X_prot_full], seed=cfg.seed)
        n_samples, d = X_prot_full.shape
        _, dnf = test_BRCG(X_prot_full, y, X_prot_full, binarizer_protected)
        # y_hat is not correct for the returned single conjuntion
        # print((y_hat == eval_terms(dnf, binarizer_protected, X_prot_full)[0]).all())
        y_hat = eval_terms(
            dnf, binarizer_protected, X_prot_full, binary_negs_only=True
        )[0]
        dist = our_metric(y, y_hat)
    elif cfg.model in ["W1", "W2"]:
        d = X_prot.shape[1]
        X0 = X_prot[~y].astype(float)
        X1 = X_prot[y].astype(float)
        dist = wasserstein_distance(X0, X1, Wtype=cfg.model)
    elif cfg.model == "TV":
        d = X_prot.shape[1]
        X0 = X_prot[~y].astype(float)
        X1 = X_prot[y].astype(float)
        dist = TV_binarized(X0, X1)
    elif cfg.model == "MMD":
        pass
    else:
        raise ValueError(f"Unknown fair classifier {cfg.model}")

    t_tot = time.time() - t_start

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {gitcommit}\n\n")
        out_file.write("RESULT\n")
        out_file.write(f"Distance reported: {dist} \n")
        out_file.write(f"Time spent: {t_tot} \n")
        out_file.write(f"Optimal/Valid flag: {opt} \n")
        out_file.write(f"True number of training samples: {n_samples} \n")
        out_file.write(f"Protected dimension: {d_prot} \n")
        out_file.write(f"Full dimension: {d} \n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")


if __name__ == "__main__":
    # result = subprocess.run(
    #     ["git", "status", "--porcelain"], capture_output=True, text=True
    # )
    # if result.stdout.strip() == "":
    #     res = subprocess.run(
    #         ["git", "rev-list", "--format=%B", "-n", "1", "HEAD"],
    #         capture_output=True,
    #         text=True,
    #     )
    #     gitcommit = res.stdout.strip()
    if True:
        run_experiment()
    else:
        raise Exception("Git status is not clean. Commit changes first.")
