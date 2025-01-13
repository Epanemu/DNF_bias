import os
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dnf_fair_classifier import DNFFairClassifier
from gerryfair.auditor import Auditor
from scenarios.folktables_scenarios import load_classif_scenario
from spsf_mio import SPSF
from utils import eval_fpsf, eval_spsf

githash = ""


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    binarizer, X_orig, y_orig, binarizer_protected, X_prot_orig = load_classif_scenario(
        cfg.scenario, cfg.seed, cfg.n_samples
    )

    X = binarizer.encode(X_orig, include_negations=False)
    X_prot = binarizer_protected.encode(X_prot_orig, include_negations=False)
    y = binarizer.encode_y(y_orig)

    dnf = DNFFairClassifier(gamma=0.01)

    result = dnf.find_dnf(X, X_prot, y, n_terms=5, time_limit=300, verbose=True)
    y_hat_train = np.zeros_like(y, dtype=bool)
    for term in result:
        y_term = np.ones_like(y, dtype=bool)
        for conj in term:
            y_term &= X[:, conj]
        y_hat_train |= y_term

    # Evaluate using gerryfair !and! SPSF_mio
    auditor = Auditor(pd.DataFrame(X_prot), pd.Series(y), "FP")
    gerrygroup_train, oracle = auditor.audit(
        pd.Series(y_hat_train), with_group_def=True
    )

    spsf = SPSF()
    mask = y == 0
    group_rule = spsf.find_rule(X_prot[mask], y_hat_train[mask])

    miogroup_train = np.ones_like(y, dtype=bool)
    for feat_i in group_rule:
        miogroup_train &= X_prot[:, feat_i]

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {githash}\n\n")
        if result is not None:
            out_file.write("RESULT\n")
            out_file.write(f"DNF: {result} \n")
            out_file.write(f"Status: {dnf.mio_result.solver.status} \n")
            out_file.write(
                f"Termination Condition: {dnf.mio_result.solver.termination_condition} \n"
            )
            out_file.write(f"Number of cuts: {dnf.n_cuts} \n")
            out_file.write(f"Number of callbacks: {dnf.n_callbacks} \n")
            out_file.write(
                f"Time in callbacks proportion: {dnf.callback_time_proportion} \n"
            )
            out_file.write(f"Accuracy: {np.mean(y_hat_train == y)} \n")
            out_file.write(f"MIO group: {group_rule} \n")
            out_file.write(f"MIO SPSF: {eval_spsf(y_hat_train, miogroup_train)} \n")
            out_file.write(f"MIO FPSF: {eval_fpsf(y, y_hat_train, miogroup_train)} \n")
            out_file.write(f"Gerry oracle b0 coef: {list(oracle.b0.coef_)} \n")
            out_file.write(f"Gerry oracle b0 intercept: {oracle.b0.intercept_} \n")
            out_file.write(f"Gerry oracle b1 coef: {list(oracle.b1.coef_)} \n")
            out_file.write(f"Gerry oracle b1 intercept: {oracle.b1.intercept_} \n")
            out_file.write(f"Gerry SPSF: {eval_spsf(y_hat_train, gerrygroup_train)} \n")
            out_file.write(
                f"Gerry FPSF: {eval_fpsf(y, y_hat_train, gerrygroup_train)} \n"
            )
        else:
            out_file.write("Error\n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")


if __name__ == "__main__":
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout.strip() == "":
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        )
        githash = res.stdout.strip()
        run_experiment()
    else:
        raise Exception("Git status is not clean. Commit changes first.")
