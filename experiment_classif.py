import os
import pickle
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dnf_fair_classifier import DNFFairClassifier
from gerryfair.model import Auditor, Model
from linear_fair_classifier import LinearFairClassifier
from NN_fair_classifier import NNFairClassifier, SimpleDataset
from scenarios.folktables_scenarios import load_classif_scenario
from spsf_mio import SPSF
from utils import eval_fpsf, eval_spsf

githash = ""


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    binarizer, dhandler, X_orig, y_orig, binarizer_protected, X_prot_orig = (
        load_classif_scenario(cfg.scenario, cfg.seed, cfg.n_samples)
    )

    n_samples = X_orig.shape[0]
    X = binarizer.encode(X_orig, include_negations=False)
    X_prot = binarizer_protected.encode(X_prot_orig, include_negations=False)
    y = binarizer.encode_y(y_orig)

    X_enc = dhandler.encode(X_orig)

    dfX_enc = pd.DataFrame(X_enc)
    dfX_prot = pd.DataFrame(X_prot)
    dfy = pd.Series(y)

    if cfg.model == "DNF":
        mio_setup = DNFFairClassifier(gamma=0.01)
        dnf_model = mio_setup.find_dnf(
            X, X_prot, y, n_terms=5, time_limit=cfg.time_limit, verbose=True
        )

        y_hat_train = np.zeros_like(y, dtype=bool)
        for term in dnf_model:
            y_term = np.ones_like(y, dtype=bool)
            for conj in term:
                y_term &= X[:, conj]
            y_hat_train |= y_term
        y_hat_train_prob = y_hat_train.astype(int)
    elif cfg.model == "Linear":
        mio_setup = LinearFairClassifier(gamma=0.01)
        coefs, threshold = mio_setup.find_classifier(
            X_enc, X_prot, y, time_limit=cfg.time_limit, epsilon=1e-3, verbose=True
        )
        y_hat_train = X_enc @ coefs.reshape((-1, 1)) >= threshold
        y_hat_train = y_hat_train.flatten()
        y_hat_train_prob = y_hat_train.astype(int)
    elif cfg.model == "NN":
        # alpha = 1/gamma
        NN = NNFairClassifier(X_enc.shape[1], [500, 200, 50, 10], gamma=0.01, alpha=100)
        np.random.seed(cfg.seed)
        eval_idx = np.random.choice(n_samples, n_samples // 10, replace=False)
        eval_mask = np.zeros_like(y, dtype=bool)
        eval_mask[eval_idx] = True
        train = SimpleDataset(X_enc[~eval_mask], X_prot[~eval_mask], y[~eval_mask])
        eval = SimpleDataset(X_enc[eval_mask], X_prot[eval_mask], y[eval_mask])
        NN.train(train, eval, batch_size=2000, fpsf_size=20000, epochs=10)
        y_hat_train_prob = NN.predict_proba(X_enc)
        y_hat_train = y_hat_train_prob >= 0.5
    elif cfg.model == "GerryFair":
        gerryfair_model = Model(printflag=True, gamma=0.01, fairness_def="FP")
        n_iters = cfg.time_limit // 5
        gerryfair_model.set_options(max_iters=n_iters)
        gerryfair_model.train(dfX_enc, dfX_prot, dfy)

        y_hat_train_prob = np.array(gerryfair_model.predict(dfX_enc))
        y_hat_train = y_hat_train_prob >= 0.5
    elif cfg.model == "FAMS":
        raise NotImplementedError("TODO: implement the FAMS")
        # y_hat_train variable must be set, a numpy array with bool values representing classifications - True for 1
        # y_hat_train_prob variable must be set as well, containing the probability of a positive classification (if model does not give probability, make it same as y_hat_train, but with ints)
    else:
        raise ValueError(f"Unknown fair classifier {cfg.model}")

    # Evaluate using gerryfair !and! SPSF_mio
    auditor = Auditor(dfX_prot, dfy, "FP")
    gerrygroup_train, oracle = auditor.audit(y_hat_train_prob, with_group_def=True)
    gerrygroup_train = np.array(gerrygroup_train, dtype=bool)

    spsf = SPSF()
    if n_samples > 20000:
        np.random.seed(cfg.seed)
        eval_idx = np.random.choice(n_samples, 10000, replace=False)
        mask = y[eval_idx] == 0
        group_rule = spsf.find_rule(X_prot[eval_idx][mask], y_hat_train[eval_idx][mask])
    else:
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
        out_file.write("RESULT\n")
        if cfg.model == "DNF":
            out_file.write(f"DNF: {dnf_model} \n")
        if cfg.model == "Linear":
            out_file.write(f"Classif Coefs: {list(coefs)} \n")
            out_file.write(f"Classif Threshold: {threshold} \n")
        if cfg.model in ["DNF", "Linear"]:
            out_file.write(f"Status: {mio_setup.mio_result.solver.status} \n")
            out_file.write(
                f"Termination Condition: {mio_setup.mio_result.solver.termination_condition} \n"
            )
            out_file.write(f"Number of cuts: {mio_setup.n_cuts} \n")
            out_file.write(f"Number of callbacks: {mio_setup.n_callbacks} \n")
            out_file.write(
                f"Time in callbacks proportion: {mio_setup.callback_time_proportion} \n"
            )
        elif cfg.model == "NN":
            out_file.write(f"Number of subgroups: {NN.n_subgroups} \n")
            out_file.write(f"List of subgroups: {NN.subgroups} \n")
            out_file.write(f"Number of checks: {NN.n_FPSF_checks} \n")
        out_file.write(f"Accuracy: {np.mean(y_hat_train == y)} \n")
        out_file.write(f"MIO group: {group_rule} \n")
        out_file.write(f"MIO SPSF: {eval_spsf(y_hat_train, miogroup_train)} \n")
        out_file.write(f"MIO FPSF: {eval_fpsf(y, y_hat_train, miogroup_train)} \n")
        out_file.write(f"Gerry oracle b0 coef: {list(oracle.b0.coef_)} \n")
        out_file.write(f"Gerry oracle b0 intercept: {oracle.b0.intercept_} \n")
        out_file.write(f"Gerry oracle b1 coef: {list(oracle.b1.coef_)} \n")
        out_file.write(f"Gerry oracle b1 intercept: {oracle.b1.intercept_} \n")
        out_file.write(f"Gerry SPSF: {eval_spsf(y_hat_train, gerrygroup_train)} \n")
        out_file.write(f"Gerry FPSF: {eval_fpsf(y, y_hat_train, gerrygroup_train)} \n")
        out_file.write(f"Protected dimension: {X_prot.shape[1]} \n")
        out_file.write(f"Full dimension: {X.shape[1]} \n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")
    if cfg.model == "GerryFair":
        with open(os.path.join(run_dir, "gerrymodel.pickle"), "wb") as f:
            pickle.dump(gerryfair_model, f)
    elif cfg.model == "NN":
        path = os.path.join(run_dir, "NN.pth")
        NN.save_model(path)


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
