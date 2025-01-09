# python experiment.py -m

import os
import subprocess
import sys

import hydra
import numpy as np
import pandas as pd
from folktables import ACSDataSource
from omegaconf import DictConfig

from binarizer import Binarizer
from data_handler import DataHandler
from gerryfair.model import Auditor, Model
from utils import eval_fpsf, eval_spsf

githash = ""


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    if cfg.scenario == "ACSIncome":
        from folktables import ACSIncome as Dataset
    elif cfg.scenario == "ACSPublicCoverage":
        from folktables import ACSPublicCoverage as Dataset
    elif cfg.scenario == "ACSMobility":
        from folktables import ACSMobility as Dataset
    elif cfg.scenario == "ACSEmployment":
        from folktables import ACSEmployment as Dataset
    elif cfg.scenario == "ACSTravelTime":
        from folktables import ACSTravelTime as Dataset
    else:
        raise ValueError(f'Scenario "{cfg.scenario}" does not exist.')

    # TODO make the configuration parameterized
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    data = data_source.get_data(states=["CA"], download=True)
    input_data, target_data, _ = Dataset.df_to_pandas(data)

    # DROP COLS WITH TOO MANY OPTIONS
    to_drop = []
    for col in input_data.columns:
        vals = input_data[col].unique().shape[0]
        if vals > 5 or vals <= 1:
            to_drop.append(col)
    input_data.drop(columns=to_drop, inplace=True)

    np.random.seed(cfg.seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(cfg.n_samples, n), replace=False)

    target_data = target_data[target_data.columns[0]]

    train_data = input_data.iloc[samples]
    train_y_data = target_data.iloc[samples]
    dhandler = DataHandler.from_data(
        input_data, target_data, categ_map={c: [] for c in input_data.columns}
    )
    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    X_test = pd.DataFrame(binarizer.encode(input_data, include_negations=False))
    y_test = pd.Series(binarizer.encode_y(target_data))
    X = pd.DataFrame(binarizer.encode(train_data, include_negations=False))
    y = pd.Series(binarizer.encode_y(train_y_data))

    fair_model = Model(printflag=True, gamma=0.01, fairness_def="FP")
    fair_model.set_options(max_iters=30)
    # train_data.index = np.arange(train_data.index.shape[0])
    # train_y_data.index = np.arange(train_data.index.shape[0])
    # fair_model.train(train_data, train_data, train_y_data)
    fair_model.train(X, X, y)

    # y_hat_train = fair_model.predict(train_data)
    # auditor = Auditor(train_data, train_y_data, "FP")
    # violated_group_train, _ = auditor.audit(y_hat_train)
    y_hat_train = fair_model.predict(X)
    auditor = Auditor(X, y, "FP")
    violated_group_train, _ = auditor.audit(y_hat_train)

    y_hat = fair_model.predict(X_test)
    auditor = Auditor(X_test, y_test, "FP")
    violated_group, _ = auditor.audit(y_hat)

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        print(f"Config:\n {cfg}", file=sys.stderr)
        out_file.write(f"Config:\n {cfg}\n")
        out_file.write(f"\nGit hash: {githash}\n\n")

        out_file.write("RESULT\n")
        out_file.write(
            f"Train Accuracy: {np.mean(np.array(y_hat_train,dtype=bool) == y.values)} \n"
        )
        out_file.write(
            f"Train SPSF: {eval_spsf(np.array(y_hat_train,dtype=bool), np.array(violated_group_train,dtype=bool))} \n"
        )
        out_file.write(
            f"Train FPSF: {eval_fpsf(y.values, np.array(y_hat_train,dtype=bool), np.array(violated_group_train,dtype=bool))} \n"
        )
        out_file.write(
            f"Test Accuracy: {np.mean(np.array(y_hat,dtype=bool) == y_test.values)} \n"
        )
        out_file.write(
            f"Test SPSF: {eval_spsf(np.array(y_hat,dtype=bool), np.array(violated_group,dtype=bool))} \n"
        )
        out_file.write(
            f"Test FPSF: {eval_fpsf(y_test.values, np.array(y_hat,dtype=bool), np.array(violated_group,dtype=bool))} \n"
        )

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
