# python experiment.py -m

import os
import subprocess
import sys

import hydra
import numpy as np
from folktables import ACSDataSource
from omegaconf import DictConfig

from binarizer import Binarizer
from data_handler import DataHandler
from dnf_fair_classifier import DNFFairClassifier

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

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        input_data, target_data, categ_map={c: [] for c in input_data.columns}
    )
    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    X = binarizer.encode(input_data, include_negations=True)
    y = binarizer.encode_y(target_data)

    dnf = DNFFairClassifier(gamma=0.01)

    result = dnf.find_dnf(X, y, n_terms=5, time_limit=300, verbose=True)

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
