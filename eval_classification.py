import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folktables import ACSDataSource

from binarizer import Binarizer
from data_handler import DataHandler
from gerryfair.model import Auditor
from scenarios.folktables_scenarios import SCENARIOS
from utils import eval_fpsf, eval_spsf

methods = [
    "nofairDNF",
    "fairDNF",
    "lpfairDNF",
]

base_dir_prefix = "multirun/2025-01-09/"

method_colors = {"nofairDNF": "red", "fairDNF": "blue", "lpfairDNF": "green"}
method_names = {
    "nofairDNF": "vanilla DNF",
    "fairDNF": "DNF + onerule cuts",
    "lpfairDNF": "DNF + LP onerule cuts",
}
method_paths = {"nofairDNF": "07-49-37", "fairDNF": "06-15-02", "lpfairDNF": "09-02-17"}


def prepare_data(scenario, seed, n_samples=1000, test=False):
    if scenario == "ACSIncome":
        from folktables import ACSIncome as Dataset
    elif scenario == "ACSPublicCoverage":
        from folktables import ACSPublicCoverage as Dataset
    elif scenario == "ACSMobility":
        from folktables import ACSMobility as Dataset
    elif scenario == "ACSEmployment":
        from folktables import ACSEmployment as Dataset
    elif scenario == "ACSTravelTime":
        from folktables import ACSTravelTime as Dataset
    else:
        raise ValueError(f'Scenario "{scenario}" does not exist.')

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

    np.random.seed(seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(n_samples, n), replace=False)

    train_input_data = input_data.iloc[samples]
    train_target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        train_input_data,
        train_target_data,
        categ_map={c: [] for c in train_input_data.columns},
    )
    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    X = binarizer.encode(train_input_data, include_negations=True)
    y = binarizer.encode_y(train_target_data)

    if not test:
        return X, y, None, None
    X_test = binarizer.encode(input_data, include_negations=True)
    y_test = binarizer.encode_y(target_data[target_data.columns[0]])

    return X, y, X_test, y_test


test_data = {}
train_data = {}
for s in SCENARIOS:
    train_data[s] = {}
    for seed in range(5):
        # X, y, Xt, yt = prepare_data(s, seed, test=seed == 0)
        X, y, Xt, yt = prepare_data(s, seed, test=False)
        train_data[s][seed] = X, y
        if seed == 0:
            test_data[s] = Xt, yt


def extract_data_for_method(method):
    base_dir = base_dir_prefix + method_paths[method]
    extracted_data = []

    for i in range(400):
        folder_path = os.path.join(base_dir, str(i))
        output_file = os.path.join(folder_path, "output.txt")

        if not os.path.isfile(output_file):
            continue

        with open(output_file, "r", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) < 2:
                continue

            command_line = lines[1].strip()

            n_samples_match = re.search(r"'n_samples': (\d+)", command_line)
            seed_match = re.search(r"'seed': (\d+)", command_line)
            scenario_match = re.search(r"'scenario': '(\S+)'", command_line)
            # n_subgroup_min = re.search(r"-nm (\d+)", command_line)
            # if "mio" in method and n_subgroup_min and n_subgroup_min.group(1) != "10":
            #     continue

            if n_samples_match and scenario_match and seed_match:
                # max_n_samples = int(n_samples_match.group(1))
                scenario = scenario_match.group(1)
                seed = int(seed_match.group(1))

                for line in lines:
                    # dnf_str = re.search(r"DNF: \[\[(\S+)\]\]", line)
                    if "DNF" in line:
                        dnf_str = line.strip()[7:-2]
                        dnf = [
                            [int(lit) for lit in term.split(",")]
                            for term in dnf_str.split("], [")
                        ]
                        X, y = train_data[scenario][seed]
                        y_hat = np.zeros_like(y, dtype=bool)
                        for term in dnf:
                            y_term = np.ones_like(y, dtype=bool)
                            for conj in term:
                                y_term &= X[:, conj]
                            y_hat |= y_term
                        auditor = Auditor(pd.DataFrame(X), pd.Series(y), "FP")
                        group, _ = auditor.audit(y_hat)

                        extracted_data.append(
                            ("Train Accuracy", scenario, np.mean(y == y_hat))
                        )
                        extracted_data.append(
                            (
                                "Train SPSF",
                                scenario,
                                eval_spsf(y_hat, np.array(group, dtype=bool)),
                            )
                        )
                        extracted_data.append(
                            (
                                "Train FPSF",
                                scenario,
                                eval_fpsf(y, y_hat, np.array(group, dtype=bool)),
                            )
                        )

                        if measure_test:
                            X, y = test_data[scenario]
                            y_hat = np.zeros_like(y, dtype=bool)
                            for term in dnf:
                                y_term = np.ones_like(y, dtype=bool)
                                for conj in term:
                                    y_term &= X[:, conj]
                                y_hat |= y_term
                            auditor = Auditor(X, y, "FP")
                            group, _ = auditor.audit(y_hat)

                            extracted_data.append(
                                ("Test Accuracy", scenario, np.mean(y == y_hat))
                            )
                            extracted_data.append(
                                (
                                    "Test SPSF",
                                    scenario,
                                    eval_spsf(y_hat, np.array(group, dtype=bool)),
                                )
                            )
                            extracted_data.append(
                                (
                                    "Test FPSF",
                                    scenario,
                                    eval_fpsf(y, y_hat, np.array(group, dtype=bool)),
                                )
                            )
                    cuts = re.search(r"Number of cuts: (\d+)", line)
                    if cuts:
                        extracted_data.append(
                            ("# Cuts", scenario, float(cuts.group(1)))
                        )
                    callbacks = re.search(r"Number of callbacks: (\d+)", line)
                    if callbacks:
                        extracted_data.append(
                            ("# Callbacks", scenario, float(callbacks.group(1)))
                        )

    data_dict = {}

    for (
        valname,
        scenario,
        val,
    ) in extracted_data:
        if valname not in data_dict:
            data_dict[valname] = defaultdict(list)
        data_dict[valname][scenario].append(val)

    return data_dict


measure_test = False
all_data = {}
for method in methods:
    print(method)
    all_data[method] = extract_data_for_method(method)

r, c = 3, 2
fig, axs = plt.subplots(r, c, figsize=(12, 12))

for method in methods:
    data_dict = all_data[method]
    for i, measure in enumerate(sorted(data_dict.keys())):
        ax = axs[i // c, i % c]

        names = sorted(data_dict[measure].keys())
        sorted_mean = [np.mean(data_dict[measure][n]) for n in names]
        sorted_std = [np.std(data_dict[measure][n]) for n in names]

        barwidth = 0.35
        shift = (barwidth / 2) if method == methods[0] else (-barwidth / 2)
        # barplot with std band for each method side by side in one axes
        ax.bar(
            np.arange(len(names)) + shift,
            sorted_mean,
            barwidth,
            yerr=sorted_std,
            capsize=5,
            color=method_colors[method],
            label=f"{method_names[method]} + std band",
        )

        ax.set_ylabel(measure)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=45)
        ax.grid(True, which="both", ls=":")
        ax.legend()

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_real_data.png"
plt.savefig(output_path)

plt.show()
