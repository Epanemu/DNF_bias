import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "fairNN",
]

base_dir_prefix = "multirun/"

method_colors = {
    "GerryFair": "red",
    "fairDNF": "blue",
    "fairLinear": "green",
    "fairNN": "magenta",
}
method_names = {
    "GerryFair": "GerryFair",
    "fairDNF": "DNF via MIO with lazy FPSF",
    "fairLinear": "Linear via MIO with lazy FPSF",
    "fairNN": "NN with FPSF loss",
}
method_paths = {
    "GerryFair": {"folktables": "2025-01-16/23-43-03"},
    "fairDNF": {"folktables": "2025-01-17/06-05-57"},
    "fairLinear": {"folktables": "2025-01-16/23-43-00"},
    "fairNN": {"folktables": "2025-01-17/MSE"},
}


def extract_data_for_method(method):
    extracted_data = []
    for run in ["folktables"]:
        base_dir = base_dir_prefix + method_paths[method][run]

        for i in range(25):
            folder_path = os.path.join(base_dir, str(i))
            output_file = os.path.join(folder_path, "output.txt")
            log_file = os.path.join(folder_path, "experiment_classif.log")

            if not os.path.isfile(log_file):
                continue

            with open(output_file, "r", errors="ignore") as file:
                lines = file.readlines()

                setup_line = lines[1].strip()
                scenario_match = re.search(r"'scenario': '(\S+)'", setup_line)

                if scenario_match:
                    scenario = scenario_match.group(1)

            with open(log_file, "r", errors="ignore") as file:
                lines = file.readlines()
                if len(lines) < 2:
                    continue

                epoch = 0
                for line in lines:
                    ep = re.search(r"EPOCH (\d+)/", line)
                    if ep:
                        epoch = int(ep.group(1))
                    if "TRAIN:" in line:
                        data_part = "_train"
                    if "VALIDATION:" in line:
                        data_part = "_eval"
                    acc = re.search(r"Accuracy:\s*([0-9.]+)", line)
                    if acc:
                        extracted_data.append(
                            (
                                "Accuracy",
                                scenario + data_part,
                                epoch,
                                float(acc.group(1)) / 100,
                            )
                        )
                    bce = re.search(r"Avg BCE loss:\s*([0-9.]+)", line)
                    if bce:
                        extracted_data.append(
                            (
                                "BCE loss",
                                scenario + data_part,
                                epoch,
                                float(bce.group(1)),
                            )
                        )
                    fpsf = re.search(r"Avg FPSF loss:\s*([0-9.]+)", line)
                    if fpsf:
                        extracted_data.append(
                            (
                                "FPSF loss",
                                scenario + data_part,
                                epoch,
                                float(fpsf.group(1)),
                            )
                        )

    data_dict = {}

    for (
        valname,
        scenario,
        epoch,
        val,
    ) in extracted_data:
        if scenario not in data_dict:
            data_dict[scenario] = {}
        if valname not in data_dict[scenario]:
            data_dict[scenario][valname] = defaultdict(list)
        data_dict[scenario][valname][epoch].append(val)

    return data_dict


all_data = {}
for method in methods:
    print(method)
    all_data[method] = extract_data_for_method(method)

r, c = 2, 5
fig, axs = plt.subplots(r, c, figsize=(30, 12))

scenario_names = [
    "ACSIncome",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSEmployment",
    "ACSTravelTime",
]
scenarios = [s + "_train" for s in scenario_names] + [
    s + "_eval" for s in scenario_names
]

# scenarios = []
# for s in scenario_names:
#     scenarios.append(s + "_train")
#     scenarios.append(s + "_eval")

names = sorted(list(all_data[methods[0]][scenarios[0]].values())[0].keys())
measuers = [
    "Accuracy",
    "BCE loss",
    "FPSF loss",
]
measure_colors = {
    "Accuracy": "blue",
    "BCE loss": "green",
    "FPSF loss": "red",
}


mehtod = methods[0]
for j, scenario in enumerate(scenarios):
    ax = axs[j // c, j % c]
    data_dict = all_data[method][scenario]
    for _, measure in enumerate(measuers):
        # ax = axs[i // c, i % c]
        # ax = axs
        if measure not in data_dict:
            continue

        sorted_mean = [np.mean(data_dict[measure][n]) for n in names]
        sorted_std = [np.std(data_dict[measure][n]) for n in names]

        n = len(methods)
        w = 0.6
        barwidth = w / n
        j = methods.index(method)
        shift = barwidth / 2 + j * barwidth - w / 2
        # barplot with std band for each method side by side in one axes
        ax.fill_between(
            np.arange(len(sorted_mean)),
            np.array(sorted_mean) - np.array(sorted_std),
            np.array(sorted_mean) + np.array(sorted_std),
            # color=method_colors[method],
            color=measure_colors[measure],
            alpha=0.2,
        )
        ax.plot(
            np.arange(len(sorted_mean)),
            sorted_mean,
            marker="x",
            linestyle="-",
            # color=method_colors[method],
            color=measure_colors[measure],
            # label=f"{method_names[method]} + std band",
            label=f"{measure} + std band",
        )

        ax.set_ylabel("Validation Measures")
        ax.set_xlabel("Epochs")
        ax.set_title(scenario)
        ax.grid(True, which="both", ls=":")
        ax.legend()

plt.tight_layout()
output_path = (
    "multirun_images/"
    + str(date.today())
    + f"_NN_training_{method_paths[mehtod]['folktables'].split('/')[-1]}.png"
)
plt.savefig(output_path)

plt.show()
