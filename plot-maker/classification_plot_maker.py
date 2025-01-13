import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "GerryFair",
    "fairDNF",
]

base_dir_prefix = "multirun/2025-01-13/"

method_colors = {"GerryFair": "red", "fairDNF": "blue"}
method_names = {"GerryFair": "GerryFair", "fairDNF": "DNF via MIO with lazy FPSF"}
method_paths = {
    "GerryFair": {"folktables": "11-29-22"},
    "fairDNF": {"folktables": "11-29-47"},
}


def extract_data_for_method(method):
    extracted_data = []
    for run in ["folktables"]:
        base_dir = base_dir_prefix + method_paths[method][run]

        for i in range(25):
            folder_path = os.path.join(base_dir, str(i))
            output_file = os.path.join(folder_path, "output.txt")

            if not os.path.isfile(output_file):
                continue

            with open(output_file, "r", errors="ignore") as file:
                lines = file.readlines()
                if len(lines) < 2:
                    continue

                setup_line = lines[1].strip()

                n_samples_match = re.search(r"'n_samples': (\d+)", setup_line)
                scenario_match = re.search(r"'scenario': '(\S+)'", setup_line)

                if n_samples_match and scenario_match:
                    n_samples = int(n_samples_match.group(1))
                    scenario = scenario_match.group(1)

                    for line in lines:
                        acc = re.search(r"Accuracy:\s*([0-9.]+)", line)
                        if acc:
                            extracted_data.append(
                                ("Accuracy", scenario, float(acc.group(1)))
                            )
                        mio_fpsf = re.search(r"MIO FPSF:\s*([0-9.]+)", line)
                        if mio_fpsf:
                            extracted_data.append(
                                (
                                    "FPSF (conjunction)",
                                    scenario,
                                    float(mio_fpsf.group(1)),
                                )
                            )
                        gerry_fpsf = re.search(r"Gerry FPSF:\s*([0-9.]+)", line)
                        if gerry_fpsf:
                            extracted_data.append(
                                (
                                    "FPSF (linear oracle)",
                                    scenario,
                                    float(gerry_fpsf.group(1)),
                                )
                            )
                        call_time = re.search(
                            r"Time in callbacks proportion:\s*([0-9.]+)", line
                        )
                        if call_time:
                            extracted_data.append(
                                (
                                    "Proportion of time spent in callbacks",
                                    scenario,
                                    float(call_time.group(1)),
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
                    extracted_data.append(("# Samples", scenario, n_samples))

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


all_data = {}
for method in methods:
    print(method)
    all_data[method] = extract_data_for_method(method)

r, c = 3, 3
fig, axs = plt.subplots(r, c, figsize=(12, 12))

names = sorted(list(all_data[methods[0]].values())[0].keys())
measuers = [
    "Accuracy",
    "FPSF (conjunction)",
    "FPSF (linear oracle)",
    "# Callbacks",
    "# Cuts",
    "Proportion of time spent in callbacks",
    "# Samples",
]

for method in methods:
    data_dict = all_data[method]
    for i, measure in enumerate(measuers):
        ax = axs[i // c, i % c]
        if measure not in data_dict:
            continue

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
        ax.set_xticklabels(names, rotation=90)
        ax.grid(True, which="both", ls=":")
        if i == 0:
            ax.legend()

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_classification.png"
plt.savefig(output_path)

plt.show()
