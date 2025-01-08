import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "spsf",
    "spsf_mio",
]

base_dir_prefix = "multirun/2025-01-06/"

method_colors = {"spsf": "red", "spsf_mio": "blue"}
method_names = {"spsf": "SPSF (linear)", "spsf_mio": "Our measure of SPSF"}


def extract_data_for_method(method):
    base_dir = base_dir_prefix + method
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

            n_samples_match = re.search(r"-n (\d+)", command_line)
            scenario_match = re.search(r"-s (\S+)", command_line)
            n_subgroup_min = re.search(r"-nm (\d+)", command_line)
            if "mio" in method and n_subgroup_min and n_subgroup_min.group(1) != "10":
                continue

            if n_samples_match and scenario_match:
                max_n_samples = int(n_samples_match.group(1))
                scenario = scenario_match.group(1)

                for line in lines:
                    spsf_str = re.search(r"SPSF violation:\s*([0-9.]+)", line)
                    if spsf_str:
                        extracted_data.append(
                            ("SPSF violation", scenario, float(spsf_str.group(1)))
                        )
                    our_str = re.search(r"Our objective:\s*([0-9.]+)", line)
                    if our_str:
                        extracted_data.append(
                            ("Our Objective", scenario, float(our_str.group(1)))
                        )
                    time_str = re.search(r"Seconds needed:\s*([0-9.]+)", line)
                    if time_str:
                        extracted_data.append(
                            ("Time [s]", scenario, float(time_str.group(1)))
                        )
                    true_n_samples_str = re.search(r", (\d+) remain", line)
                    if true_n_samples_str:
                        extracted_data.append(
                            (
                                "% of samples remained",
                                scenario,
                                int(true_n_samples_str.group(1)) / max_n_samples * 100,
                            )
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


all_data = {}
for method in methods:
    print(method)
    all_data[method] = extract_data_for_method(method)

r, c = 2, 2
fig, axs = plt.subplots(r, c, figsize=(12, 8))

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
