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

# base_dir_prefix = "multirun/2025-01-06/"
base_dir_prefix = "multirun/2025-01-08/"

method_colors = {"spsf": "red", "spsf_mio": "blue"}
method_names = {"spsf": "SPSF (linear)", "spsf_mio": "Our measure of SPSF"}
method_paths = {
    "spsf": {"MEPS": "21-58-42", "folktables": "21-58-44"},
    "spsf_mio": {"MEPS": "21-59-27", "folktables": "11-28-59"},
}


def extract_data_for_method(method):
    extracted_data = []
    for run in ["MEPS", "folktables"]:
        base_dir = base_dir_prefix + method_paths[method][run]

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
                # n_subgroup_min = re.search(r"-nm (\d+)", command_line)
                # if (
                #     "mio" in method
                #     and n_subgroup_min
                #     and n_subgroup_min.group(1) != "10"
                # ):
                #     continue

                if n_samples_match and scenario_match:
                    n_samples = int(n_samples_match.group(1))
                    scenario = scenario_match.group(1)

                    for line in lines:
                        spsf_str = re.search(r"SPSF violation:\s*([0-9.]+)", line)
                        if spsf_str:
                            extracted_data.append(
                                (
                                    "SPSF violation",
                                    scenario,
                                    n_samples,
                                    float(spsf_str.group(1)),
                                )
                            )
                        time_str = re.search(r"Seconds needed:\s*([0-9.]+)", line)
                        if time_str:
                            extracted_data.append(
                                (
                                    "Time [s]",
                                    scenario,
                                    n_samples,
                                    float(time_str.group(1)),
                                )
                            )

    data_dict = {}

    for (
        valname,
        scenario,
        n_samples,
        val,
    ) in extracted_data:
        if valname not in data_dict:
            data_dict[valname] = {}
        if scenario not in data_dict[valname]:
            data_dict[valname][scenario] = defaultdict(list)
        data_dict[valname][scenario][n_samples].append(val)

    return data_dict


all_data = {}
for method in methods:
    all_data[method] = extract_data_for_method(method)

r, c = 10, 2
fig, axs = plt.subplots(r, c, figsize=(12, 30))

for method in methods:
    data_dict = all_data[method]
    for i, measure in enumerate(sorted(data_dict.keys())):
        for j, scenario in enumerate(sorted(data_dict[measure].keys())):
            ax = axs[j, i]

            sorted_n_samples = sorted(data_dict[measure][scenario].keys())
            sorted_mean = [
                np.mean(data_dict[measure][scenario][n]) for n in sorted_n_samples
            ]
            sorted_std = [
                np.std(data_dict[measure][scenario][n]) for n in sorted_n_samples
            ]

            ax.fill_between(
                sorted_n_samples,
                np.array(sorted_mean) - np.array(sorted_std),
                np.array(sorted_mean) + np.array(sorted_std),
                color=method_colors[method],
                alpha=0.2,
            )
            ax.plot(
                sorted_n_samples,
                sorted_mean,
                marker="x",
                linestyle="-",
                color=method_colors[method],
                label=f"{method_names[method]} + std band",
            )

            # if method == methods[-1]:
            #     ax.plot(
            #         [10, 50000],
            #         [0.6 / (2**4)] * 2,
            #         linestyle="--",
            #         color="black",
            #         label="True SPSF violation",
            #     )

            ax.set_title(f"{scenario}")
            ax.set_xscale("log")
            ax.set_xlabel("Number of Samples [Log Scale]")
            ax.set_ylabel(f"{measure}")
            ax.grid(True, which="both", ls=":")
            ax.legend()

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_real_data_line.png"
plt.savefig(output_path)

plt.show()
