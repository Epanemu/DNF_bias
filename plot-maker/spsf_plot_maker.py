import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "spsf",
    "spsf_mio",
    "spsf_conj",
    "spsf_custom",
]

base_dir_prefix = "multirun/2024-12-13/"

method_colors = {"spsf": "red", "spsf_mio": "blue", "spsf_conj": "green", "spsf_custom": "magenta"}
method_names = {"spsf": "SPSF (linear)", "spsf_mio":"Our measure of SPSF", "spsf_conj":"SPSF (conjunction)", "spsf_custom":"SPSF (ideal linear)"}


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

            dimension_match = re.search(r"-d (\d+)", command_line)
            n_samples_match = re.search(r"-n (\d+)", command_line)
            n_subgroup_min = re.search(r"-nm (\d+)", command_line)
            if "mio" in method and n_subgroup_min and n_subgroup_min.group(1) != "5":
                continue

            if dimension_match and n_samples_match:
                dimension = int(dimension_match.group(1))
                n_samples = int(n_samples_match.group(1))

                spsf = None
                for line in lines:
                    spsf_str = re.search(r"SPSF violation:\s*([0-9.]+)", line)
                    if spsf_str:
                        spsf = float(spsf_str.group(1))

                if spsf is not None:
                    extracted_data.append(
                        (
                            dimension,
                            n_samples,
                            spsf,
                        )
                    )

    data_dict_spsf = {}

    for (
        dimension,
        n_samples,
        spsf,
    ) in extracted_data:
        if dimension not in data_dict_spsf:
            data_dict_spsf[dimension] = defaultdict(list)
        data_dict_spsf[dimension][n_samples].append(spsf)

    return data_dict_spsf


all_data = {}
for method in methods:
    all_data[method] = extract_data_for_method(method)

r, c = 3, 2
fig, axs = plt.subplots(r, c, figsize=(10, 12))

for method in methods:
    data_dict_spsf = all_data[method]
    for i, dim in enumerate(sorted(data_dict_spsf.keys())):
        ax = axs[i // c, i % c]

        sorted_n_samples = sorted(data_dict_spsf[dim].keys())
        sorted_mean = [np.mean(data_dict_spsf[dim][n]) for n in sorted_n_samples]
        sorted_std = [np.std(data_dict_spsf[dim][n]) for n in sorted_n_samples]

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

        if method == methods[-1]:
            ax.plot(
                [10, 50000],
                [0.6 / (2**4)] * 2,
                linestyle="--",
                color="black",
                label="True SPSF violation",
            )

        ax.set_title(f"Dimension {dim}")
        ax.set_xscale("log")
        ax.set_xlabel("Number of Samples [Log Scale]")
        ax.set_ylabel("SPSF violation")
        ax.grid(True, which="both", ls=":")
        ax.legend()

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_spsf.png"
plt.savefig(output_path)

plt.show()
