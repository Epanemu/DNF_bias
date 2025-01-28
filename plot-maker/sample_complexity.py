import os
import re
from collections import defaultdict
from datetime import date

import matplotlib.pyplot as plt
import numpy as np

methods = [
    "OneRule",
    "W1",
    "W2",
    "TV",
]

base_dir_prefix = "multirun/2025-01-28/12-39-15"

method_colors = {
    "OneRule": "red",
    "TV": "green",
    "W1": "blue",
    "W2": "magenta",
}
method_names = {
    "OneRule": "MSD (ours)",
    "W1": "Wasserstein-1",
    "W2": "Wasserstein-2",
    "TV": "Total Variation",
}


def extract_data():
    extracted_data = []
    base_dir = base_dir_prefix

    for i in range(100):
        folder_path = os.path.join(base_dir, str(i))
        output_file = os.path.join(folder_path, "output.txt")

        if not os.path.isfile(output_file):
            print(f"passing {i} {output_file}")
            continue

        with open(output_file, "r", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) < 2:
                continue

            setup_line = lines[1].strip()

            model_match = re.search(r"'model': '(\S+)'", setup_line)
            scenario_match = re.search(r"'scenario': '(\S+)'", setup_line)

            if model_match and scenario_match:
                method = model_match.group(1)
                scenario = scenario_match.group(1)

                for line in lines:
                    dist = re.search(r"Distances reported: \[(.+)\]", line)
                    if dist:
                        extracted_data.append(
                            (
                                "Distance",
                                scenario,
                                method,
                                [float(v) for v in dist.group(1).split(", ")],
                            )
                        )
                    time = re.search(r"Times spent: \[(.+)\]", line)
                    if time:
                        extracted_data.append(
                            (
                                "Time",
                                scenario,
                                method,
                                [float(v) for v in time.group(1).split(", ")],
                            )
                        )
                    opt = re.search(r"Optimal/Valid flags: \[(.+)\]", line)
                    if opt:
                        extracted_data.append(
                            (
                                "Optimal/Valid",
                                scenario,
                                method,
                                [v.strip() == "True" for v in opt.group(1).split(", ")],
                            )
                        )
                    nsamples = re.search(
                        r"True numbers of training samples: \[(.+)\]", line
                    )
                    if nsamples:
                        extracted_data.append(
                            (
                                "# Samples",
                                scenario,
                                method,
                                [int(v) for v in nsamples.group(1).split()],
                            )
                        )
                    prot_dim = re.search(r"Protected dimension: (\d+)", line)
                    if prot_dim:
                        extracted_data.append(
                            (
                                "Dimension - protected",
                                scenario,
                                method,
                                int(prot_dim.group(1)),
                            )
                        )
                    full_dim = re.search(r"Full dimension: (\d+)", line)
                    if full_dim:
                        extracted_data.append(
                            (
                                "Dimension - all",
                                scenario,
                                method,
                                int(full_dim.group(1)),
                            )
                        )

    data_dict = {}

    for (
        valname,
        scenario,
        method,
        val,
    ) in extracted_data:
        if method not in data_dict:
            data_dict[method] = {}
        if valname not in data_dict[method]:
            data_dict[method][valname] = defaultdict(list)
        data_dict[method][valname][scenario].append(val)

    return data_dict


all_data = extract_data()

r, c = 2, 5
fig, axs = plt.subplots(r, c, figsize=(30, 12))

# scenarios = sorted(list(all_data[methods[0]].values())[0].keys())
scenarios = [
    "ACSIncome",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSEmployment",
    "ACSTravelTime",
]
measuers = [
    "Distance",
    "Time",
    # "# Samples",
    # "Dimension - all",
    # "Dimension - protected",
]
print(all_data["W1"].keys())

for method in methods:
    data_dict = all_data[method]
    for i, measure in enumerate(measuers):
        if measure not in data_dict:
            continue
        for j, scenario in enumerate(scenarios):
            if scenario not in data_dict[measure]:
                continue
            ax = axs[i, j]

            vals = np.array(data_dict[measure][scenario])
            validity = np.array(data_dict["Optimal/Valid"][scenario], dtype=bool)
            sorted_mean = [
                np.mean(vals[:, k][validity[:, k]])
                for k in range(vals.shape[1])
                if validity[:, k].any()
            ]
            sorted_std = [
                np.std(vals[:, k][validity[:, k]])
                for k in range(vals.shape[1])
                if validity[:, k].any()
            ]
            x = np.array(data_dict["# Samples"][scenario][0])[validity.any(axis=0)]

            ax.fill_between(
                x,
                np.array(sorted_mean) - np.array(sorted_std),
                np.array(sorted_mean) + np.array(sorted_std),
                color=method_colors[method],
                alpha=0.2,
            )
            ax.plot(
                x,
                sorted_mean,
                marker="x",
                linestyle="-",
                color=method_colors[method],
                label=f"{method_names[method]} + std band",
            )

            ax.set_ylabel(measure)
            ax.set_xlabel("Number of samples")
            ax.set_xscale("log")
            ax.set_title(scenario)
            ax.grid(True, which="both", ls=":")
            if i == 0:
                ax.legend(loc="upper right")

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_complexity.png"
plt.savefig(output_path)

plt.show()
