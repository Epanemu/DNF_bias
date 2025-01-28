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

# base_dir_prefix = "multirun/2025-01-27/23-09-30"
# base_dir_prefix = "multirun/2025-01-27/23-46-32"
# base_dir_prefix = "multirun/2025-01-28/00-15-10"
# base_dir_prefix = "multirun/2025-01-28/06-30-55"
# base_dir_prefix = "multirun/2025-01-28/06-45-57"
# base_dir_prefix = "multirun/2025-01-28/07-26-21"
# base_dir_prefix = "multirun/2025-01-28/08-30-25"  # only protected
# base_dir_prefix = "multirun/2025-01-28/09-44-30"  # to send
# base_dir_prefix = "multirun/2025-01-28/11-06-09"  # full data - mobility not enough time
base_dir_prefix = "multirun/2025-01-28/13-59-47"  # full data - 100k sub samples

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
                model = model_match.group(1)
                scenario = scenario_match.group(1)

                for line in lines:
                    dist = re.search(r"Distance reported:\s*([0-9.]+)", line)
                    if dist:
                        extracted_data.append(
                            ("Distance", scenario, model, float(dist.group(1)))
                        )
                    time = re.search(r"Time spent:\s*([0-9.]+)", line)
                    if time:
                        extracted_data.append(
                            ("Time [s]", scenario, model, float(time.group(1)))
                        )
                    nsamples = re.search(
                        r"True number of training samples: (\d+)", line
                    )
                    if nsamples:
                        extracted_data.append(
                            ("# Samples", scenario, model, int(nsamples.group(1)))
                        )
                    prot_dim = re.search(r"Protected dimension: (\d+)", line)
                    if prot_dim:
                        extracted_data.append(
                            (
                                "Dimension - protected",
                                scenario,
                                model,
                                int(prot_dim.group(1)),
                            )
                        )
                    full_dim = re.search(r"Full dimension: (\d+)", line)
                    if full_dim:
                        extracted_data.append(
                            ("Dimension - all", scenario, model, int(full_dim.group(1)))
                        )

    data_dict = {}

    for (
        valname,
        scenario,
        model,
        val,
    ) in extracted_data:
        if model not in data_dict:
            data_dict[model] = {}
        if valname not in data_dict[model]:
            data_dict[model][valname] = defaultdict(list)
        data_dict[model][valname][scenario].append(val)

    return data_dict


all_data = extract_data()

r, c = 2, 2
fig, axs = plt.subplots(r, c, figsize=(12, 12))

names = sorted(list(all_data[methods[0]].values())[0].keys())
measuers = [
    "Distance",
    "# Samples",
    "Dimension - all",
    # "Dimension - protected",
    "Time [s]",
]

for method in methods:
    data_dict = all_data[method]
    for i, measure in enumerate(measuers):
        ax = axs[i // c, i % c]
        if measure not in data_dict:
            continue

        # print(data_dict[measure][names[0]])
        sorted_mean = [np.mean(data_dict[measure][n]) for n in names]
        sorted_std = [np.std(data_dict[measure][n]) for n in names]

        n = len(methods)
        w = 0.6
        barwidth = w / n
        j = methods.index(method)
        shift = barwidth / 2 + j * barwidth - w / 2
        # barplot with std band for each method side by side in one axes
        ax.bar(
            np.arange(len(names)) + shift,
            sorted_mean,
            barwidth,
            yerr=sorted_std,
            capsize=5,
            color=method_colors[method],
            label=f"{method_names[method]} + std",
        )

        ax.set_ylabel(measure)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.grid(True, which="both", ls=":")
        if i == 0:
            ax.legend(loc="upper right")

plt.tight_layout()
output_path = "multirun_images/" + str(date.today()) + "_distance.png"
plt.savefig(output_path)

plt.show()
