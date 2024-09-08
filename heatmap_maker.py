import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

method = "brcg"
method = "onerule"
method = "ripper"
method = "mdss"
method = "dnf_mio"

base_dir = "multirun/2024-08-31/" + method

extracted_data = []

extracted_true_value = []

for i in range(400):
    folder_path = os.path.join(base_dir, str(i))
    output_file = os.path.join(folder_path, "output.txt")

    if not os.path.isfile(output_file):
        # print(f"Skipping folder {i}, no output.txt found.")
        continue

    with open(output_file, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            # print(f"Skipping folder {i}, insufficient lines in output.txt.")
            continue

        # The command is on the second line
        command_line = lines[1].strip()

        # Extract values after -d and -n using regex
        dimension_match = re.search(r"-d (\d+)", command_line)
        n_samples_match = re.search(r"-n (\d+)", command_line)

        if dimension_match and n_samples_match:
            dimension = int(dimension_match.group(1))
            n_samples = int(n_samples_match.group(1))

            if dimension <= 8 and n_samples <= 50000:
                objective_value = None
                true_objective_value = None

                for line in lines:
                    if "Our final objective:" in line:
                        objective_match = re.search(r"Our final objective:\s*([0-9.]+)", line)
                        if objective_match:
                            objective_value = float(objective_match.group(1))
                    if "The true theoretical sup(\\mu - \\nu) = " in line:
                        true_objective_match = re.search(r"The true theoretical sup\(\\mu - \\nu\) =\s*([0-9.]+)", line)
                        if true_objective_match:
                            true_objective_value = float(true_objective_match.group(1))

                if objective_value is not None and true_objective_value is not None:
                    extracted_data.append((dimension, n_samples, objective_value))
                    extracted_true_value.append((dimension, n_samples, true_objective_value))

# for data in extracted_data:
#     print(f"Dimension: {data[0]}, N Samples: {data[1]}, Absolute Difference: {data[2]}")

data_dict = defaultdict(dict)
data_cnt = defaultdict(dict)
data_true = defaultdict(dict)
for dimension, n_samples, objective_value in extracted_data:
    data_dict[n_samples][dimension] = 0
    data_cnt[n_samples][dimension] = 0
for dimension, n_samples, objective_value in extracted_data:
    data_dict[n_samples][dimension] += objective_value
    data_cnt[n_samples][dimension] += 1
for dimension, n_samples, true_objective_value in extracted_true_value:
    data_true[n_samples][dimension] = true_objective_value

for n_samples, storage in data_dict.items():
    for dimension, value in storage.items():
        data_dict[n_samples][dimension] /= data_cnt[n_samples][dimension]
        data_dict[n_samples][dimension] = abs(data_true[n_samples][dimension] - data_dict[n_samples][dimension]) / data_true[n_samples][dimension]


# dimensions = sorted({dim for values in data_dict.values() for dim in values.keys()})
# n_samples = sorted(data_dict.keys())
dimensions = [1, 2, 3, 4, 5, 6, 7, 8]
n_samples = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

# print(dimensions)
# print(n_samples)

heatmap_data = np.zeros((len(n_samples), len(dimensions)))

for i, n in enumerate(n_samples):
    for j, d in enumerate(dimensions):
        heatmap_data[i, j] = data_dict[n].get(d, np.nan)

plt.figure(figsize=(10, 8))

from matplotlib.colors import LogNorm
sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=dimensions, yticklabels=n_samples, cmap="viridis", norm=LogNorm(vmin=10**(-2.5), vmax=10**1.5)) # vmin=0, vmax=1)
plt.xlabel("Dimension")
plt.ylabel("N Samples")
plt.title("Logarithmic Difference Heatmap with method '" + method + "' and group type 'smallest_subclass'")
plt.gca().invert_yaxis()

from datetime import date
plt.savefig("multirun_images/" + str(date.today()) + "_" + method + "_absolute_difference_heatmap.png")

plt.show()
