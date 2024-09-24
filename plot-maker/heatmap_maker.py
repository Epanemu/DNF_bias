import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns
from collections import defaultdict
from datetime import date

# method = "brcg"
# method = "onerule"
# method = "ripper"
method = "mdss"
# method = "dnf_mio"

# scenario = "smallest_subclass"
# scenario = "linear_dependence"
scenario = "constant_subclass"

base_dir = "multirun/2024-09-21-" + scenario + "/" + method

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

            if dimension <= 9 and n_samples <= 50000:
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


EPS = 10**(-10)
fndMin = 10**10
fndMax = 0
for i, n in enumerate(n_samples):
    for j, d in enumerate(dimensions):
        heatmap_data[i, j] = data_dict[n].get(d, np.nan)

        if heatmap_data[i, j] > EPS:
            fndMin = min(fndMin, heatmap_data[i, j])
        fndMax = max(fndMax, heatmap_data[i, j])

# Set values < EPS to NaN for masking, but store a mask for them
mask = heatmap_data < EPS
heatmap_data_clipped = np.where(heatmap_data < EPS, EPS, heatmap_data)

# Create a custom color map with black color for masked (NaN) values
cmap = sns.color_palette("viridis", as_cmap=True)
cmap_with_black = ListedColormap(['black'] + list(cmap.colors))

# Plot the heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    heatmap_data_clipped, 
    annot=True, 
    fmt=".2f", 
    xticklabels=dimensions, 
    yticklabels=n_samples, 
    cmap=cmap_with_black, 
    # mask=mask,
    norm=LogNorm(vmin=fndMin, vmax=fndMax)
)

plt.xlabel("Dimension")
plt.ylabel("N Samples")
plt.title("Logarithmic Difference Heatmap with method '" + method + "' and group type '" + scenario + "'")
plt.gca().invert_yaxis()


output_path = "multirun_images/" + str(date.today()) + "_" + method + "_'" + scenario + "'.png"
plt.savefig(output_path)

plt.show()
