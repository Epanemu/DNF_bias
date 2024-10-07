import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns
from collections import defaultdict
from datetime import date

# one of ["brcg", "onerule", "mdss"]
method = "brcg"

# one of ["smallest_subclass", "linear_dependence", "constant_subclass"]
scenario = "constant_subclass"

# custom dimensions and n_samples for plotting
dimensions_custom = None # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_samples_custom = None # [10, 50, 100, 500, 1000, 5000, 10000, 50000]

# main text on plot
plot_title = "Hamming distance measure '" + method + "' and group type '" + scenario + "'"

#------------------------------

if scenario == "smallest_subclass":
    base_dir = "multirun/2024-08-31-smallest_subclass/" + method
elif scenario == "linear_dependence":
    base_dir = "multirun/2024-09-20-linear_dependence/" + method
elif scenario == "constant_subclass":
    base_dir = "multirun/2024-10-06-constant_sublass(k=3)/" + method
else:
    raise BaseException("Unknown scenario!")

# method = "brcg"
# method = "onerule"
# method = "ripper"
# method = "mdss"
# method = "dnf_mio"

# scenario = "smallest_subclass"
# scenario = "linear_dependence"
# scenario = "constant_subclass"

# base_dir = "multirun/2024-08-31-" + scenario + "/" + method

extracted_data = []

for i in range(500):
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

            if dimension <= 10 and n_samples <= 50000:
                hamming_distance = None

                for line in lines:
                    if "Shortest hamming distance: " in line:
                        distance_match = re.search(r"Shortest hamming distance:\s*([0-9.]+)", line)
                        if distance_match:
                            hamming_distance = float(distance_match.group(1))

                if hamming_distance is not None:
                    extracted_data.append((dimension, n_samples, hamming_distance))

# for data in extracted_data:
#     print(f"Dimension: {data[0]}, N Samples: {data[1]}, Absolute Difference: {data[2]}")

data_dict = defaultdict(dict)
data_cnt = defaultdict(dict)
for dimension, n_samples, objective_value in extracted_data:
    data_dict[n_samples][dimension] = 0
    data_cnt[n_samples][dimension] = 0
for dimension, n_samples, objective_value in extracted_data:
    data_dict[n_samples][dimension] += objective_value
    data_cnt[n_samples][dimension] += 1

for n_samples, storage in data_dict.items():
    for dimension, value in storage.items():
        data_dict[n_samples][dimension] /= data_cnt[n_samples][dimension]


# dimensions = sorted({dim for values in data_dict.values() for dim in values.keys()})
# n_samples = sorted(data_dict.keys())

dimensions = []
for dimension, n_samples, objective_value in extracted_data:
    if objective_value is not None and dimension not in dimensions:
        dimensions.append(dimension)
n_samples = [10, 50, 100, 500, 1000, 5000, 10000, 50000]

if dimensions_custom is not None:
    dimensions = dimensions_custom
if n_samples_custom is not None:
    n_samples = n_samples_custom

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

mask = heatmap_data < EPS
heatmap_data_clipped = np.where(heatmap_data < EPS, EPS, heatmap_data)

# cmap = sns.color_palette("viridis", as_cmap=True)
# cmap_with_black = ListedColormap(['black'] + list(cmap.colors))

# Plot the heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    heatmap_data_clipped, 
    annot=True, 
    fmt=".2f", 
    xticklabels=dimensions, 
    yticklabels=n_samples, 
    # cmap=cmap_with_black, 
    # mask=mask,
    vmin=0.1,
    vmax=6
    # norm=LogNorm(vmin=0.1, vmax=10)
)

plt.xlabel("Dimension")
plt.ylabel("N Samples")
plt.title(plot_title)
plt.gca().invert_yaxis()


output_path = "multirun_images/" + str(date.today()) + "_" + "hamming-distance_" + method + "_'" + scenario + "'.png"
plt.savefig(output_path)

plt.show()
