import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

method = "onerule"

base_dir = "multirun/2024-08-13/" + method

extracted_data = []

for i in range(100):
    folder_path = os.path.join(base_dir, str(i))
    output_file = os.path.join(folder_path, "output.txt")

    if not os.path.isfile(output_file):
        print(f"Skipping folder {i}, no output.txt found.")
        continue

    with open(output_file, "r") as file:
        lines = file.readlines()
        if len(lines) < 2:
            print(f"Skipping folder {i}, insufficient lines in output.txt.")
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
                # Search for "Our objective:" line and extract the number
                objective_value = None
                for line in lines:
                    if "Our objective:" in line:
                        objective_match = re.search(r"Our objective:\s*([0-9.]+)", line)
                        if objective_match:
                            objective_value = float(objective_match.group(1))
                            break

                # if objective_value is not None:
                extracted_data.append((dimension, n_samples, objective_value))



for data in extracted_data:
    print(f"Dimension: {data[0]}, N Samples: {data[1]}, Objective: {data[2]}")



data_dict = defaultdict(dict)
for dimension, n_samples, objective_value in extracted_data:
    data_dict[n_samples][dimension] = objective_value


dimensions = sorted({dim for values in data_dict.values() for dim in values.keys()})
n_samples = sorted(data_dict.keys())

# print(dimensions)
# print(n_samples)

heatmap_data = np.zeros((len(n_samples), len(dimensions)))

for i, n in enumerate(n_samples):
    for j, d in enumerate(dimensions):
        heatmap_data[i, j] = data_dict[n].get(d, np.nan)  # Use NaN for missing values

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=dimensions, yticklabels=n_samples, cmap="viridis", vmin=0, vmax=1)
plt.xlabel("Dimension")
plt.ylabel("N Samples")
plt.title("Objective Value Heatmap with method '" + method + "'")
plt.gca().invert_yaxis()

plt.savefig(method + "_objective_value_heatmap.png")
plt.show()
