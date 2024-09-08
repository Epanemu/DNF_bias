import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

method = "brcg"

base_dir = "multirun/2024-08-31/" + method

extracted_data = []

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

        command_line = lines[1].strip()

        dimension_match = re.search(r"-d (\d+)", command_line)
        n_samples_match = re.search(r"-n (\d+)", command_line)

        if dimension_match and n_samples_match:
            dimension = int(dimension_match.group(1))
            n_samples = int(n_samples_match.group(1))

            if dimension != 3: continue

            total_variation = None
            final_objective = None
            true_theoretical_sup = None
            for line in lines:
                if "Computed total variation:" in line:
                    total_variation_match = re.search(r"Computed total variation:\s*([0-9.]+)", line)
                    if total_variation_match:
                        total_variation = float(total_variation_match.group(1))

                if "Our final objective:" in line:
                    final_objective_match = re.search(r"Our final objective:\s*([0-9.]+)", line)
                    if final_objective_match:
                        final_objective = float(final_objective_match.group(1))

                if "The true theoretical sup(\\mu - \\nu) = " in line:
                    theoretical_sup_match = re.search(r"The true theoretical sup\(\\mu - \\nu\) =\s*([0-9.]+)", line)
                    if theoretical_sup_match:
                        true_theoretical_sup = float(theoretical_sup_match.group(1))



            if total_variation is not None and final_objective is not None and true_theoretical_sup is not None:
                extracted_data.append((dimension, n_samples, total_variation, final_objective, true_theoretical_sup))


data_dict_total_variation = defaultdict(float)
data_dict_final_objective = defaultdict(float)
data_dict_true_sup = defaultdict(float)
cnt_dict = defaultdict(int)

for dimension, n_samples, total_variation, final_objective, true_theoretical_sup in extracted_data:
    data_dict_total_variation[n_samples] += total_variation
    data_dict_final_objective[n_samples] += final_objective
    data_dict_true_sup[n_samples] += true_theoretical_sup
    cnt_dict[n_samples] += 1

for n_samples in data_dict_total_variation:
    data_dict_total_variation[n_samples] /= cnt_dict[n_samples]
    data_dict_final_objective[n_samples] /= cnt_dict[n_samples]
    data_dict_true_sup[n_samples] /= cnt_dict[n_samples]

for n_samples in data_dict_total_variation:
    print(f"{n_samples} -- Avg Total Variation: {data_dict_total_variation[n_samples]}, "
          f"Avg Final Objective: {data_dict_final_objective[n_samples]}, "
          f"Avg True Sup: {data_dict_true_sup[n_samples]}")

sorted_n_samples = sorted(data_dict_total_variation.keys())
sorted_variations = [data_dict_total_variation[n] for n in sorted_n_samples]
sorted_objectives = [data_dict_final_objective[n] for n in sorted_n_samples]
sorted_true_sup = [data_dict_true_sup[n] for n in sorted_n_samples]

plt.figure(figsize=(10, 6))
plt.plot(sorted_n_samples, sorted_variations, marker='o', linestyle='-', color='b', label='Average Total Variation')
plt.plot(sorted_n_samples, sorted_objectives, marker='x', linestyle='--', color='r', label='Average Final Objective')
plt.plot(sorted_n_samples, sorted_true_sup, marker='s', linestyle='-.', color='g', label='Average True Sup')

plt.xscale('log')
plt.xlabel('Number of Samples (n_samples) [Log Scale]')
plt.ylabel('Value')
plt.title(f'Average Total Variation, Final Objective, and True Sup vs. Number of Samples (dimension = 3)')
plt.grid(True, which="both", ls=":")
plt.legend()

from datetime import date
plt.savefig("multirun_images/" + str(date.today()) + "_" + method + "_total_variation_final_objective_true_sup_plot.png")

plt.show()