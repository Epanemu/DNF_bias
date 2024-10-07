import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# methods for plotting
methods = ["brcg", "mdss", "onerule"]

# one of ["smallest_subclass", "linear_dependence", "constant_subclass"]
scenario = "linear_dependence"

# specified dimension for data plotting
interested_dimension = 5

# main text on plot
plot_title = "Plot for methods ['brcg', 'mdss', 'onerule'], scenario='" + scenario + "',  with dimension=" + str(interested_dimension)

#------------------------------

if scenario == "smallest_subclass":
    base_dir_prefix = "multirun/2024-08-31-smallest_subclass/"
elif scenario == "linear_dependence":
    base_dir_prefix = "multirun/2024-09-20-linear_dependence/"
elif scenario == "constant_subclass":
    base_dir_prefix = "multirun/2024-10-06-constant_sublass(k=3)/"
else:
    raise BaseException("Unknown scenario!")

# scenario = "smallest_subclass"
# scenario = "linear_dependence"
# scenario = "constant_subclass"


# base_dir_prefix = "multirun/2024-09-20-" + scenario + "/"

method_colors = {
    "brcg": "red",
    "mdss": "blue",
    "onerule": "green"
}

def extract_data_for_method(method):
    base_dir = base_dir_prefix + method
    extracted_data = []

    for i in range(500):
        folder_path = os.path.join(base_dir, str(i))
        output_file = os.path.join(folder_path, "output.txt")

        if not os.path.isfile(output_file):
            continue

        with open(output_file, "r") as file:
            lines = file.readlines()
            if len(lines) < 2:
                continue

            command_line = lines[1].strip()

            dimension_match = re.search(r"-d (\d+)", command_line)
            n_samples_match = re.search(r"-n (\d+)", command_line)

            if dimension_match and n_samples_match:
                dimension = int(dimension_match.group(1))
                n_samples = int(n_samples_match.group(1))

                if dimension != interested_dimension:
                    continue

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

    data_dict_total_variation = defaultdict(list)
    data_dict_final_objective = defaultdict(list)
    data_dict_true_sup = defaultdict(list)

    for dimension, n_samples, total_variation, final_objective, true_theoretical_sup in extracted_data:
        data_dict_total_variation[n_samples].append(total_variation)
        data_dict_final_objective[n_samples].append(final_objective)
        data_dict_true_sup[n_samples].append(true_theoretical_sup)

    return data_dict_total_variation, data_dict_final_objective, data_dict_true_sup

all_data = {}
for method in methods:
    all_data[method] = extract_data_for_method(method)

plt.figure(figsize=(10, 6))

plotted_total_variation = False
plotted_true_sup = False

for method in methods:
    # if method == "mdss": continue
    data_dict_total_variation, data_dict_final_objective, data_dict_true_sup = all_data[method]
    sorted_n_samples = sorted(data_dict_total_variation.keys())
    sorted_variations = [np.mean(data_dict_total_variation[n]) for n in sorted_n_samples]
    sorted_objectives = [np.mean(data_dict_final_objective[n]) for n in sorted_n_samples]
    std_devs = [np.std(data_dict_final_objective[n]) for n in sorted_n_samples]

    plt.plot(sorted_n_samples, sorted_objectives, marker='x', linestyle='-', color=method_colors[method], label=f'{method} - Avg Final Objective')
    plt.fill_between(sorted_n_samples, np.array(sorted_objectives) - np.array(std_devs), np.array(sorted_objectives) + np.array(std_devs), color=method_colors[method], alpha=0.2)

    if not plotted_total_variation:
        plt.plot(sorted_n_samples, sorted_variations, linestyle='--', color='grey', label='Avg Total Variation')
        plotted_total_variation = True

    if not plotted_true_sup:
        sorted_true_sup = [np.mean(data_dict_true_sup[n]) for n in sorted_n_samples]
        plt.plot(sorted_n_samples, sorted_true_sup, linestyle='--', color='black', label='Avg True Sup')
        plotted_true_sup = True

plt.xscale('log')
plt.xlabel('Number of Samples (n_samples) [Log Scale]')
plt.ylabel('Value')
plt.title(plot_title)
plt.grid(True, which="both", ls=":")
plt.legend()

from datetime import date
output_path = "multirun_images/" + str(date.today()) + "_plot_" + scenario + ".png"
plt.savefig(output_path)

plt.show()
