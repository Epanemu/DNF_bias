# python experiment.py -m

import hydra
from omegaconf import DictConfig
import subprocess
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_experiment(cfg: DictConfig):
    command = ["python", "test_script.py"]

    # Add required parameters
    command.extend(["-s", cfg.task])
    if cfg.dimension is not None:
        command.extend(["-d", str(cfg.dimension)])
    if cfg.n_samples is not None:
        command.extend(["-n", str(cfg.n_samples)])
    if cfg.seed is not None:
        command.extend(["--seed", str(cfg.seed)])
    if cfg.rho is not None:
        command.extend([str(cfg.rho)])
    if cfg.method is not None:
        command.extend(["-m", str(cfg.method)])
    if cfg.k is not None:
        command.extend(["-k", str(cfg.k)])
    else:
        command.extend(["-k", str(cfg.dimension)])

    command.extend(["--verbose"])

    print("Running command:", " ".join(command))
    
    # Run the command and capture the output
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=310)
    except subprocess.TimeoutExpired as exc:
        print("Worked too long. Process finished without result.")

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        out_file.write("Command:\n" + " ".join(command) + "\n")
        out_file.write("Output:\n" + result.stdout + "\n")
        out_file.write("Errors:\n" + result.stderr + "\n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")

if __name__ == "__main__":
    run_experiment()
