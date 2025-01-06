# python experiment.py -m

import os
import subprocess

import hydra
from omegaconf import DictConfig

githash = ""

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
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
        command.extend(["--rho", str(cfg.rho)])
    if cfg.method is not None:
        command.extend(["-m", str(cfg.method)])
    if cfg.method is not None:
        command.extend(["-nm", str(cfg.n_min)])
    if cfg.k is not None:
        command.extend(["-k", str(cfg.k)])

    command.extend(["--verbose"])

    print("Running command:", " ".join(command))

    # Run the command and capture the output
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=310)
    except subprocess.TimeoutExpired as exc:
        print("Worked too long. Process finished without result.")
        result = None

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        out_file.write("Command:\n" + " ".join(command) + "\n")
        out_file.write(f"Git hash: {githash}\n")
        if result is not None:
            out_file.write("Output:\n" + result.stdout + "\n")
            out_file.write("Errors:\n" + result.stderr + "\n")

    print(f"Result saved to {os.path.join(run_dir, 'output.txt')}")


if __name__ == "__main__":
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if result.stdout.strip() == "":
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        githash = res.stdout.strip()
        run_experiment()
    else:
        raise Exception("Git status is not clean. Commit changes first.")
