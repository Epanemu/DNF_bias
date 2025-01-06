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
    command.extend(["-s", cfg.scenario])
    if hasattr(cfg, "seed") and cfg.seed is not None:
        command.extend(["--seed", str(cfg.seed)])

    if hasattr(cfg, "n_samples") and cfg.n_samples is not None:
        command.extend(["-n", str(cfg.n_samples)])
    if hasattr(cfg, "dimension") and cfg.dimension is not None:
        command.extend(["-d", str(cfg.dimension)])
    if hasattr(cfg, "k") and cfg.k is not None:
        command.extend(["-k", str(cfg.k)])
    if hasattr(cfg, "rho") and cfg.rho is not None:
        command.extend(["--rho", str(cfg.rho)])

    if hasattr(cfg, "method") and cfg.method is not None:
        command.extend(["-m", str(cfg.method)])
    if hasattr(cfg, "n_min") and cfg.n_min is not None:
        command.extend(["-nm", str(cfg.n_min)])

    command.extend(["--verbose"])

    print("Running command:", " ".join(command))

    # Run the command and capture the output
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as exc:
        print("Worked too long. Process finished without result.")
        result = None

    # Get the current working directory, which Hydra sets for each run
    run_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Save the output and error logs to a file in the current run directory
    with open(os.path.join(run_dir, "output.txt"), "w") as out_file:
        out_file.write("Command:\n" + " ".join(command) + "\n")
        out_file.write(f"\nGit hash: {githash}\n\n")
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
