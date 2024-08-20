import argparse

import numpy as np

from methods import test_BRCG, test_one_rule, test_RIPPER
from scenarios.MEPS_scenarios import SCENARIOS as MEPS_SCENARIOS
from scenarios.MEPS_scenarios import load_scenario as load_MEPS_scenario
from scenarios.synthetic_scenarios import SCENARIOS as SYNTH_SCENARIOS
from scenarios.synthetic_scenarios import sample_scenario
from utils import accuracy, balance_datasets, our_metric

SCENARIOS = MEPS_SCENARIOS + SYNTH_SCENARIOS

parser = argparse.ArgumentParser(
    prog="test_script.py",
    description="Benchmark search for bias in data using DNF learning",
    epilog="Research work, subject to change",
)
parser.add_argument(
    "-s",
    "--scenario",
    required=True,
    choices=SCENARIOS,
    help="Scenario to select (required)",
)
parser.add_argument(
    "-d", "--dimension", help="Dimension of sampled data", type=int, default=6
)
parser.add_argument(
    "-k",
    help="Number of features fixed to 0 that will represent the subclass. Only valid for `constant_subclass` scenario.",
    type=int,
    default=3,
)
parser.add_argument(
    "-n", "--n_samples", help="Number of samples to sample", type=int, default=1000
)
parser.add_argument(
    "--seed", help="Seed integer for random processes", type=int, default=0
)
parser.add_argument(
    "--rho", help="Rho parameter for sampling synthetic data", type=float, default=0.8
)
parser.add_argument(
    "--brcg", action="store_true", help="Use the BRCG method to find a DNF"
)
parser.add_argument(
    "--ripper", action="store_true", help="Use the RIPPER method to find a DNF"
)
parser.add_argument(
    "--onerule", action="store_true", help="Find a single rule (conjunction) using MIO"
)
args = parser.parse_args()

if args.scenario in MEPS_SCENARIOS:
    binarizer, input_data, target_data = load_MEPS_scenario(args.scenario)
elif args.scenario in SYNTH_SCENARIOS:
    binarizer, input_data, target_data = sample_scenario(
        args.scenario, args.dimension, args.n_samples, args.seed, rho=args.rho, k=args.k
    )

X_bin = binarizer.encode(input_data)
X_bin_neg = binarizer.encode(input_data, include_negations=True)
y_bin = binarizer.encode_y(target_data)

n_orig, d = X_bin.shape

if np.mean(y_bin) >= 0.5:
    print("TRIVIAL ACCURACY - always TRUE:", np.mean(y_bin))
else:
    print("TRIVIAL ACCURACY - always FALSE:", 1 - np.mean(y_bin))


y_bin, X_bin, X_bin_neg = balance_datasets(
    y_bin, [y_bin, X_bin, X_bin_neg], seed=args.seed
)
n, d = X_bin.shape
print(f"Balancing dropped {n_orig-n} samples, {n} remain. \nDimension is {d}.\n")

if args.ripper:
    y_est, rules = test_RIPPER(
        X_bin,
        y_bin,
        X_bin,
        binarizer,
        verbose=True,
    )
if args.brcg:
    y_est, rules = test_BRCG(
        X_bin_neg,
        y_bin,
        X_bin_neg,
        binarizer,
        verbose=True,
        brcg_params={
            "lambda0": 0.0001,
            "lambda1": 0,
            "solver": "CLARABEL",
        },
    )
if args.onerule:
    y_est, rules = test_one_rule(
        X_bin_neg,
        y_bin,
        X_bin_neg,
        binarizer,
        verbose=True,
    )

print("Accruacy:", accuracy(y_bin, y_est))
print("Our objective:", our_metric(y_bin, y_est))
