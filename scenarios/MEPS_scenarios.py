import pandas as pd

from binarizer import Binarizer
from data_handler import DataHandler

SCENARIOS = [
    "stress",
    "insurance_uninsured",
    "insurance_private",
    "home_problems",
    "socializing",
]


def load_scenario(name):
    data_cols = [
        "SEX",
        "RACEV1X",
        "HISPANX",
        "POVCAT21",
        # "AGE21X",
        "BORNUSA",
        # "INTVLANG", # too many
        "HWELLSPK",
        "MARRY21X",
        # "HIDEG", # not persent, for some reason
        # "YRSINUS", # too many
    ]

    data = pd.read_csv("data.csv")

    if name == "stress":
        target_label = "SDSTRESS"
        target_vals = ["NOT AT ALL"]
    elif name == "insurance_uninsured":
        target_label = "INSCOV21"
        target_vals = ["UNINSURED"]
    elif name == "insurance_private":
        target_label = "INSCOV21"
        target_vals = ["ANY PRIVATE"]
    elif name == "home_problems":
        target_label = "SDPROBNONE"
        target_vals = ["YES"]
    elif name == "socializing":
        target_label = "SDGETTGT"
        target_vals = ["1 TIME", "NEVER", "2 TIMES"]
    else:
        raise ValueError(f'Scenario "{name}" does not exist.')

    input_data = data[data_cols]
    target_data = data[target_label]
    dhandler = DataHandler.from_data(
        input_data, target_data, categ_map={c: [] for c in data_cols}
    )

    binarizer = Binarizer(dhandler, target_positive_vals=target_vals)

    return binarizer, input_data, target_data
