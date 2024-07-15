import pandas as pd

from binarizer import Binarizer
from data_handler import DataHandler


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
    # elif name == "stress":
    # elif name == "stress":
    # elif name == "stress":
    else:
        raise ValueError(f'Scenario "{name}" does not exist.')
    # for col in data_cols:
    #     print(f"{col}:\n  " + "\n  ".join([str(v) for v in data[col].unique()]))

    input_data = data[data_cols]
    target_data = data[target_label]
    dhandler = DataHandler.from_data(
        input_data, target_data, categ_map={c: [] for c in data_cols}
    )

    binarizer = Binarizer(dhandler, target_positive_vals=target_vals)

    return binarizer, input_data, target_data
