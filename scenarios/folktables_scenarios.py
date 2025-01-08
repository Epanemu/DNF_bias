import numpy as np
from folktables import ACSDataSource

from binarizer import Binarizer
from data_handler import DataHandler

SCENARIOS = [
    "ACSIncome",
    "ACSPublicCoverage",
    "ACSMobility",
    "ACSEmployment",
    "ACSTravelTime",
]


def load_scenario(name, seed, n_max):
    if name == "ACSIncome":
        from folktables import ACSIncome as Dataset
    elif name == "ACSPublicCoverage":
        from folktables import ACSPublicCoverage as Dataset
    elif name == "ACSMobility":
        from folktables import ACSMobility as Dataset
    elif name == "ACSEmployment":
        from folktables import ACSEmployment as Dataset
    elif name == "ACSTravelTime":
        from folktables import ACSTravelTime as Dataset
    else:
        raise ValueError(f'Scenario "{name}" does not exist.')

    # TODO make the configuration parameterized
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    data = data_source.get_data(states=["CA"], download=True)
    input_data, target_data, _ = Dataset.df_to_pandas(data)

    # DROP COLS WITH TOO MANY OPTIONS
    to_drop = []
    for col in input_data.columns:
        vals = input_data[col].unique().shape[0]
        if vals > 5 or vals <= 1:
            to_drop.append(col)
    input_data.drop(columns=to_drop, inplace=True)

    # print(input_data.shape, target_data[target_data.columns[0]].unique())
    # for col in input_data.columns:
    #     print(col, input_data[col].unique())

    np.random.seed(seed)
    n = input_data.shape[0]
    samples = np.random.choice(n, size=min(n_max, n), replace=False)

    input_data = input_data.iloc[samples]
    target_data = target_data[target_data.columns[0]].iloc[samples]
    dhandler = DataHandler.from_data(
        input_data, target_data, categ_map={c: [] for c in input_data.columns}
    )

    binarizer = Binarizer(dhandler, target_positive_vals=[True])

    return binarizer, input_data, target_data
