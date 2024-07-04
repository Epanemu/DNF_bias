import sys

import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")


# insurance
# label = "INSCOV21"
# target_val = "ANY PRIVATE"
# target_val = "PUBLIC ONLY"
# target_val = "UNINSURED"

# stress
label = "SDSTRESS"
target_val = "NOT AT ALL"

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

X = pd.get_dummies(data[data_cols])

n, d = X.shape
y = data[label]
y_bin = y == target_val
print("base acc:", 1 - np.mean(y_bin))

for col in data_cols:
    print(f"{col}:\n  " + "\n  ".join([str(v) for v in data[col].unique()]))
X_bin = np.concatenate([X == 1, X == 0], axis=1)
# X_bin = X == 1

# print(X_bin[0])

if len(sys.argv) > 1 and sys.argv[1] == "AIX":
    from aix360.algorithms.rbm import FeatureBinarizer
    from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
    from aix360.algorithms.rule_induction.ripper import RipperExplainer

    fb = FeatureBinarizer(negations=True)
    X_bin = fb.fit_transform(data[data_cols])
    # print(X_bin.columns)
    # model = BooleanRuleCG(lambda0=0, lambda1=0, solver="CLARABEL")
    model = BooleanRuleCG(lambda0=0.0001, lambda1=0, solver="CLARABEL")
    model.fit(X_bin, y_bin)

    print("\nBRCG:")
    print("IF")
    print(" " + "\n OR ".join(model.explain()["rules"]))
    print(
        f"THEN\n {label} == {target_val} ELSE {label} in {[v for v in y.unique() if v != target_val]}"
    )

    print("\nRIPPER:")
    r = RipperExplainer(d=2, k=100, pruning_threshold=100)
    r.fit(X.astype(int), y_bin.astype(int), target_label=1)
    ruleset = r.explain()
    print(str(ruleset))

else:
    from one_rule import OneRule

    onerule = OneRule()
    res = onerule.find_rule(X_bin, y_bin, warmstart=False, verbose=True)

    # print(res, d, X_bin.shape, true_cols)
    colnames = [c.replace("_", " == ") for c in X.columns]
    colnames = colnames + [c.replace("_", " != ") for c in X.columns]
    resnames = [colnames[r] for r in res]
    print(
        "IF \n    ",
        "\n AND ".join(resnames),
        f"\nTHEN\n {label} == {target_val} ELSE {label} in {[v for v in y.unique() if v != target_val]}",
    )

    mask = np.ones((n,), dtype=bool)
    for feat_i in res:
        mask &= X_bin[:, feat_i]
    print("Acc:", (np.sum(y_bin[mask]) + np.sum(~y_bin[~mask])) / n)
    # print(data[data.columns[:10]])
