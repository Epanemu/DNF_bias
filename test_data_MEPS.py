import sys

import numpy as np
import pandas as pd

from scenarios.MEPS_scenarios import load_scenario

binarizer, input_data, target_data = load_scenario("stress")

X_bin = binarizer.encode(input_data)
X_bin_neg = binarizer.encode(input_data, include_negations=True)
y_bin = binarizer.encode_y(target_data)

n, d = X_bin.shape

if np.mean(y_bin) >= 0.5:
    print("TRIVIAL ACCURACY - always TRUE:", np.mean(y_bin))
else:
    print("TRIVIAL ACCURACY - always FALSE:", 1 - np.mean(y_bin))

if len(sys.argv) > 1 and sys.argv[1] == "AIX":
    from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
    from aix360.algorithms.rule_induction.ripper import RipperExplainer

    # fb = FeatureBinarizer(negations=True)
    pos = np.sum(y_bin)
    neg = y_bin.shape[0] - pos
    if pos > neg:
        to_drop = np.random.choice(np.where(y_bin)[0], size=(pos - neg,), replace=False)
    elif neg > pos:
        to_drop = np.random.choice(
            np.where(~y_bin)[0], size=(neg - pos,), replace=False
        )
    else:
        to_drop = np.array([])
    keep_mask = np.ones_like(y_bin, dtype=bool)
    keep_mask[to_drop] = False
    y_bin = y_bin[keep_mask]
    X_bin = X_bin[keep_mask]
    X_bin_neg = X_bin_neg[keep_mask]

    X_bin_pd = pd.DataFrame(X_bin, columns=binarizer.multi_index_feats())
    X_bin_neg_pd = pd.DataFrame(
        X_bin_neg, columns=binarizer.multi_index_feats(include_negations=True)
    )
    # print(X_bin.columns)
    # model = BooleanRuleCG(lambda0=0, lambda1=0, solver="CLARABEL")
    model = BooleanRuleCG(lambda0=0.0001, lambda1=0, solver="CLARABEL")
    model.fit(X_bin_neg_pd, y_bin)

    print("\nBRCG:")
    print("IF")
    print(" " + "\n OR ".join(model.explain()["rules"]))
    positive, negative = binarizer.target_name()
    print(
        f"THEN\n {positive} ELSE {negative}",
    )
    print("Acc on balanced:", np.mean(model.predict(X_bin_neg_pd) == y_bin))

    print("\nRIPPER:")
    r = RipperExplainer()
    # r = RipperExplainer(d=2, k=100, pruning_threshold=100)
    y_bin_pd = pd.Series(y_bin, name="target")
    r.fit(X_bin_pd.astype(int), y_bin_pd.astype(int), target_label=1)
    ruleset = r.explain()
    # print(type(ruleset))
    print(ruleset)
    print(
        "Acc on balanced:",
        np.mean(r.predict(X_bin_pd.astype(int)) == y_bin.astype(int)),
    )

else:
    from one_rule import OneRule

    onerule = OneRule()
    res = onerule.find_rule(X_bin_neg, y_bin, warmstart=False, verbose=True)

    # print(res, d, X_bin.shape, true_cols)
    # colnames = [c.replace("_", " == ") for c in X.columns]
    # colnames = colnames + [c.replace("_", " != ") for c in X.columns]
    colnames = binarizer.feature_names(include_negations=True)
    resnames = [colnames[r] for r in res]
    positive, negative = binarizer.target_name()
    print(
        "IF \n    ",
        "\n AND ".join(resnames),
        f"\nTHEN\n {positive} ELSE {negative}",
    )

    mask = np.ones((n,), dtype=bool)
    for feat_i in res:
        mask &= X_bin_neg[:, feat_i]
    print("Acc on all:", (np.sum(y_bin[mask]) + np.sum(~y_bin[~mask])) / n)
    # print(data[data.columns[:10]])
