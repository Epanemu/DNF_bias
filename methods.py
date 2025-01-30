import numpy as np
import pandas as pd

from binarizer import Bin, Binarizer


def test_RIPPER(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    ripper_params: dict = {},
):
    from aix360.algorithms.rule_induction.ripper import RipperExplainer

    bin_feats = binarizer.get_bin_encodings(include_negations=False)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("Ripper method assumes that negations are NOT included")

    colnames = ["".join(b) for b in binarizer.multi_index_feats()]
    X_pd = pd.DataFrame(X_train, columns=colnames).astype(int)
    X_test_pd = pd.DataFrame(X_test, columns=colnames).astype(int)
    y_pd = pd.Series(y_train, name="target").astype(int)

    if verbose:
        print("RIPPER:")
    ripper = RipperExplainer(**ripper_params)
    ripper.fit(X_pd, y_pd, target_label=1)
    ruleset = ripper.explain()

    # print("\n\nHERE IS THE RULESET")
    # print(ruleset)
    # print("END OF RULESET\n\n")

    def uncover_value(literal):
        var_name = (
            literal.feature.variable_names[0]
            .replace(",", ", ")
            .replace("^", "(")
            .replace("$", ")")
        )
        feat = bin_feats[colnames.index(var_name)]
        if literal.value == 1:
            return feat
        else:
            return feat.negate_self()

    dnf = [
        [uncover_value(literal) for literal in term.predicates]
        for term in ruleset.conjunctions
    ]

    return ripper.predict(X_test_pd) == 1, dnf


def test_BRCG(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    brcg_params: dict = {},
):
    from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG

    bin_feats = binarizer.get_bin_encodings(include_negations=True)
    colnames = binarizer.multi_index_feats(include_negations=True)
    if X_train.shape[1] != len(bin_feats):
        bin_feats = binarizer.get_bin_encodings(
            include_negations=False, include_binary_negations=True
        )
        colnames = binarizer.multi_index_feats(
            include_negations=False, include_binary_negations=True
        )
        if X_train.shape[1] != len(bin_feats):
            raise ValueError("BRCG method assumes that negations are also included")

    X_train_pd = pd.DataFrame(X_train, columns=colnames)
    X_test_pd = pd.DataFrame(X_test, columns=colnames)

    if verbose:
        print("BRCG")
        brcg_params["verbose"] = True
    if "solver" not in brcg_params:
        brcg_params["solver"] = "GUROBI"
    model = BooleanRuleCG(**brcg_params)
    model.fit(X_train_pd, y_train)

    # print("\n\nEXPLANATION")
    # print(model.explain()["rules"])
    # print("END OF EXPLANATION\n\n")

    split_dnf = [term.split(" AND ") for term in model.explain()["rules"]]
    colnames = [" ".join(b) for b in colnames]
    dnf = [
        [bin_feats[colnames.index(literal)] for literal in term] for term in split_dnf
    ]
    return model.predict(X_test_pd) == 1, dnf

