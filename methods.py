import numpy as np
import pandas as pd

from binarizer import Binarizer


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

    if X_train.shape[1] != len(binarizer.feature_names(include_negations=False)):
        raise ValueError("Ripper method assumes that negations are NOT included")

    X_pd = pd.DataFrame(X_train, columns=binarizer.multi_index_feats()).astype(int)
    X_test_pd = pd.DataFrame(X_test, columns=binarizer.multi_index_feats()).astype(int)
    y_pd = pd.Series(y_train, name="target").astype(int)

    if verbose:
        print("\nRIPPER:")
    ripper = RipperExplainer(**ripper_params)
    ripper.fit(X_pd, y_pd, target_label=1)
    ruleset = ripper.explain()
    if verbose:
        print(ruleset)

    # TODO unify the format of the rules - list of lists of Bins?
    return ripper.predict(X_test_pd) == 1, ruleset


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

    if X_train.shape[1] != len(binarizer.feature_names(include_negations=True)):
        raise ValueError("BRCG method assumes that negations are also included")

    X_train_pd = pd.DataFrame(
        X_train, columns=binarizer.multi_index_feats(include_negations=True)
    )
    X_test_pd = pd.DataFrame(
        X_test, columns=binarizer.multi_index_feats(include_negations=True)
    )
    model = BooleanRuleCG(**brcg_params)
    model.fit(X_train_pd, y_train)

    if verbose:
        print("\nBRCG:")
        print("IF")
        print(" " + "\n OR ".join(model.explain()["rules"]))
        positive, negative = binarizer.target_name()
        print(
            f"THEN\n {positive} ELSE {negative}",
        )
    return model.predict(X_test_pd) == 1, model.explain()


def test_one_rule(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    onerule_params: dict = {},
):
    from one_rule import OneRule

    if X_train.shape[1] != len(binarizer.feature_names(include_negations=True)):
        raise ValueError("One rule method assumes that negations are also included")

    if verbose:
        print("One Rule (using MIO)")
        onerule_params["verbose"] = True
    onerule = OneRule()
    res = onerule.find_rule(X_train, y_train, **onerule_params)

    colnames = binarizer.feature_names(include_negations=True)
    resnames = [colnames[r] for r in res]
    positive, negative = binarizer.target_name()
    if verbose:
        print(
            "IF \n    ",
            "\n AND ".join(resnames),
            f"\nTHEN\n {positive} ELSE {negative}",
        )

    mask = np.ones((X_test.shape[0],), dtype=bool)
    for feat_i in res:
        mask &= X_test[:, feat_i]

    # TODO unify the format of the rules - list of lists of Bins?
    return mask, resnames
