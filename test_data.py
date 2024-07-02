from os import path

import numpy as np
import pandas as pd

from one_rule import OneRule

data_folder = "data/"
d20_name = "age_sex2020.csv"
d10_name = "age_sex2010.csv"

d20 = pd.read_csv(path.join(data_folder, d20_name), skiprows=1)
d10 = pd.read_csv(path.join(data_folder, d10_name), skiprows=1)

# for col in d20.columns:
#     cname = col[3:] if col[:3] == " !!" else col
#     cname = "".join(cname.split(":"))
#     print(cname, cname in d10.columns)
# print(d10.columns)


# continuing just with the 2010 data


# binary_variant = True
binary_variant = False
one_hot = False

if binary_variant:
    data_cols = ["Total!!Male", "Total!!Female"]
    label = "Label for GEO_ID"
    d10_data = d10[data_cols]

    distros = []
    row_is = [0, 1]
    for i in row_is:
        distros.append(
            d10_data.iloc[i].values / int(d10.iloc[i]["Total"].split("(")[0])
        )

    print(distros)

    n = 10000

    Xs = []
    for distro in distros:
        Xs.append(np.random.choice([0, 1], (n // 2, 1), p=distro))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate([np.ones((n // 2,)), np.zeros((n // 2,))], axis=0) == 1

    X_bin = np.concatenate([X == 1, X == 0], axis=1)

    onerule = OneRule()
    res = onerule.find_rule(X_bin, y == 1, warmstart=False)

    d = len(data_cols)
    resnames = [data_cols[r][7:] for r in res if r < d] + [
        "NOT " + data_cols[r - d][7:] for r in res if r >= d
    ]
    print(
        f"if {' AND '.join(resnames)} then {d10[label].values[row_is[0]]} else {d10[label].values[row_is[1]]}"
    )
elif one_hot:
    # put each sugroup as a coordinate - each sample is a one hot encoding (single 1 among 0s)
    # not the way to do it...
    data_cols = [c for c in d10.columns if c.count("!") == 4]
    label = "Label for GEO_ID"
    d10_data = d10[data_cols]

    distros = []
    row_is = [0, 1]
    for i in row_is:
        distros.append(
            d10_data.iloc[i].values / int(d10.iloc[i]["Total"].split("(")[0])
        )

    # print(distros)

    n = 10000

    Xs = []
    d = len(data_cols)
    for distro in distros:
        vals = np.random.choice(np.arange(d), (n // 2,), p=distro)
        onehot = np.zeros((n // 2, d))
        onehot[np.arange(n // 2), vals] = 1
        Xs.append(onehot)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate([np.ones((n // 2,)), np.zeros((n // 2,))], axis=0) == 1

    X_bin = np.concatenate([X == 1, X == 0], axis=1)

    # print(X_bin[0])

    onerule = OneRule()
    res = onerule.find_rule(X_bin, y == 1, warmstart=False)

    print(res, d, X_bin.shape)
    resnames = [data_cols[r][7:] for r in res if r < d] + [
        "NOT " + data_cols[r - d][7:] for r in res if r >= d
    ]
    print(
        "if",
        "\n AND ".join(resnames),
        f"then {d10[label].values[row_is[0]]} else {d10[label].values[row_is[1]]}",
    )
else:
    data_cols = [c for c in d10.columns if c.count("!") == 4]
    label = "Label for GEO_ID"
    d10_data = d10[data_cols]

    distros = []
    row_is = [0, 1]
    for i in row_is:
        distros.append(
            d10_data.iloc[i].values / int(d10.iloc[i]["Total"].split("(")[0])
        )

    # print(distros)

    n = 10000

    Xs = []
    colmap = {col: col[7:].split("!!") for col in data_cols}
    true_cols = set()
    for _, vals in colmap.items():
        for v in vals:
            true_cols.add(v)
    true_cols = list(true_cols)
    d = len(true_cols)
    for distro in distros:
        vals = np.random.choice(np.arange(len(data_cols)), (n // 2,), p=distro)
        onehot = np.zeros((n // 2, d))
        for j, (_, mapped) in enumerate(colmap.items()):
            for m in mapped:
                i = true_cols.index(m)
                onehot[vals == j, i] = 1
        Xs.append(onehot)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate([np.ones((n // 2,)), np.zeros((n // 2,))], axis=0) == 1

    X_bin = np.concatenate([X == 1, X == 0], axis=1)
    # X_bin = X == 1

    # print(X_bin[0])

    onerule = OneRule()
    # res = onerule.find_rule(X_bin, y == 1, warmstart=False)
    res = onerule.find_rule(X_bin, y == 1, warmstart=True)

    # print(res, d, X_bin.shape, true_cols)
    resnames = [true_cols[r] for r in res if r < d] + [
        "NOT " + true_cols[r - d] for r in res if r >= d
    ]
    print(
        "IF \n    ",
        "\n AND ".join(resnames),
        f"\nTHEN {d10[label].values[row_is[0]]} ELSE {d10[label].values[row_is[1]]}",
    )

    # thresh = 0.95
    # print(np.sum(X_bin[y == 1], axis=0) / sum(y == 1))
    # PAC_selected = np.sum(X_bin[y == 1], axis=0) / sum(y == 1) > thresh
    # print(f"PAC: {np.sum(PAC_selected)} / {2*d}")
    # print(res, np.where(PAC_selected)[0])

    distro_diff = distros[0] - distros[1]
    # print(pd.DataFrame(distro_diff.reshape(1, -1), columns=data_cols))
    # print(np.array(data_cols)[np.argsort(-distro_diff)[:10]])

    idx = np.argsort(-distro_diff)
    data = pd.DataFrame(
        distro_diff[idx].reshape(1, -1), columns=np.array(data_cols)[idx]
    )
    # data.to_csv("tmp.csv")
    feat_distro_diff = []
    for i in range(X_bin.shape[1]):
        if i < d:
            val = sum(
                distro_diff[j] for j, c in enumerate(data_cols) if true_cols[i] in c
            )
        else:
            val = sum(
                # 1 - distros[0][j] - (1 - distros[1][j]) # or equivalently
                distros[1][j] - distros[0][j]
                for j, c in enumerate(data_cols)
                if true_cols[i - d] in c
            )
        feat_distro_diff.append(val)
    # print([v for v in zip(true_cols + ["!" + c for c in true_cols], feat_distro_diff)])

    all_passing = data_cols
    for r in res:
        if r < d:
            all_passing = [p for p in all_passing if true_cols[r] in p]
        else:
            all_passing = [p for p in all_passing if true_cols[r - d] not in p]
    print(
        "Difference in p for the resulting conjunction is ",
        sum(distro_diff[data_cols.index(p)] for p in all_passing),
    )

    mask = np.ones((n,), dtype=bool)
    for feat_i in res:
        mask &= X_bin[:, feat_i]
    print(n // 2 - np.sum(mask[: n // 2]) + np.sum(mask[n // 2 :]))
    # print(data[data.columns[:10]])
