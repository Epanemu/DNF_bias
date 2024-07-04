import itertools

import numpy as np

from one_rule import OneRule

np.random.seed(0)
n = 10000


def _to_ohe(dims, vals):
    ohe = np.zeros((sum(dims),))
    offset = 0
    for d, v in zip(dims, vals):
        ohe[offset + v] = 1
        offset += d
    return ohe


simple_way = False
if simple_way:
    # either influences also its supergroups, or the samples are too few to discern
    distros = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25],
        np.array([1, 1, 1]) / 3,
    ]
    Xs = []
    for distro in distros:
        d = len(distro)
        vals = np.random.choice(np.arange(d), (n,), p=distro)
        ohe = np.zeros((n, d))
        ohe[np.arange(n), vals] = 1
        Xs.append(ohe)

    X = np.concatenate(Xs, axis=1)
    y = np.random.choice([0, 1], (n,), p=[0.5, 0.5])
    subgroup = np.concatenate(
        [
            [1, 0, 0, 0, 0],
            [1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0],
        ]
    )
    subgroup_mask = np.all(X == subgroup, axis=1)
    print(sum(subgroup_mask))
    y[subgroup_mask] = 1
    # y[subgroup_mask] = np.random.choice([0, 1], (sum(subgroup_mask),), p=[0.3, 0.7])
else:
    distros = [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    Xs = []
    for distro in distros:
        d = len(distro)
        vals = np.random.choice(np.arange(d), (n,), p=distro)
        ohe = np.zeros((n, d))
        ohe[np.arange(n), vals] = 1
        Xs.append(ohe)
    X = np.concatenate(Xs, axis=1)
    general_distro = np.array([0.5, 0.5])
    y = np.random.choice([0, 1], (n,), p=general_distro)

    dims = [len(d) for d in distros]
    protected_group = [0] * len(distros)
    protected = _to_ohe(dims, protected_group)
    protected_mask = np.all(X == protected, axis=1)
    print(sum(protected_mask))
    prot_distro = np.array([0.2, 0.8])
    y[protected_mask] = np.random.choice([0, 1], (sum(protected_mask),), p=prot_distro)

    # correct the distros
    corrected_mask = protected_mask
    for sub_size in range(len(distros) - 1, -1, -1):
        # for sub_size in range(len(distros)):
        # for sub_size in [0]:
        # for sub_size in [len(distros) - 1]:
        # take all sets of size = sub_size out of the features
        for selected in itertools.combinations(list(range(len(distros))), sub_size):
            group_mask = np.ones_like(y, dtype=bool)
            for i in selected:
                val = protected_group[i]
                offset = sum(d for d in dims[:i])
                group_mask &= X[:, offset + val] == 1
            # coeff = np.sum(protected_mask) / np.sum(group_mask)
            # group_mask &= ~protected_mask
            # modif_distro = (general_distro - prot_distro * coeff) / (1 - coeff)
            fixed_mask = corrected_mask & group_mask
            coeff = np.sum(fixed_mask) / np.sum(group_mask)
            group_mask &= ~corrected_mask
            corr_distro = np.array(
                [np.mean(y[fixed_mask] == 0), np.mean(y[fixed_mask] == 1)]
            )
            modif_distro = (general_distro - corr_distro * coeff) / (1 - coeff)
            print(modif_distro)
            # smoothing the distro
            # smoothing = (sub_size + 3) / (len(distros) + 3)
            smoothing = (sub_size + 1) / (len(distros) + 1)
            modif_distro = modif_distro * smoothing + general_distro * (1 - smoothing)
            print("smoothed", modif_distro)
            corrected_mask |= group_mask

            print(coeff)
            print(
                f"Modifying [{','.join([f'X{s}' for s in selected])}] = 0  with distro {modif_distro}"
            )
            y[group_mask] = np.random.choice([0, 1], (sum(group_mask),), p=modif_distro)
    print()

    # check y for each subgroup
    for sub_size in range(0, len(distros) + 1):
        # take all sets of size = sub_size out of the features
        for selected in itertools.combinations(list(range(len(distros))), sub_size):
            for comb in itertools.product(*[list(range(dims[i])) for i in selected]):
                subgroup_mask = np.ones_like(y, dtype=bool)
                print("Setting [", end="")
                for i, val in zip(selected, comb):
                    print(f"X{i} = {val}", end=", " if i != selected[-1] else "")
                    offset = sum(d for d in dims[:i])
                    subgroup_mask &= X[:, offset + val] == 1
                accuracy = np.empty_like(y)
                accuracy[subgroup_mask] = y[subgroup_mask] == 1
                accuracy[~subgroup_mask] = y[~subgroup_mask] == 0
                print(
                    f"], we get p(y=1|X) = {np.mean(y[subgroup_mask]):.2f} and accuracy = {np.mean(accuracy)}"
                )


# X_bin = np.concatenate([X == 1, X == 0], axis=1)
X_bin = X == 1

onerule = OneRule()
res = onerule.find_rule(X_bin, y == 1, warmstart=False)

print(res)
# print(onerule.model.use_feat.pprint())
# errors = np.array([v.value for v in onerule.model.error.values()])
# print(np.all(errors[y == 0].astype(int) == X_bin[y == 0][:, 2].astype(int)))
# print(np.all(errors[y == 1].astype(int) != X_bin[y == 1][:, 2].astype(int)))
