import pandas as pd

from aix360.algorithms.rbm import FeatureBinarizer
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
from aix360.algorithms.rule_induction.ripper import RipperExplainer

X = pd.DataFrame(
    [
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
    ],
    columns=["col1", "col2", "col3", "col4", "col5"],
)
y = pd.Series([0, 0, 0, 1, 1, 1])

fb = FeatureBinarizer(negations=True)
X_bin = fb.fit_transform(X)
# print(X_bin)

model = BooleanRuleCG(lambda0=0.33, lambda1=0, solver="CLARABEL")
model.fit(X_bin, y)

print(model.explain())

r = RipperExplainer()
r.fit(X, y, target_label=1)
ruleset = r.explain()
print(str(ruleset))
