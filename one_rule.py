import numpy as np
import pyomo.environ as pyo

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


class OneRule:
    """Implementation of a MIO formulation for finding an optimal conjunction with lowest 0-1 error.
    The formulation is inspired by 1Rule method from http://proceedings.mlr.press/v28/malioutov13.pdf
    """

    def __init__(self) -> None:
        pass

    def _make_int_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction using 0-1 loss.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape
        Xint = np.zeros_like(X, dtype=int)
        Xint[X] = 1

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        model.error = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # model.non_empty = pyo.Constraint(
        #     expr=sum(model.use_feat[j] for j in model.feat_i) >= 1
        # )

        model.pos = pyo.Constraint(
            model.pos_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.use_feat[j] - Xint[i, j] * m.use_feat[j] <= m.error[i]
            ),
        )
        model.neg = pyo.Constraint(
            model.neg_i,
            rule=lambda m, i: (
                sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
                + m.error[i]
                >= 1
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.error[i] for i in model.all_i), sense=pyo.minimize
        )

        return model

    def find_rule(
        self, X: np.ndarray[bool], y: np.ndarray[bool], warmstart: bool = False
    ) -> list[int]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true an approximate solution will be created first to warmstart the MIO.
                Defaults to False.

        Returns:
            list[int]: List of indices of the literals in the final conjunction
        """
        assert y.shape == (X.shape[0],)
        assert X.dtype == bool and y.dtype == bool

        if warmstart:
            print("No warmstart available, previous attempts were useless")

        int_model = self._make_int_model(X, y)
        opt = pyo.SolverFactory("gurobi")
        opt.solve(int_model, tee=True)

        # print([int_model.error[i].value for i in int_model.all_i])

        return [i for i in int_model.feat_i if int_model.use_feat[i].value != 0]
