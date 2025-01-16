import numpy as np
import pyomo.environ as pyo

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


class SPSF:
    """Implementation of a MIO formulation for evaluating Statistical Parity Subgroup Fairness.
    From https://proceedings.mlr.press/v80/kearns18a/kearns18a.pdf
    """

    def __init__(self) -> None:
        pass

    def _make_int_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool] | np.ndarray[float],
        n_min: int = 0,
        probabilistic: bool = False,
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction using 0-1 loss.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool] | np.ndarray[float]): target labels or probability of label 1 if argument probabilistic is True
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

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        model.ingroup = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.pos = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                m.ingroup[i]
                >= 1 - sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
            ),
        )
        model.neg = pyo.Constraint(
            model.all_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.ingroup[i] <= 1 - (m.use_feat[j] - Xint[i, j] * m.use_feat[j])
            ),
        )

        if n_min:
            model.minsize = pyo.Constraint(
                expr=(sum(model.ingroup[i] for i in model.all_i) >= n_min),
            )

        if probabilistic:
            p_base = np.mean(y)
            subgroup = sum(model.ingroup[i] for i in model.all_i)
            joint = sum(y[i] * model.ingroup[i] for i in model.all_i)
            term1 = (p_base / n) * subgroup
            term2 = (1 / n) * joint
        else:
            model.pos_i = pyo.Set(initialize=np.where(y)[0])
            model.neg_i = pyo.Set(initialize=np.where(~y)[0])
            negw = len(model.pos_i) / n**2
            posw = len(model.neg_i) / n**2
            term1 = negw * sum(model.ingroup[i] for i in model.neg_i)
            term2 = posw * sum(model.ingroup[i] for i in model.pos_i)

        model.o = pyo.Var(domain=pyo.NonNegativeReals)
        model.b = pyo.Var(domain=pyo.Binary)
        model.abs_obj_u1 = pyo.Constraint(expr=model.o <= term1 - term2 + 2 * model.b)
        model.abs_obj_u2 = pyo.Constraint(
            expr=model.o <= term2 - term1 + 2 * (1 - model.b)
        )
        model.abs_obj_l1 = pyo.Constraint(expr=model.o >= term1 - term2)
        model.abs_obj_l2 = pyo.Constraint(expr=model.o >= term2 - term1)
        model.obj = pyo.Objective(
            expr=model.o,
            sense=pyo.maximize,
        )

        return model

    def find_rule(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool] | None = None,
        y_prob: np.ndarray[float] | None = None,
        n_min: int = 0,
        verbose: bool = False,
    ) -> list[int]:
        """Find a single conjunction with highest SPSF violation

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            list[int]: List of indices of the literals in the final conjunction
        """
        assert (y is not None and y.shape == (X.shape[0],)) or (
            y_prob is not None and y_prob.shape == (X.shape[0],)
        )
        assert X.dtype == bool

        if y_prob is not None:
            int_model = self._make_int_model(X, y_prob, n_min=n_min, probabilistic=True)
        else:
            int_model = self._make_int_model(X, y, n_min=n_min, probabilistic=False)
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt.solve(int_model, tee=verbose)

        self.model = int_model

        if verbose:
            print("OBJECTIVE:", int_model.o.value)

        return [i for i in int_model.feat_i if int_model.use_feat[i].value != 0]

    # TODO move this wrapper to utils?
    def find_subgroup(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        y_prob: np.ndarray[float] | None = None,
        verbose: bool = False,
    ) -> np.ndarray[bool]:
        """Find a single conjunction with highest SPSF violation and returns the y_hat vector of the group

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            np.ndarray[int]: List of indices of the literals in the final conjunction
        """
        conjuncts = self.find_rule(X, y, y_prob=y_prob, verbose=verbose)
        y_hat = np.ones_like(y, dtype=bool)
        for conj in conjuncts:
            y_hat &= X[:, conj]
        return y_hat
