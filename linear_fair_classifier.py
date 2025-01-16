import logging
import time

import numpy as np
import pyomo.environ as pyo
from gurobipy import GRB

# from one_rule import OneRule
# from one_rule_lp import OneRule
from spsf_mio import SPSF
from utils import eval_fpsf

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


logger = logging.getLogger(__name__)


class LinearFairClassifier:
    """Implementation of a MIO formulation for finding an optimal DNF with the lowest 0-1 error constrained to gamma-fairness.
    The formulation is based on an implementation of eq. (10) in https://krvarshney.github.io/pubs/SuWVM_mlsp2016.pdf
    """

    def __init__(self, gamma: float) -> None:
        self._gamma = gamma

    # TODO implement hamming loss
    def _make_linear_classif(
        self,
        X: np.ndarray[float],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
        epsilon: float = 1e-6,
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal linear classifier using 0-1 loss.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.coefs = pyo.Var(
            model.feat_i,
            domain=pyo.Reals,
            initialize=feat_init,
            bounds=(-1, 1),
        )
        model.intercept = pyo.Var(
            domain=pyo.Reals, initialize=feat_init, bounds=(-1, 1)
        )
        model.y_hat = pyo.Var(model.all_i, domain=pyo.Binary)

        bigM = d + epsilon
        model.positive = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(m.coefs[j] * X[i, j] for j in model.feat_i) + m.intercept
                >= (m.y_hat[i] - 1) * bigM
            ),
        )
        model.negative = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                sum(m.coefs[j] * X[i, j] for j in model.feat_i) + m.intercept
                <= m.y_hat[i] * bigM - epsilon
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.y_hat[i] for i in model.neg_i)
            + sum(1 - model.y_hat[i] for i in model.pos_i),
            sense=pyo.minimize,
        )

        model.fair_cuts = pyo.ConstraintList()

        return model

    def _add_cut(self, ingroup_i: np.ndarray[int], positive_direction: bool):
        neg_i = self.model.neg_i

        self.n_cuts += 1
        # ingroup contains only those indices where y_true == 0!
        p_group = ingroup_i.shape[0] / self.n_samples
        sum_y_hat = sum(self.model.y_hat[i] for i in neg_i)
        sum_group_y_hat = sum(self.model.y_hat[i] for i in ingroup_i)
        logger.debug(f"ADDING CUT ingroup_i={ingroup_i} p={p_group}")
        if positive_direction:
            return self.model.fair_cuts.add(
                (p_group / len(neg_i)) * sum_y_hat
                - (1 / self.n_samples) * sum_group_y_hat
                <= self._gamma
            )
        else:
            return self.model.fair_cuts.add(
                (1 / self.n_samples) * sum_group_y_hat
                - (p_group / len(neg_i)) * sum_y_hat
                <= self._gamma
            )

    def _find_subgroup(
        self,
        y_hat: np.ndarray[bool],
    ) -> tuple[np.ndarray[int], float]:
        self.n_callbacks += 1
        spsf_mio = SPSF()
        mask = self.true_y == 0
        rule = spsf_mio.find_rule(self.X_prot[mask], y_hat[mask], verbose=self.verbose)
        group = np.ones_like(self.true_y, dtype=bool)
        for conj in rule:
            group &= self.X_prot[:, conj]
        violation, direction = eval_fpsf(self.true_y, y_hat, group, get_direction=True)
        logger.debug(
            f"FOUND SUBGROUP {rule} with violation {violation} in {'positive' if direction else 'negative'} direction"
        )
        ingroup_i = np.where(group & ~self.true_y)[0]
        return ingroup_i, violation, direction

    @property
    def n_samples(self):
        return self.X.shape[0]

    def find_classifier(
        self,
        X: np.ndarray[float],
        X_prot: np.ndarray[bool],
        y: np.ndarray[bool],
        time_limit: int = 120,
        epsilon: float = 1e-6,
        warmstart: bool = False,
        verbose: bool = False,
    ) -> tuple[list[float], float]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            X_prot (np.ndarray[bool]): Input data only for protected attributes (boolean values), shape (n, d2)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            tuple[list[float], float]: Tuple of linear combination coefficients nad a threshold value
        """
        assert y.shape == (X.shape[0],) and y.shape == (X_prot.shape[0],)
        assert y.dtype == bool and X_prot.dtype == bool

        if warmstart:
            print("No warmstart available")

        self.X = X
        self.X_prot = X_prot
        self.true_y = y
        self.verbose = verbose

        w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        w[y] = 1 / size1
        w[~y] = 1 / (y.shape[0] - size1)
        int_model = self._make_linear_classif(X, y, weights=w, epsilon=epsilon)
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.options["TimeLimit"] = time_limit
        opt.set_instance(int_model)
        opt.set_gurobi_param("PreCrush", 1)
        opt.set_gurobi_param("LazyConstraints", 1)
        self.model = int_model

        def callback(cb_m, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL:
                t_start = time.perf_counter()
                cb_opt.cbGetSolution(
                    vars=[self.model.y_hat[i] for i in self.model.all_i]
                )
                y_hat = np.array(
                    [self.model.y_hat[i].value > 1e-4 for i in self.model.all_i]
                )
                group_i, violation, pos_direction = self._find_subgroup(y_hat)
                if violation > self._gamma:
                    cb_opt.cbLazy(self._add_cut(group_i, pos_direction))
                self.__callback_time += time.perf_counter() - t_start

        opt.set_callback(callback)
        self.n_cuts = 0
        self.n_callbacks = 0
        self.__callback_time = 0
        t_start_solve = time.perf_counter()
        result = opt.solve(tee=verbose)
        t_diff = time.perf_counter() - t_start_solve
        print(f"~Time~ in callbacks: {self.__callback_time}")
        self.callback_time_proportion = self.__callback_time / t_diff
        self.mio_result = result

        coefs = np.array(
            [
                int_model.coefs[i].value if int_model.coefs[i].value is not None else 0
                for i in int_model.feat_i
            ],
            dtype=float,
        )
        threshold = -int_model.intercept.value

        return coefs, threshold
