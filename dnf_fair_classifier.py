import numpy as np
import pyomo.environ as pyo
from gurobipy import GRB

# from one_rule import OneRule
# from one_rule_lp import OneRule
from spsf_mio import SPSF
from utils import eval_fpsf, eval_spsf

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


class DNFFairClassifier:
    """Implementation of a MIO formulation for finding an optimal DNF with the lowest 0-1 error constrained to gamma-fairness.
    The formulation is based on an implementation of eq. (10) in https://krvarshney.github.io/pubs/SuWVM_mlsp2016.pdf
    """

    def __init__(self, gamma: float) -> None:
        self._gamma = gamma

    # TODO implement hamming loss
    def _make_int_cnf(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_clauses: int,
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
        model.clause_i = pyo.Set(initialize=np.arange(n_clauses))
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(
            model.feat_i,
            model.clause_i,
            domain=pyo.Binary,
            initialize=feat_init,
        )
        model.error = pyo.Var(model.all_i, domain=pyo.NonNegativeReals)

        model.pos = pyo.Constraint(
            model.pos_i,
            model.clause_i,
            rule=lambda m, i, r: (
                m.error[i] >= 1 - sum(Xint[i, j] * m.use_feat[j, r] for j in m.feat_i)
            ),
        )

        model.clause_res = pyo.Var(
            model.neg_i, model.clause_i, domain=pyo.NonNegativeReals, bounds=(0, 1)
        )
        model.neg_clause = pyo.Constraint(
            model.neg_i,
            model.feat_i,
            model.clause_i,
            rule=lambda m, i, j, r: m.clause_res[i, r] >= Xint[i, j] * m.use_feat[j, r],
        )
        model.neg = pyo.Constraint(
            model.neg_i,
            rule=lambda m, i: (
                m.error[i]
                >= sum(m.clause_res[i, r] for r in m.clause_i) - (n_clauses - 1)
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.error[i] * weights[i] for i in model.all_i),
            sense=pyo.minimize,
        )

        model.fair_cuts = pyo.ConstraintList()

        return model

    def _add_cut(self, ingroup_i: np.ndarray[int], positive_direction: bool):
        self.n_cuts += 1
        p_group = ingroup_i.shape[0] / self.n_samples
        sum_all_y_hat = sum(self.model.error[i] for i in self.model.neg_i)
        sum_group_y_hat = sum(
            0 if i in self.model.pos_i else self.model.error[i] for i in ingroup_i
        )
        if positive_direction:
            return self.model.fair_cuts.add(
                (p_group / len(self.model.neg_i)) * sum_all_y_hat
                - (1 / self.n_samples) * sum_group_y_hat
                <= self._gamma
            )
        else:
            return self.model.fair_cuts.add(
                (1 / self.n_samples) * sum_group_y_hat
                - (p_group / len(self.model.neg_i)) * sum_all_y_hat
                <= self._gamma
            )

    def _find_subgroup(
        self,
        errors: np.ndarray[bool],
    ) -> tuple[np.ndarray[int], float]:
        self.n_callbacks += 1
        y_hat = np.logical_xor(self.true_y, errors)
        spsf_mio = SPSF()
        mask = self.true_y == 0
        group = spsf_mio.find_subgroup(self.X[mask], y_hat[mask], verbose=self.verbose)
        violation, direction = eval_fpsf(y_hat, group, get_direction=True)
        ingroup_i = np.where(group)[0]
        return ingroup_i, violation, direction

    @property
    def n_samples(self):
        return self.X.shape[0]

    def find_dnf(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        n_terms: int,
        time_limit: int = 120,
        warmstart: bool = False,
        verbose: bool = False,
    ) -> list[int]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            list[int]: List of indices of the literals in the final conjunction
        """
        assert y.shape == (X.shape[0],)
        assert X.dtype == bool and y.dtype == bool

        if warmstart:
            print("No warmstart available")

        self.X = X
        self.true_y = y
        self.verbose = verbose

        w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        w[y] = 1 / size1
        w[~y] = 1 / (y.shape[0] - size1)
        # it finds a CNF -> negate inputs and outputs
        int_model = self._make_int_cnf(~X, ~y, weights=w, n_clauses=n_terms)
        # opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt = pyo.SolverFactory("gurobi_persistent")
        opt.options["TimeLimit"] = time_limit
        opt.set_instance(int_model)
        opt.set_gurobi_param("PreCrush", 1)
        opt.set_gurobi_param("LazyConstraints", 1)
        self.model = int_model

        def callback(cb_m, cb_opt, cb_where):
            if cb_where == GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(
                    vars=[self.model.error[i] for i in self.model.all_i]
                )
                errors = np.array(
                    [self.model.error[i].value > 1e-4 for i in self.model.all_i]
                )
                group_i, violation, pos_direction = self._find_subgroup(errors)
                if violation > self._gamma:
                    cb_opt.cbLazy(self._add_cut(group_i, pos_direction))

        opt.set_callback(callback)
        self.n_cuts = 0
        self.n_callbacks = 0
        result = opt.solve(tee=verbose)
        self.mio_result = result

        dnf = [
            [i for i in int_model.feat_i if int_model.use_feat[i, r].value != 0]
            for r in int_model.clause_i
        ]

        return dnf
