import numpy as np
import pyomo.environ as pyo

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


class DNF_MIO:
    """Implementation of a MIO formulation for finding an optimal conjunction with lowest 0-1 error.
    The formulation is an implementation of eq. (10) in https://krvarshney.github.io/pubs/SuWVM_mlsp2016.pdf
    """

    def __init__(self) -> None:
        pass

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

        return model

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

        w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        w[y] = 1 / size1
        w[~y] = 1 / (y.shape[0] - size1)
        # it finds a CNF -> negate inputs and outputs
        int_model = self._make_int_cnf(~X, ~y, weights=w, n_clauses=n_terms)
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt.options["TimeLimit"] = time_limit
        opt.solve(int_model, tee=verbose)

        self.model = int_model

        dnf = [
            [i for i in int_model.feat_i if int_model.use_feat[i, r].value != 0]
            for r in int_model.clause_i
        ]

        return dnf
