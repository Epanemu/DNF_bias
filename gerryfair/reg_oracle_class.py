import numpy as np
import pyomo.environ as pyo

class ConjunctionOracle:
    """custom made oracle, utilizing a conjunction"""

    def __init__(self, X, cost0, cost1):
        """If the X_i is classified as 1 it receives cost1, otherwise cost0"""
        n, d = X.shape
        Xtotal = np.concatenate([X, ~X], axis=1)

        int_model = self._make_int_model(Xtotal, cost0, cost1)
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt.solve(int_model)

        self.pos_lits = [i for i in range(d) if int_model.use_feat[i].value > 0]
        self.neg_lits = [i-d for i in range(d, 2*d) if int_model.use_feat[i].value > 0]

    def predict(self, X):
        mask = np.ones(X.shape[0], dtype=bool)
        for i in self.pos_lits:
            mask &= X[i] == 1
        for i in self.neg_lits:
            mask &= X[i] == 0
        return list(mask.astype(int))

    def _make_int_model(
        self,
        X: np.ndarray[bool],
        cost0: np.ndarray[float],
        cost1: np.ndarray[float],
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

        # print(Xint)
        # print(cost0)
        # print(cost1)

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary)
        model.yhat = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # model.tmp = pyo.Constraint(expr=model.use_feat[1] == 1)
        # model.tmp2 = pyo.Constraint(expr=model.use_feat[2] == 1)

        model.pos = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                m.yhat[i]
                >= 1 - sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
            ),
        )
        model.neg = pyo.Constraint(
            model.all_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.yhat[i] <= 1 - (m.use_feat[j] - Xint[i, j] * m.use_feat[j])
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.yhat[i] * cost1[i] + (1 - model.yhat[i]) * cost0[i] for i in model.all_i),
            sense=pyo.minimize,
        )

        return model

class RegOracle:
    """Class RegOracle, linear threshold classifier."""
    def __init__(self, b0, b1):
        self.b0 = b0
        self.b1 = b1

    def predict(self, X):
        """Predict labels on data set X."""
        reg0 = self.b0
        reg1 = self.b1
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            x_i = x_i.values.reshape(1, -1)
            c_0 = reg0.predict(x_i)
            c_1 = reg1.predict(x_i)
            y_i = int(c_1 < c_0)
            y.append(y_i)
        return y

class RandomLinearThresh:
    """Class random hyperplane classifier."""
    def __init__(self, d):
        self.coefficient = [np.random.uniform(-1, 1) for _ in range(d)]

    def predict(self, X):
        """Predict labels on data set X."""
        beta = self.coefficient
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            c_1 = np.dot(beta, x_i)
            y_i = int(c_1 < 0)
            y.append(y_i)
        return y

class LinearThresh:
    """Class hyperplane classifier."""
    def __init__(self, d):
        self.coefficient = d

    def predict(self, X):
        """Predict labels on data set X."""
        beta = self.coefficient
        n = X.shape[0]
        y = []
        for i in range(n):
            x_i = X.iloc[i, :]
            c_1 = np.dot(beta, x_i)
            y_i = int(c_1 < 0)
            y.append(y_i)
        return y

