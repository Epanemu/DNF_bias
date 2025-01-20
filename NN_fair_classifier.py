import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from spsf_mio import SPSF
from utils import eval_fpsf

logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    def __init__(
        self, X: np.ndarray[float], X_prot: np.ndarray[bool], y: np.ndarray[bool]
    ):
        self.X = torch.tensor(X).type(torch.float)
        self.X_prot = X_prot
        self.y = torch.tensor(y).type(torch.float).reshape((-1, 1))

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.X_prot[idx], self.y[idx]


class NNFairClassifier(torch.nn.Module):
    """
    Implementation of a Neural Network classifier, trained with extra fairness loss
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        alpha: float,
        gamma: float,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma
        self.verbose = verbose

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # Assume binary classification
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())

        self._model = nn.Sequential(*layers)
        # self._sigmoid = nn.Sigmoid()
        self._sigmoid = nn.Identity()
        # self._bce_loss = nn.BCEWithLogitsLoss()
        self._bce_loss = nn.MSELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)

    def predict(self, x: np.ndarray[float]) -> np.ndarray[bool]:
        self._model.eval()
        X = torch.asarray(x)
        with torch.no_grad():
            out = np.array(self._model(X) >= 0, dtype=bool)
        return out

    def predict_proba(self, x: np.ndarray[float]) -> np.ndarray[float]:
        self._model.eval()
        X = torch.tensor(x).type(torch.float)
        with torch.no_grad():
            p = self._sigmoid(self._model(X))
        return p.numpy().flatten()

    def train(
        self,
        train: SimpleDataset,
        eval: SimpleDataset,
        batch_size: int,
        fpsf_size: int,
        epochs: int,
    ) -> np.ndarray[float]:
        self.n_FPSF_checks = 0
        self._subgroups = {}

        trainsize = len(train)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval, batch_size=fpsf_size, shuffle=True)

        log_time = (trainsize // batch_size + 1) // 20
        log_time = max(log_time, 1)

        fpsf_X = []
        fpsf_X_prot = []
        fpsf_y = []
        fair_loss = torch.tensor(torch.nan)

        n_data = len(train_loader.dataset)
        n_batches = len(train_loader)

        for epoch_i in range(epochs):
            logger.info(f"EPOCH {epoch_i+1}/{epochs}: ---------------------------")

            cum_class_loss = 0
            cum_fair_loss = 0
            n_corr = 0
            self._model.train()
            for batch_i, (X, X_prot, y) in enumerate(train_loader):
                pred = self._model(X)
                class_loss = self._bce_loss(pred, y)
                cum_class_loss += class_loss.item()
                loss = 0
                loss += class_loss

                n_corr += (
                    ((self._sigmoid(pred) >= 0.5) == y).type(torch.float).sum().item()
                )
                fair_loss = self._fpsf_loss(y, self._sigmoid(pred), X_prot.numpy())
                cum_fair_loss += fair_loss.item()
                # multiply the loss to account for all the batch updates
                loss += self._alpha * fair_loss

                fpsf_X.append(X)
                fpsf_X_prot.append(X_prot)
                fpsf_y.append(y)
                if len(fpsf_X) * batch_size >= fpsf_size:
                    Xin = torch.concat(fpsf_X)
                    if Xin.shape[0] > 2000:
                        np.random.seed(epoch_i * batch_i + batch_i)
                        eval_idx = np.random.choice(Xin.shape[0], 2000, replace=False)
                    else:
                        eval_idx = np.arange(Xin.shape[0])
                    with torch.no_grad():
                        preds = self._sigmoid(self._model(Xin[eval_idx]))
                    fpsf_X_prot = np.concatenate(fpsf_X_prot)[eval_idx]
                    fpsf_y = torch.concat(fpsf_y)[eval_idx]

                    # maybe only in the eval loops, after each epoch?
                    rule, violation, direction = self._find_subgroup(
                        preds, fpsf_y, fpsf_X_prot
                    )
                    if violation > self._gamma:
                        self._add_subgroup(rule, direction)

                    chunk_fair_loss = self._fpsf_loss(fpsf_y, preds, fpsf_X_prot)
                    # # multiply the loss to account for all the batch updates
                    loss += self._alpha * len(fpsf_X) * chunk_fair_loss

                    fpsf_X = []
                    fpsf_X_prot = []
                    fpsf_y = []

                # Backpropagation
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                if batch_i % log_time == 0:
                    data = batch_i * batch_size + X.shape[0]
                    logger.info(
                        f"[{data:>5d}/{trainsize:>5d}] - BCE loss: {class_loss.item():>7f} | FPSF loss: {(fair_loss.item() if fair_loss is not None else torch.nan):>7f}"
                    )

            logger.info("TRAIN:")
            logger.info(f"Accuracy: {(100*(n_corr/n_data)):>0.1f}%")
            logger.info(f"Avg BCE loss: {cum_class_loss/n_batches:>8f}")
            logger.info(f"Avg FPSF loss: {cum_fair_loss/n_batches:>8f}")
            self._eval(eval_loader)

    def _eval(self, eval_loader):
        self._model.eval()
        n_data = len(eval_loader.dataset)
        n_batches = len(eval_loader)
        eval_loss = 0
        fair_loss = 0
        n_corr = 0

        with torch.no_grad():
            for X, X_prot, y in eval_loader:
                pred = self._model(X)
                eval_loss += self._bce_loss(pred, y).item()
                fair_loss += self._fpsf_loss(
                    y, self._sigmoid(pred), X_prot.numpy()
                ).item()
                n_corr += (
                    ((self._sigmoid(pred) >= 0.5) == y).type(torch.float).sum().item()
                )

                # shouldn't be in eval
                # rule, violation, direction = self._find_subgroup(
                #     self._sigmoid(pred), y, X_prot.numpy()
                # )
                # if violation > self._gamma:
                #     self._add_subgroup(rule, direction)
        # rule, violation, _ = self._find_subgroup(pred, y, X_prot.numpy())

        logger.info("VALIDATION:")
        logger.info(f"Accuracy: {(100*(n_corr/n_data)):>0.1f}%")
        logger.info(f"Avg BCE loss: {eval_loss/n_batches:>8f}")
        logger.info(f"Avg FPSF loss: {fair_loss/n_batches:>8f}")
        # logger.info(f"FPSF violation of last batch ([{rule}]): {violation:>8f}")

    def _fpsf_loss(self, y_true, y_pred, X_prot):
        if len(self._subgroups) == 0:
            return torch.tensor(0)

        y_true = y_true.numpy().flatten()
        y_pred = y_pred.flatten()
        violations = []
        n = y_true.shape[0]

        tot_subgroups = 0
        for rule, positive, n_occurences in self._subgroups.values():
            group = np.ones_like(y_true, dtype=bool)
            for conj in rule:
                group &= X_prot[:, conj]
            mask = y_true == 0
            indices = np.where(group & mask)[0]

            subg_size = len(indices)
            p_base = torch.mean(y_pred[mask])
            p_joint = torch.sum(y_pred[indices]) / n
            if positive:
                fpsf = (subg_size / n) * p_base - p_joint
            else:
                fpsf = p_joint - (subg_size / n) * p_base
            violations.append(F.relu(fpsf - self._gamma) ** 2 * n_occurences)
            # violations.append(
            #     F.softplus(self._alpha * (fpsf - self._gamma)) * n_occurences
            # )
            tot_subgroups += n_occurences
        return torch.sum(torch.tensor(violations)) / tot_subgroups

    def _add_subgroup(self, rule: list[int], positive_direction: bool):
        key = (str(rule), positive_direction)
        if key not in self._subgroups:
            logger.info(f"ADDING NEW SUBGROUP on features {rule}")
            self._subgroups[key] = (rule, positive_direction, 1)
        else:
            r, dir, n = self._subgroups[key]
            self._subgroups[key] = r, dir, n + 1

    def _find_subgroup(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        X_prot: np.ndarray[bool],
    ) -> tuple[list[int], float, bool]:
        self.n_FPSF_checks += 1
        y_true = y_true.numpy().flatten()
        y_pred = y_pred.detach().numpy().flatten()
        spsf_mio = SPSF()
        mask = y_true == 0
        rule = spsf_mio.find_rule(
            X_prot[mask], y_prob=y_pred[mask], verbose=self.verbose
        )
        group = np.ones_like(y_true, dtype=bool)
        for conj in rule:
            group &= X_prot[:, conj]
        violation, direction = eval_fpsf(y_true == 1, y_pred, group, get_direction=True)
        logger.debug(
            f"FOUND SUBGROUP {rule} with violation {violation} in {'positive' if direction else 'negative'} direction"
        )
        return rule, violation, direction

    @property
    def n_subgroups(self):
        return len(self._subgroups)

    @property
    def subgroups(self):
        return list(self._subgroups.values())

    def save_model(self, path: str):
        torch.save(self._model.state_dict(), path)
        with open(path + ".conf", "w") as f:
            f.write(
                str(
                    [
                        (layer.in_features, layer.out_features)
                        for layer in self._model
                        if isinstance(layer, nn.Linear)
                    ]
                )
                + "\n"
            )
