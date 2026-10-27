from typing import Callable, Literal

import torch
from torch import Tensor

# c[:, 0] = p_da, c[:, 1] = p_o, c[:, 2] = p_u (₪/MWh; for training the model
# divides them by its p_scale, see model.Model.set_scale).
# A slot's own prices are never model inputs (the day-ahead price is set only
# after the procurement decision); they enter training only through the
# "cost" loss. Yesterday's day-ahead price is an input (see model.Data).
LossFn = Callable[[Tensor, Tensor, Tensor], Tensor]
Name = Literal[
    "l2",
    "l1",
    "Pinball med",
    "Pinball opt",
    "Pinball p95",
    "cost",
]


def pinball(pred: Tensor, y: Tensor, fac: float = 5) -> Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


# Fixed-quantile baselines whose single quantile is derived from the training
# prices (estimated on each training batch). Prices set only that constant.

def pinball_opt(pred: Tensor, y: Tensor, c: Tensor) -> Tensor:
    '''
    Cost-optimal constant quantile tau* = E[c_u]/(E[c_u]+E[c_o]) (optimal when
    errors are independent of prices).
    '''
    p_da, p_o, p_u = c[:, 0], c[:, 1], c[:, 2]
    fac = float((p_u - p_da).sum() / p_o.sum())
    return pinball(pred, y, fac=fac)


def pinball_q(q: float) -> LossFn:
    '''Constant quantile = the q-quantile of the per-slot optimal tau_t.'''
    def loss(pred: Tensor, y: Tensor, c: Tensor) -> Tensor:
        p_da, p_o, p_u = c[:, 0], c[:, 1], c[:, 2]
        c_u = p_u - p_da
        t = torch.quantile(c_u / (c_u + p_o), q).item()
        return pinball(pred, y, fac=t / (1 - t))
    return loss


def excess(pred: Tensor, y: Tensor, c: Tensor) -> Tensor:
    '''
    Per-slot cost above a perfect forecast:
        p_da*min(y,p) + p_u*(y-p)+ + p_o*(p-y)+  -  p_da*y
      = (p_u - p_da)*(y-p)+ + p_o*(p-y)+
    '''
    p_da, p_o, p_u = c[:, 0], c[:, 1], c[:, 2]
    under = torch.relu(y - pred)
    over = torch.relu(pred - y)
    return (p_u - p_da) * under + p_o * over


def total(pred: Tensor, y: Tensor, c: Tensor) -> Tensor:
    '''Per-slot total procurement cost.'''
    return c[:, 0] * y + excess(pred, y, c)


loss_fns: dict[Name, LossFn] = {
    "l2": lambda pred, y, c: torch.mean((pred - y) ** 2),
    "l1": lambda pred, y, c: torch.mean(torch.abs(pred - y)),
    "Pinball med": pinball_q(0.5),
    "Pinball opt": pinball_opt,
    "Pinball p95": pinball_q(0.95),
    "cost": lambda pred, y, c: torch.mean(excess(pred, y, c)),
}

# Short names for the command line and experiment names.
SHORT: dict[Name, str] = {
    "l2": "mse",
    "l1": "l1",
    "Pinball med": "med",
    "Pinball opt": "opt",
    "Pinball p95": "p95",
    "cost": "cost",
}

# Losses that do not depend on prices (e.g. for plotting).
PRICE_FREE: list[Name] = ["l2", "l1"]

