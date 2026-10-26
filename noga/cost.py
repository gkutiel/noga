from typing import Callable, Iterator, Literal, get_args

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam

# c[:, 0] = p_da, c[:, 1] = p_o, c[:, 2] = p_u (scaled prices, see model.P_SCALE).
LossFn = Callable[[Tensor, Tensor, Tensor | None], Tensor]
Name = Literal[
    "l2",
    "l1",
    "Pinball 5:1",
    "Pinball 10:1",
    "Pinball 20:1",
    "l2+price",
    "cost-noprice",
    "cost",
]

# Models that receive the known prices (p_da, p_o) as input features.
USE_PRICE: set[Name] = {"l2+price", "cost"}

# Fixed-ratio pinball models and the quantile each one targets.
PINBALL_TAU: dict[Name, float] = {
    "l1": 0.5,
    "Pinball 5:1": 5 / 6,
    "Pinball 10:1": 10 / 11,
    "Pinball 20:1": 20 / 21,
}


def pinball(pred: Tensor, y: Tensor, fac=5) -> Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


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
    "l2": lambda pred, y, c=None: torch.mean((pred - y) ** 2),
    "l1": lambda pred, y, c=None: torch.mean(torch.abs(pred - y)),
    "Pinball 5:1": lambda pred, y, c=None: pinball(pred, y),
    "Pinball 10:1": lambda pred, y, c=None: pinball(pred, y, fac=10),
    "Pinball 20:1": lambda pred, y, c=None: pinball(pred, y, fac=20),
    "l2+price": lambda pred, y, c=None: torch.mean((pred - y) ** 2),
    "cost-noprice": lambda pred, y, c: torch.mean(excess(pred, y, c)),
    "cost": lambda pred, y, c: torch.mean(excess(pred, y, c)),
}

# Losses that do not depend on prices (e.g. for plotting).
PRICE_FREE: list[Name] = [
    n for n in get_args(Name) if n not in {"cost-noprice", "cost"}]


def opt(params: Iterator[Parameter]):
    return Adam(params, lr=2e-4)


optims: dict[Name, Callable[[Iterator[Parameter]], torch.optim.Optimizer]] = {
    name: opt for name in get_args(Name)
}
