from typing import Callable, Iterator, Literal

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal[
    # "l2",
    "l1", "pinball", "pinball2"
    # "PLF"
]


def pinball(pred: torch.Tensor, y: torch.Tensor, fac=5) -> torch.Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


def plf(pred: Tensor, y: Tensor) -> Tensor:
    bp = -1.0
    e = pred - y
    c = 5
    u = 1 / (1 + 5)

    normal = torch.where(e > 0, u * e, (1 - u) * -e)
    severe = (1 - u) * (-bp) + (1 - u + c) * (bp - e)

    return torch.where(e > bp, normal, severe).mean()


loss_fns: dict[Name, LossFn] = {
    # "l2": lambda pred, y: torch.mean((pred - y) ** 2),
    "l1": lambda pred, y: torch.mean(torch.abs(pred - y)),
    "pinball": lambda pred, y: pinball(pred, y),
    "pinball2": lambda pred, y: pinball(pred, y, fac=20),
    # 'PLF': plf,
}


def opt(params: Iterator[Parameter]):
    return Adam(params, lr=2e-4)


optims: dict[Name, Callable[[Iterator[Parameter]], torch.optim.Optimizer]] = {
    # "l2": opt,
    "l1": opt,
    "pinball": opt,
    "pinball2": opt,
    # "PLF": opt,
}
