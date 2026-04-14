from typing import Callable, Iterator, Literal

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam


def pinball(pred: torch.Tensor, y: torch.Tensor, fac=5) -> torch.Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


def gen(pred: Tensor, y: Tensor):
    e = pred - y
    return torch.where(
        e < 0,
        1.2 * e.abs() ** 1.2,
        .5 * e
    ).mean()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l2", "l1", "pinball", "gen"]

loss_fns: dict[Name, LossFn] = {
    "l2": lambda pred, y: torch.mean((pred - y) ** 2),
    "l1": lambda pred, y: torch.mean(torch.abs(pred - y)),
    "pinball": lambda pred, y: pinball(pred, y),
    'gen': gen,
}


def opt(params: Iterator[Parameter]):
    return Adam(params, lr=2e-2)


optims: dict[Name, Callable[[Iterator[Parameter]], torch.optim.Optimizer]] = {
    "l2": opt,
    "l1": opt,
    "pinball": opt,
    "gen": opt,
}
