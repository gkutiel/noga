from typing import Callable, Iterator, Literal

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Adam

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l2", "l1", "pinball", "plf"]


def pinball(pred: torch.Tensor, y: torch.Tensor, fac=5) -> torch.Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


def plf(pred: Tensor, y: Tensor) -> Tensor:
    e = pred - y
    bp = -1
    c1 = .8
    c2 = 3
    return torch.where(
        e > 0,
        .5 * e,
        torch.where(
            e >= bp,
            c1 * e.abs(),
            (e - bp).abs() * c2 - bp * c1
        )
    ).mean()


loss_fns: dict[Name, LossFn] = {
    "l2": lambda pred, y: torch.mean((pred - y) ** 2),
    "l1": lambda pred, y: torch.mean(torch.abs(pred - y)),
    "pinball": lambda pred, y: pinball(pred, y),
    'plf': plf,
}


def opt(params: Iterator[Parameter]):
    return Adam(params, lr=3e-2)


optims: dict[Name, Callable[[Iterator[Parameter]], torch.optim.Optimizer]] = {
    "l2": opt,
    "l1": opt,
    "pinball": opt,
    "plf": opt,
}
