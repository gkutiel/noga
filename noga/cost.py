from typing import Callable, Literal

import torch
from torch import Tensor


def pinball(pred: torch.Tensor, y: torch.Tensor, fac=10) -> torch.Tensor:
    u = 1 / (1 + fac)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


def gen(pred: Tensor, y: Tensor):
    e = pred - y
    return torch.where(e < 0, -.5 * e + e**2, e * .2).mean()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l1", "pinball", "gen"]

loss_fns: dict[Name, LossFn] = {
    "l1": lambda pred, y: torch.mean(torch.abs(pred - y)),
    "pinball": lambda pred, y: pinball(pred, y),
    'gen': gen,
}

epochs: dict[Name, int] = {
    "l1": 10_000,
    "pinball": 10_000,
    "gen": 10_000,
}

cal_epochs: dict[tuple[Name, Name], int] = {
    ('l1', 'gen'): 10_000,
}

lrs: dict[Name, float] = {
    "l1": 1e-2,
    "pinball": 1e-2,
    "gen": 1e-2,
}
