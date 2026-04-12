from typing import Callable, Literal

import torch
from torch import Tensor

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l1", "pinball", "pwa"]


def pinball(pred: torch.Tensor, y: torch.Tensor, fact=5) -> torch.Tensor:
    u = 1 / (1 + fact)
    error = pred - y
    loss = torch.where(error > 0, u * error, (1-u) * -error)
    return torch.mean(loss * 2)


def pwa(
        *,
        bp: float = -0.7,
        costs: tuple[float, float, float] = (
            # UNDER
            18/6, 8/6,
            # OVER
            .9/6)):

    c1, c2, c3 = costs

    def cost(pred: Tensor, y: Tensor):
        e = pred - y
        return torch.where(
            e <= bp,
            abs(c2 * bp) + c1 * (e - bp).abs(),
            torch.where(
                e <= 0,
                c2 * -e,
                c3 * e))

    return lambda pred, y: cost(pred, y).mean()


SYM_FACTOR = 1
loss_fns: dict[Name, LossFn] = {
    "l1": lambda pred, y: torch.mean(torch.abs(pred - y) * SYM_FACTOR),
    "pinball": lambda pred, y: pinball(pred, y),
    'pwa': pwa(),
}
