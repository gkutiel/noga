from typing import Callable, Literal

import torch
from torch import Tensor

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["sym", "pinball", "pwa"]


def pinball(pred: torch.Tensor, y: torch.Tensor, under=6) -> torch.Tensor:
    error = pred - y
    loss = torch.where(error > 0, error, -under * error)
    return torch.mean(loss)


def pwa(
        *,
        bp: float = -0.7,
        costs: tuple[float, float, float] = (
            # UNDER
            12, 4,
            # OVER
            .9)):

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


loss_fns: dict[Name, LossFn] = {
    "sym": lambda pred, y: torch.mean(torch.abs(pred - y) * 3),
    "pinball": lambda pred, y: pinball(pred, y),
    'pwa': pwa(),
}
