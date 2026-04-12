from typing import Callable, Literal

import torch
from torch import Tensor

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["sym", "pinball", "pwa"]


def pinball(pred: torch.Tensor, y: torch.Tensor, under=5) -> torch.Tensor:
    error = pred - y
    loss = torch.where(error > 0, error, -under * error)
    return torch.mean(loss)


def pwa(
        *,
        breakpoints: tuple[float, float] = (-0.5, 0.5),
        costs: tuple[float, float, float, float] = (
            # UNDER
            8, 2,
            # OVER
            .5, 2)):

    c1, c2, c3, c4 = costs
    b1, b2 = breakpoints

    def cost(pred: Tensor, y: Tensor):
        e = pred - y
        return torch.where(
            e <= b1,
            abs(c2 * b1) + c1 * (e - b1).abs(),
            torch.where(
                e <= 0,
                c2 * e.abs(),
                torch.where(
                    # x > 0
                    e <= b2,
                    c3 * e,
                    c3 * b2 + c4 * (e - b2),
                )))

    return lambda pred, y: cost(pred, y).mean()


loss_fns: dict[Name, LossFn] = {
    "sym": lambda pred, y: torch.mean(torch.abs(pred - y) * 3),
    "pinball": lambda pred, y: pinball(pred, y),
    'pwa': pwa(),
}
