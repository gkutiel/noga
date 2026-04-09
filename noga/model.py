import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset

UNDER = 5
EMBED_SIZE = 5
LR = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 64
BATCH_SIZE = 256
DEPTH = 6
NEGATIVE_SLOPE = 0.1


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x): ...

    def training_step(self, batch, batch_idx): ...

    def predict_step(self, batch, batch_idx): ...

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=LR)


class Data(Dataset):
    def __init__(self):
        daily = pd.read_csv("data/daily.csv")
        # TODO: X is:
        # day of week, month, temperature x 3
        # y is total demand
        # Extract features

    def __len__(self): ...

    def __getitem__(self, idx): ...


if __name__ == "__main__":
    ...
