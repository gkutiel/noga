import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

EMBED_SIZE = 2
LR = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 16
BATCH_SIZE = 512


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.balance = torch.nn.Parameter(torch.tensor([20, 20, 20]))
        self.net = nn.Sequential(
            nn.Linear(5, HIDDEN_SIZE),
        )

    def forward(self, X):
        _, _, *temps = X
        f = (temps - self.balance) ** 2
        return self.net(f)

    def training_step(self, batch, batch_idx): ...

    def predict_step(self, batch, batch_idx): ...

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=LR)


class Data(Dataset):
    def __init__(self):
        daily = pd.read_csv("data/daily.csv")
        date = pd.to_datetime(daily["date"], format="%d-%m-%Y")

        X = pd.DataFrame({
            "day": (date.dt.dayofweek + 1) % 7,
            "month": date.dt.month,
            "temperature_Haifa": daily["temperature_Haifa"],
            "temperature_Jerusalem": daily["temperature_Jerusalem"],
            "temperature_Tel_Aviv": daily["temperature_Tel_Aviv"],
        })

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(
            daily["total_demand"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    ds = Data()
    print(ds.X.shape, ds.y.shape)
