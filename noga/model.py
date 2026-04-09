import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

LR = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 16
BATCH_SIZE = 512


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.balance = torch.nn.Parameter(torch.tensor([20.0, 20.0, 20.0]))
        self.net = nn.Sequential(
            nn.Linear(3, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, X):
        temps = X[:, 2:]
        f = (temps - self.balance) ** 2
        return self.net(f).squeeze(1)

    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = (pred - y).abs().mean()
        self.log("mae", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx): ...

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=LR)


class Data(Dataset):
    def __init__(self, df: pd.DataFrame):

        date = df['date']

        X = pd.DataFrame({
            "day": (date.dt.dayofweek + 1) % 7,
            "month": date.dt.month,
            "temperature_Haifa": df["temperature_Haifa"],
            "temperature_Jerusalem": df["temperature_Jerusalem"],
            "temperature_Tel_Aviv": df["temperature_Tel_Aviv"],
        })

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(
            df["total_demand"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    daily = pd.read_csv(
        "data/daily.csv",
        parse_dates=["date"])

    test_date = "2025-01-01"
    train = daily[daily["date"] < test_date]
    test = daily[daily["date"] >= test_date]

    train_ds = Data(train)
    test_ds = Data(test)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    pl.seed_everything(42)
    model = Model()
    trainer = pl.Trainer(max_epochs=EPOCHS, deterministic=True)
    trainer.fit(model, train_loader)
