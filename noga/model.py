import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

# TRAIN
LR = 1e-3
EPOCHS = 1_000
B_SIZE = 64

# DATA
Y_SCALE = 100_000

# MODEL
DAY_EMBED = 2
MONTH_EMBED = 2
SEQ_LEN = 1
INPUT_SIZE = 3 + DAY_EMBED + MONTH_EMBED + SEQ_LEN


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.day = nn.Embedding(7, DAY_EMBED)
        self.month = nn.Embedding(12, MONTH_EMBED)

        self.balance = torch.nn.Parameter(torch.tensor([20.0, 20.0, 20.0]))
        self.net = nn.Linear(INPUT_SIZE, 1)

    def forward(self, X, h):
        day = self.day(X[:, 0].long())
        month = self.month(X[:, 1].long())
        temps = X[:, 2:]

        f = (temps - self.balance).abs()
        f = torch.cat([f, h, day, month], dim=1)

        return self.net(f).squeeze(1)

    def training_step(self, batch, batch_idx):
        X, h, y = batch

        assert X.shape == (B_SIZE, 5), f"Unexpected X shape: {X.shape}"
        assert h.shape == (B_SIZE, SEQ_LEN), f"Unexpected h shape: {h.shape}"
        assert y.shape == (B_SIZE,), f"Unexpected y shape: {y.shape}"

        pred = self(X, h)
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
        cols = [
            # X
            "date",
            "temperature_Haifa",
            "temperature_Jerusalem",
            "temperature_Tel_Aviv",
            # y
            "total_demand"]

        df = df[cols]

        assert not df.isnull().any().any(
        ), f"NaN values found:\n{df.isnull().sum()}"

        date = df['date']

        X = pd.DataFrame({
            "day": (date.dt.dayofweek + 1) % 7,
            "month": date.dt.month - 1,
            "temperature_Haifa": df["temperature_Haifa"],
            "temperature_Jerusalem": df["temperature_Jerusalem"],
            "temperature_Tel_Aviv": df["temperature_Tel_Aviv"],
        })

        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(
            df["total_demand"].values,
            dtype=torch.float32) / Y_SCALE

    def __len__(self):
        return len(self.y) - SEQ_LEN

    def __getitem__(self, idx):
        return self.X[idx + SEQ_LEN], self.y[idx:idx+SEQ_LEN], self.y[idx+SEQ_LEN]


if __name__ == "__main__":
    pl.seed_everything(42)

    daily = pd.read_csv(
        "data/daily.csv",
        parse_dates=["date"])

    test_date = "2025-01-01"
    train = daily[daily["date"] < test_date]
    test = daily[daily["date"] >= test_date]

    train_ds = Data(train)
    test_ds = Data(test)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=B_SIZE,
        shuffle=False,
        drop_last=True)

    model = Model()
    trainer = pl.Trainer(max_epochs=EPOCHS, deterministic=True)
    trainer.fit(model, train_loader)

    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
