from typing import Callable, Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# TRAIN
LR = 2e-3
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


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l1", "pinball"]


def pinball(pred: torch.Tensor, y: torch.Tensor, under=5) -> torch.Tensor:
    unit = 1 / (1 + under)
    error = y - pred
    loss = torch.where(error > 0, unit * error, (unit - 1) * error)
    return torch.mean(loss)


loss_fns: dict[Name, LossFn] = {
    "l1": nn.L1Loss(),
    "pinball": pinball
}


class Model(pl.LightningModule):
    def __init__(self, name: Name):
        super().__init__()

        self.day = nn.Embedding(7, DAY_EMBED)
        self.month = nn.Embedding(12, MONTH_EMBED)

        self.balance = torch.nn.Parameter(torch.tensor([20.0, 20.0, 20.0]))
        self.net = nn.Linear(INPUT_SIZE, 1)

        self.loss = loss_fns[name]

    def forward(self, X, h):
        day = self.day(X[:, 0].long())
        month = self.month(X[:, 1].long())
        temps = X[:, 2:]

        f = (temps - self.balance).abs()
        f = torch.cat([f, h, day, month], dim=1)

        return self.net(f).squeeze(1)

    def step(self, batch, batch_idx, step='train'):
        X, h, y = batch

        pred = self(X, h)
        loss = self.loss(pred, y)
        self.log(f"{step}/mae", loss, on_epoch=True,
                 on_step=False, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step='val')

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


def load_data():
    daily = pd.read_csv(
        "data/daily.csv",
        parse_dates=["date"])

    test_date = "2025-01-01"
    train_df = daily[daily["date"] < test_date]
    test_df = daily[daily["date"] >= test_date]

    train_ds = Data(train_df)
    test_ds = Data(test_df)

    train_dl = DataLoader(
        train_ds,
        batch_size=B_SIZE,
        shuffle=False,
        drop_last=True)

    val_dl = DataLoader(
        test_ds,
        batch_size=1024,
        shuffle=False,
        drop_last=False)

    return train_dl, val_dl


def train(name: Name):

    pl.seed_everything(42)

    model = Model(name=name)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        deterministic=True,
        check_val_every_n_epoch=EPOCHS // 2)

    train_dl, val_dl = load_data()
    trainer.fit(model, train_dl, val_dl)

    torch.save(model.state_dict(), f"data/{name}.pt")


def load_model(name: Name):
    model = Model(name=name)
    model.load_state_dict(torch.load(f"data/{name}.pt"))
    model.eval()
    return model


def eval(*, model_name: Name, loss_name: Name):
    model = load_model(model_name)
    model.eval()

    _, val_dl = load_data()

    X, h, y = next(iter(val_dl))
    with torch.no_grad():
        pred = model(X, h)

    loss = loss_fns[loss_name](pred, y).item()
    print(f"Model: {model_name}, Loss: {loss_name} = {loss:.4f}")


if __name__ == "__main__":
    # train(
    #     loss_fn=nn.L1Loss(),
    #     name="l1")

    eval(model_name="l1", loss_name="l1")
    eval(model_name="l1", loss_name="pinball")
