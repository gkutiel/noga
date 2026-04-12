from itertools import product
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor, nn
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

# CALIBRATION
CAL_EPOCHS = 1_000
CAL_LR = 1e-2


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Name = Literal["l1", "pin_5", "pin_10", "pwa"]


def pwa(
        *,
        breakpoints: tuple[float, float] = (-0.5, 0.5),
        costs: tuple[float, float, float, float] = (
            # UNDER
            10/6, 5/6,
            # OVER
            1/6, 2/6)):

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


def pinball(pred: torch.Tensor, y: torch.Tensor, under=5) -> torch.Tensor:
    unit = 1 / (1 + under)
    error = pred - y
    loss = torch.where(error > 0, unit * error, (unit - 1) * error)
    return torch.mean(loss) * (1 + under) / 2


loss_fns: dict[Name, LossFn] = {
    "l1": nn.L1Loss(),
    "pin_5": lambda pred, y: pinball(pred, y, under=5),
    "pin_10": lambda pred, y: pinball(pred, y, under=10),
    'pwa': pwa(),
}


class Model(pl.LightningModule):
    def __init__(self, name: Name):
        super().__init__()

        self.day = nn.Embedding(7, DAY_EMBED)
        self.month = nn.Embedding(12, MONTH_EMBED)

        self.balance = torch.nn.Parameter(torch.tensor([20.0, 20.0, 20.0]))
        self.net = nn.Linear(INPUT_SIZE, 1)

        self.name = name
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
        self.log(f"{step}/{self.name}", loss, on_epoch=True,
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


class Calibration(pl.LightningModule):
    def __init__(self, loss_name: Name):
        super().__init__()

        self.net = nn.Linear(1, 1)
        self.loss_name = loss_name
        self.loss = loss_fns[loss_name]

    def forward(self, X):
        return self.net(X).squeeze(1)

    def training_step(self, batch, batch_idx):
        pred, y = batch
        y_hat = self(pred)
        loss = self.loss(y_hat, y)

        self.log(f"calibrate/{self.loss_name}", loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=CAL_LR)


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

    out = pt.model(name)

    if out.exists():
        print(f"Skipping training: {out} already exists")
        return

    model = Model(name=name)
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        deterministic=True,
        check_val_every_n_epoch=EPOCHS // 2)

    train_dl, val_dl = load_data()
    trainer.fit(model, train_dl, val_dl)

    torch.save(model.state_dict(), out)


def load_model(name: Name):
    model = Model(name=name)
    model.load_state_dict(torch.load(pt.model(name)))
    model.eval()
    return model


def pred_train(*, model_name: Name):
    model = load_model(model_name)

    train_dl, _ = load_data()

    X, h, y = next(iter(train_dl))
    with torch.no_grad():
        pred = model(X, h)

    return pred, y


def pred_test(*, model_name: Name):
    model = load_model(model_name)

    _, val_dl = load_data()

    X, h, y = next(iter(val_dl))
    with torch.no_grad():
        pred = model(X, h)

    return pred, y


def eval(*, model_name: Name, loss_name: Name):
    pred, y = pred_test(model_name=model_name)

    return loss_fns[loss_name](pred, y).item()


def calibrate(*, model_name: Name, loss_name: Name):
    pl.seed_everything(42)

    path = pt.cal(model_name=model_name, loss_name=loss_name)
    if path.exists():
        return

    pred, y = pred_train(model_name=model_name)

    ds = torch.utils.data.TensorDataset(pred.unsqueeze(1), y)
    dl = DataLoader(ds, batch_size=1024)

    cal = Calibration(loss_name=loss_name)
    trainer = pl.Trainer(max_epochs=CAL_EPOCHS, deterministic=True)
    trainer.fit(cal, dl)

    torch.save(cal.state_dict(), path)

    weight = cal.net.weight.item()
    bias = cal.net.bias.item()
    print(
        f"Calibration ({model_name} → {loss_name}): weight={weight:.4f}, bias={bias:.4f}")


def load_calibrated(*, model_name: Name, loss_name: Name):
    path = pt.cal(model_name=model_name, loss_name=loss_name)

    cal = Calibration(loss_name=loss_name)
    cal.load_state_dict(torch.load(path))
    cal.eval()

    return cal


def eval_calibrated(*, model_name: Name, loss_name: Name):
    if model_name == loss_name:
        return eval(model_name=model_name, loss_name=loss_name)

    pred, y = pred_test(model_name=model_name)

    cal = load_calibrated(
        model_name=model_name,
        loss_name=loss_name)

    with torch.no_grad():
        pred_cal = cal(pred.unsqueeze(1))

    return loss_fns[loss_name](pred_cal, y).item()


class pt:
    @staticmethod
    def model(name: Name):
        return Path(f'models/{name}.pt')

    @staticmethod
    def cal(*, model_name: Name, loss_name: Name):
        return Path(f'models/cal_{model_name}_on_{loss_name}.pt')


def report():
    train(name="l1")
    train(name="pin_5")
    train(name="pin_10")

    names: list[Name] = ["l1", "pin_5", "pin_10"]

    for model_name, loss_name in product(names, names):
        if model_name == loss_name:
            continue

        calibrate(model_name=model_name, loss_name=loss_name)

    rows = [
        {"model": m, "loss": ls, "value": eval(model_name=m, loss_name=ls)}
        for m in names
        for ls in names
    ]

    results = pd.DataFrame(rows).pivot(
        index="model",
        columns="loss",
        values="value")

    print(results.to_string(float_format="{:.4f}".format))
    results.to_csv("data/eval.csv")

    # TODO: when model and loss are the same, eval_calibrated should be the same as eval
    cal_rows = [
        {
            "model": m,
            "loss": ls,
            "value": eval_calibrated(
                model_name=m,
                loss_name=ls)}

        for m in names
        for ls in names
    ]

    cal_results = pd.DataFrame(cal_rows).pivot(
        index="model",
        columns="loss",
        values="value")

    print(cal_results.to_string(float_format="{:.4f}".format))
    cal_results.to_csv("data/eval_calibrated.csv")


if __name__ == "__main__":
    report()
