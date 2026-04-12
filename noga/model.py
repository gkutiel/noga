from itertools import product
from pathlib import Path
from typing import get_args

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset

from noga.cost import Name, loss_fns

# TRAIN
LR = 1e-3
EPOCHS = 1_000
B_SIZE = 128

# DATA
Y_SCALE = 100_000

# MODEL
SEQ_LEN = 1
BP = 20.0
INPUT_SIZE = 3 + SEQ_LEN

# CALIBRATION
CAL_EPOCHS = 300
CAL_LR = 1e-2


class Model(pl.LightningModule):
    def __init__(self, name: Name):
        super().__init__()

        self.day_bias = Parameter(torch.zeros(7))
        self.month_bias = Parameter(torch.zeros(12))

        self.balance = Parameter(torch.tensor([BP, BP, BP]))
        self.net = nn.Linear(INPUT_SIZE, 1, bias=False)

        self.name = name
        self.loss = loss_fns[name]

    def forward(self, X, h):
        day_bias = self.day_bias[X[:, 0].long()]
        month_bias = self.month_bias[X[:, 1].long()]
        temps = X[:, 2:]

        f = (temps - self.balance).abs()
        f = torch.cat([f, h], dim=1)

        return (self.net(f) + day_bias + month_bias).squeeze(1)

    def step(self, batch, batch_idx, step='train'):
        X, h, y = batch

        pred = self(X, h)
        loss = self.loss(pred, y)

        self.log(f"{step}/{self.name}", loss, prog_bar=True)
        self.log(f"{step}/l1", torch.mean(torch.abs(pred - y)), prog_bar=True)

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
    def __init__(self, *, model_name,  loss_name: Name):
        super().__init__()

        self.model_name = model_name
        self.loss_name = loss_name

        self.net = nn.Linear(1, 1)
        self.loss = loss_fns[loss_name]

    def forward(self, X):
        return self.net(X).squeeze(1)

    def training_step(self, batch, batch_idx):
        pred, y = batch
        y_hat = self(pred)
        loss = self.loss(y_hat, y)

        self.log(
            f"cal/{self.model_name}-{self.loss_name}",
            loss,
            prog_bar=True)

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
    save_params(name)


def load_model(name: Name):
    model = Model(name=name)
    model.load_state_dict(torch.load(pt.model(name)))
    model.eval()
    return model


def save_params(name: Name):
    model = load_model(name)

    day_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    # month_labels = [
    #     "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    #     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    city_labels = ["Haifa", "Jerusalem", "Tel_Aviv"]

    rows: list[dict] = []

    for i, city in enumerate(city_labels):
        rows.append({"param": f"balance_{city}",
                    "value": model.balance[i].item()})

    day_bias = model.day_bias.detach()
    for i, day in enumerate(day_labels):
        rows.append({"param": f"day_bias_{day}", "value": day_bias[i].item()})

    net_w = model.net.weight.detach().squeeze()
    for i, v in enumerate(net_w):
        rows.append({"param": f"net_weight_{i}", "value": v.item()})

    # rows.append({"param": "net_bias", "value": model.net.bias.item()})

    df = pd.DataFrame(rows)
    out = Path(f"params/{name}.csv")
    df.to_csv(out, index=False)
    print(f"Saved params to {out}")


def pred_train(*, model_name: Name):
    model = load_model(model_name)
    train_dl, _ = load_data()

    preds, ys = [], []
    with torch.no_grad():
        for X, h, y in train_dl:
            preds.append(model(X, h))
            ys.append(y)

    return torch.cat(preds), torch.cat(ys)


def pred_test(*, model_name: Name):
    model = load_model(model_name)

    _, val_dl = load_data()

    X, h, y = next(iter(val_dl))
    with torch.no_grad():
        pred = model(X, h)

    return pred, y


def save_preds(name: Name):
    model = load_model(name)

    _, val_dl = load_data()

    preds, actuals = [], []
    with torch.no_grad():
        for X, h, y in val_dl:
            preds.append(model(X, h))
            actuals.append(y)

    pred = torch.cat(preds).numpy()
    actual = torch.cat(actuals).numpy()

    out = Path(f"csv/pred_{name}.csv")
    pd.DataFrame({"pred": pred, "actual": actual}).to_csv(out, index=False)
    print(f"Saved predictions to {out}")


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

    cal = Calibration(
        model_name=model_name,
        loss_name=loss_name)

    trainer = pl.Trainer(max_epochs=CAL_EPOCHS, deterministic=True)
    trainer.fit(cal, dl)

    torch.save(cal.state_dict(), path)

    weight = cal.net.weight.item()
    bias = cal.net.bias.item()
    print(
        f"Calibration ({model_name} → {loss_name}): weight={weight:.4f}, bias={bias:.4f}")


def load_calibrated(*, model_name: Name, loss_name: Name):
    path = pt.cal(model_name=model_name, loss_name=loss_name)

    cal = Calibration(
        model_name=model_name,
        loss_name=loss_name)

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
        return Path(f'cal/cal_{model_name}_on_{loss_name}.pt')


def report():
    names = get_args(Name)

    for name in names:
        train(name)

    for name in names:
        save_preds(name)

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
    # report()
    train("sym")
