from itertools import product
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from noga.cost import Name, loss_fns, optims

# TRAIN
MAX_EPOCHS = 10
BATCH_SIZE = 1024

# MODEL
D_EMBD = 1
M_EMBD = 1
INPUT_SIZE = 3 + D_EMBD + M_EMBD
HIDDEN_SIZE = 12
Y_SCALE = 100


# CALIBRATION
CAL_LR = 2e-2

DAY_IN_5_MIN = 288
HISTORY_LEN = 2
FEATURES = [
    'day',
    'temp_Haifa',
    'temp_Jerusalem',
    'temp_TelAviv'
]
N = len(FEATURES) + HISTORY_LEN


def slice_j(idx: int):
    # In 5min units.
    return slice(idx, idx + HISTORY_LEN), idx + DAY_IN_5_MIN + HISTORY_LEN - 1


class Data(Dataset):
    def __init__(self, df: pd.DataFrame):
        X = df[FEATURES]
        y = df["actual"]
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(
            y.values,
            dtype=torch.float32) / Y_SCALE

    def __len__(self):
        return len(self.y) - DAY_IN_5_MIN - HISTORY_LEN

    def __getitem__(self, idx):
        s, j = slice_j(idx)
        h = self.y[s]
        return torch.concat([h, self.X[j]], dim=0), self.y[j]


class Model(pl.LightningModule):
    def __init__(self, name: Name):
        super().__init__()

        self.name: Name = name
        self.loss = loss_fns[name]

        # self.balance = nn.Parameter(torch.tensor([20.0, 20.0, 20.0]))
        # self.h = nn.Embedding(7, 1)
        self.day = nn.Embedding(7, D_EMBD)
        # self.month = nn.Embedding(12, M_EMBD)

        # self.neg = nn.Linear(N, 1, bias=False)
        # self.pos = nn.Linear(N, 1, bias=False)
        self.net = nn.Sequential(
            nn.Linear(N, N),
            nn.LeakyReLU(),
            nn.Linear(N, 1))

    def forward(self, X: Tensor):
        day = X[:, 0].long()
        f = torch.concat([
            self.day(day),
            X[:, 1:].float()
        ], dim=1)
        # month = X[:, 1].long()
        # temps = X[:, 2:]
        # temps = X

        # dev = (temps - self.balance)
        # neg = self.neg(dev.clamp(max=0))
        # pos = self.pos(dev.clamp(min=0))

        return self.net(f).squeeze(1)

    def step(self, batch, batch_idx, step='train'):
        X, y = batch

        pred = self(X)
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
        return optims[self.name](self.parameters())


class Calibration(pl.LightningModule):
    def __init__(self, *, model_name,  loss_name: Name):
        super().__init__()

        self.model_name = model_name
        self.loss_name = loss_name

        self.net = nn.Linear(1, 1)
        self.loss = loss_fns[loss_name]

    def forward(self, X):
        return self.net(X).squeeze(1)

    def _step(self, batch, step: str):
        pred, y = batch
        y_hat = self(pred)
        loss = self.loss(y_hat, y)

        self.log(
            f"{step}/{self.model_name}-{self.loss_name}",
            loss,
            prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "cal")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "cal_val")

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=CAL_LR)


def load_data():
    data = pd.read_csv("data/data.csv")

    train_df = data[data["year"] < 2025]
    test_df = data[data["year"] == 2025]

    train_ds = Data(train_df)
    test_ds = Data(test_df)

    indices = np.random.choice(
        len(test_ds),
        size=int(0.3 * len(test_ds)),
        replace=False)

    val_ds = Subset(test_ds, indices)  # type: ignore

    def dl(ds: Dataset):
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
            drop_last=False)

    return dl(train_ds), dl(val_ds), dl(test_ds)


def train(name: Name):
    pl.seed_everything(42)

    out = pt.model(name)

    if out.exists():
        print(f"Skipping training: {out} already exists")
        return

    model = Model(name=name)
    logger = TensorBoardLogger("logs", name=name)

    early_stopping = EarlyStopping(
        monitor=f"val/{name}",
        patience=20,
        mode="min")

    ckpt = pt.ckpt(name)

    checkpoint = ModelCheckpoint(
        dirpath=ckpt.parent,
        filename=ckpt.stem,
        save_weights_only=True,
        monitor=f"val/{name}",
        mode="min",
        save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        log_every_n_steps=4,
        check_val_every_n_epoch=10,
        callbacks=[early_stopping, checkpoint],
        logger=logger)

    train_dl, val_dl, _ = load_data()
    trainer.fit(model, train_dl, val_dl)

    best_weights = torch.load(
        checkpoint.best_model_path,
        weights_only=True)

    torch.save(best_weights["state_dict"], out)


def load_model(name: Name):
    model = Model(name=name)
    model.load_state_dict(torch.load(pt.model(name)))
    model.eval()
    return model


def pred_train(*, model_name: Name):
    model = load_model(model_name)
    train_dl, _, _ = load_data()

    X, h, y = next(iter(train_dl))
    with torch.no_grad():
        pred = model(X, h)

    return pred, y


def pred_test(*, model_name: Name):
    model = load_model(model_name)

    _, _, test_dl = load_data()

    X, h, y = next(iter(test_dl))
    with torch.no_grad():
        pred = model(X, h)

    return pred, y


def save_preds(name: Name):
    model = load_model(name)

    _, _, test_dl = load_data()

    preds, actuals = [], []
    with torch.no_grad():
        for X, h, y in test_dl:
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

    path = pt.cal(
        model_name=model_name,
        loss_name=loss_name)

    if path.exists():
        return

    train_pred, train_y = pred_train(model_name=model_name)
    train_ds = torch.utils.data.TensorDataset(train_pred.unsqueeze(1), train_y)
    train_dl = DataLoader(train_ds, batch_size=1024)

    val_pred, val_y = pred_test(model_name=model_name)
    val_ds = torch.utils.data.TensorDataset(val_pred.unsqueeze(1), val_y)
    n_val = max(1, int(0.1 * len(val_ds)))
    val_ds, _ = torch.utils.data.random_split(
        val_ds,
        [n_val, len(val_ds) - n_val])

    val_dl = DataLoader(val_ds, batch_size=1024)

    cal = Calibration(
        model_name=model_name,
        loss_name=loss_name)

    logger = TensorBoardLogger("logs", name=f'cal_{model_name}_on_{loss_name}')

    val_metric = f"cal_val/{model_name}-{loss_name}"

    early_stopping = EarlyStopping(
        monitor=val_metric,
        patience=3,
        mode="min")

    ckpt_path = path.with_suffix(".ckpt")

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_path.parent,
        filename=ckpt_path.stem,
        save_weights_only=True,
        monitor=val_metric,
        mode="min",
        save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        check_val_every_n_epoch=10,
        deterministic=True,
        callbacks=[
            early_stopping,
            checkpoint
        ],
        logger=logger)

    trainer.fit(cal, train_dl, val_dl)

    best_weights = torch.load(
        checkpoint.best_model_path,
        weights_only=True)

    torch.save(best_weights["state_dict"], path)

    # torch.save(cal.state_dict(), path)


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
    def ckpt(name: Name):
        return Path(f'models/{name}.ckpt')

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
    results.to_csv("res/eval.csv", float_format="%.3f")

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

    print()
    print('-' * 40)
    print('Calibrated results:')
    print(cal_results.to_string(float_format="{:.4f}".format))
    cal_results.to_csv("res/eval_calibrated.csv", float_format="%.3f")


if __name__ == "__main__":
    # train_dl, val_dl, test_dl = load_data()

    # for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
    #     X, h, y = next(iter(dl))
    #     print(f"\n--- {name} ---")
    #     print(f"  X shape: {X.shape}, sample: {X[0]}")
    #     print(f"  h shape: {h.shape}, sample: {h[0]}")
    #     print(f"  y shape: {y.shape}, sample: {y[0]:.4f}")

    train('l1')
