import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from noga.cost import SHORT, Name, loss_fns
from noga.price import P_DA, P_O, P_U, SURPRISE, PriceParams, synth

# TRAIN
# Fixed for every model, no early stopping (the cost on a held-out window is
# too noisy to select a stopping point).
EPOCHS = 600
BATCH_SIZE = 5_000
# AdamW for every model. Decay applies to the Linear weights only (not biases
# or embeddings); each step shrinks them by LR * WEIGHT_DECAY, so small values
# (1e-2 shrinks ~11% over 3000 epochs) barely act.
LR = 5e-4
WEIGHT_DECAY = 0.2
# First test day; training uses everything before it. 2025-07-01 is the second
# rolling-origin fold (test 2025-H2). Set with the NOGA_TRAIN_END variable.
TRAIN_END = os.environ.get("NOGA_TRAIN_END", "2025-01-01")

# MODEL
M_EMBD = 2
D_EMBD = 3
T_EMBED = 4
N_LAYERS = 2
HIDDEN_SIZE = 16
Y_SCALE = 100
P_SCALE = 100
# Mean scaled excess per slot -> M₪ per year.
COST_PER_YEAR = Y_SCALE * P_SCALE * 8760 / 1e6
TEST_EVERY = 25  # epochs between cost/test evaluations

DAY_IN_5_MIN = 288
HISTORY_LEN = 24
FEATURES = [
    'temp_Haifa',
    'temp_Jerusalem',
    'temp_TelAviv',

    'wet_bulb_Haifa',
    'wet_bulb_Jerusalem',
    'wet_bulb_TelAviv',

    'relative_humidity_percent_Haifa',
    'relative_humidity_percent_Jerusalem',
    'relative_humidity_percent_TelAviv',
    # 'wind_speed_m_s_Haifa',
    # 'wind_speed_m_s_Jerusalem',
    # 'wind_speed_m_s_TelAviv',
]

# Models whose training depends on the synthetic price draw.
PRICE_DEPENDENT: set[Name] = {
    "cost", "Pinball med", "Pinball opt", "Pinball p95"}
# Models with auxiliary heads predicting log p_o and log p_u (multi-task
# training only; the slot's own prices are targets, never inputs).
PRICE_HEADS: set[Name] = {"cost"}
PRICE_WEIGHT = 1.0
# Forecast = a * attention + b (a = exp(s) > 0, per slot) instead of
# attention + b. Models get an "-ab" file suffix. Set NOGA_AFFINE_HEAD=1.
AFFINE_HEAD = os.environ.get("NOGA_AFFINE_HEAD") == "1"


# + log of the day-ahead price 24h before t (published before the decision).
N = len(FEATURES) + HISTORY_LEN + 1


class Model(pl.LightningModule):
    def __init__(self, name: Name, test: "Data | None" = None):
        super().__init__()

        self.name: Name = name
        self.loss = loss_fns[name]
        self.test = test  # logged as cost/test during training

        self.day = nn.Embedding(7, D_EMBD)
        self.month = nn.Embedding(12, M_EMBD)
        self.time = nn.Embedding(DAY_IN_5_MIN, T_EMBED)

        self.net = nn.Sequential(
            nn.Linear(N + D_EMBD + M_EMBD + T_EMBED, HIDDEN_SIZE),
            *[
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                ) for _ in range(N_LAYERS)
            ],
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, HISTORY_LEN + 1 + AFFINE_HEAD),
        )
        # Created last so the seeded init matches earlier commits.
        head = cast(nn.Linear, self.net[-1])
        if AFFINE_HEAD:  # start from a = 1
            with torch.no_grad():
                head.weight[HISTORY_LEN].zero_()
                head.bias[HISTORY_LEN].zero_()
        # Shares the hidden representation with the demand forecast.
        self.price = (
            nn.Linear(HIDDEN_SIZE, 2) if name in PRICE_HEADS else None)

    def hidden(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        f = torch.concat([
            self.month(month),
            self.day(day),
            self.time(time),
            X
        ], dim=1)
        return self.net[:-1](f)

    def demand(self, z: Tensor, X: Tensor):
        hist = X[:, :HISTORY_LEN]

        out = self.net[-1](z)

        hist = (hist * torch.softmax(out[:, :HISTORY_LEN], dim=1)).sum(dim=1)

        if AFFINE_HEAD:
            return torch.exp(out[:, HISTORY_LEN]) * hist + out[:, HISTORY_LEN + 1]
        return hist + out[:, HISTORY_LEN]

    def forward(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        return self.demand(self.hidden(month, day, time, X), X)

    def prices(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        '''Predicted (p_o, p_u), scaled by P_SCALE.'''
        assert self.price is not None
        return torch.exp(self.price(self.hidden(month, day, time, X)))

    def training_step(self, batch, batch_idx):
        month, day, time, X, y, c = batch

        z = self.hidden(month, day, time, X)
        pred = self.demand(z, X)
        loss = self.loss(pred, y, c)

        if self.name != "cost":
            self.log_epoch(f"loss/{self.name}", loss, len(y))
        with torch.no_grad():
            cost = loss_fns["cost"](pred, y, c) * COST_PER_YEAR
        self.log_epoch("cost/train", cost, len(y), prog_bar=True)

        if self.price is not None:
            price_loss = self.price_loss(z, c)
            self.log_epoch("price/train", price_loss, len(y))
            loss = loss + PRICE_WEIGHT * price_loss

        return loss

    def price_loss(self, z: Tensor, c: Tensor):
        '''MSE of the predicted log (p_o, p_u).'''
        assert self.price is not None
        return torch.mean((self.price(z) - torch.log(c[:, 1:])) ** 2)

    def log_epoch(self, name: str, value: Tensor, batch_size: int,
                  prog_bar: bool = False):
        self.log(name, value, on_step=False, on_epoch=True,
                 batch_size=batch_size, prog_bar=prog_bar)

    def on_train_epoch_end(self):
        '''Monitoring only: the epoch budget is fixed, never chosen on test.'''
        epoch = self.current_epoch + 1
        if self.test is None or (epoch % TEST_EVERY and epoch != 1):
            return
        t = self.test
        with torch.no_grad():
            z = self.hidden(t.month, t.day, t.time, t.X)
            pred = self.demand(z, t.X)
            cost = loss_fns["cost"](pred, t.y, t.c) * COST_PER_YEAR
            self.log("cost/test", cost, prog_bar=True)
            if self.price is not None:
                self.log("price/test", self.price_loss(z, t.c))

    def configure_optimizers(self):
        decay = [m.weight for m in self.modules() if isinstance(m, nn.Linear)]
        ids = {id(p) for p in decay}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return torch.optim.AdamW([
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": rest, "weight_decay": 0.0},
        ], lr=LR)


def with_prices(
        df: pd.DataFrame,
        *,
        price_seed: int,
        params: PriceParams = PriceParams()) -> pd.DataFrame:
    '''Adds synthetic p_o, p_u (one draw per settlement period).'''
    surprise = (df["actual"] - df["forecast"]) / df["forecast"]
    periods = df[["period", P_DA]].drop_duplicates("period")
    periods[SURPRISE] = surprise.groupby(df["period"]).mean().loc[
        periods["period"]].to_numpy()
    periods = synth(periods, seed=price_seed, params=params)

    df = df.merge(periods[["period", P_O, P_U]], on="period", how="left")
    return df


class Data(Dataset):
    '''
    Item j predicts slot t = j + DAY_IN_5_MIN + HISTORY_LEN - 1 from the
    HISTORY_LEN demand readings ending 24h before t, the features at t and
    the day-ahead price 24h before t (today's price is set only after the
    procurement decision, yesterday's is already public).
    Indexable by a list of indices, so a BatchSampler yields whole batches.
    '''

    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        n = len(df) - DAY_IN_5_MIN - HISTORY_LEN + 1
        t = np.arange(n) + DAY_IN_5_MIN + HISTORY_LEN - 1

        y = torch.tensor(df["actual"].to_numpy(), dtype=torch.float32)
        y = y / Y_SCALE
        hist = y.unfold(0, HISTORY_LEN, 1)[:n]

        X = torch.tensor(df[FEATURES].to_numpy()[t], dtype=torch.float32)
        lag_p = torch.tensor(
            np.log(df[P_DA].to_numpy()[t - DAY_IN_5_MIN] / P_SCALE),
            dtype=torch.float32)
        X = torch.concat([X, lag_p[:, None]], dim=1)

        self.month = torch.tensor(df["month"].to_numpy()[t].astype(int))
        self.day = torch.tensor(df["day"].to_numpy()[t].astype(int))
        self.time = torch.tensor((df["time"].to_numpy()[t] / 5).astype(int))
        self.X = torch.concat([hist, X], dim=1)
        self.y = y[t]
        self.c = torch.tensor(
            df[[P_DA, P_O, P_U]].to_numpy()[t],
            dtype=torch.float32) / P_SCALE

        # Per-item context for evaluation and post-hoc calibration.
        self.meta = df.iloc[t][[
            "date_ord", "time", "forecast", "actual", P_DA, P_O, P_U
        ]].reset_index(drop=True)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            # EMBEDDINGS
            self.month[idx], self.day[idx], self.time[idx],
            # X
            self.X[idx],
            # y
            self.y[idx],
            # prices
            self.c[idx])


def splits(df: pd.DataFrame):
    '''
    train: 2023-04 .. TRAIN_END (exclusive)
    test:  TRAIN_END .. 2025-12
    '''
    end = (pd.Timestamp(TRAIN_END) - pd.Timestamp("2023-01-01")).days
    train = df["date_ord"] < end
    return df[train], df[~train & (df["year"] == 2025)]


def load_data(*, price_seed: int, params: PriceParams = PriceParams()):
    '''Returns the (train, test) datasets.'''
    data = pd.read_csv("data/data.csv")
    data = with_prices(data, price_seed=price_seed, params=params)
    return tuple(Data(df) for df in splits(data))


def dl(ds: Dataset, shuffle=True):
    sampler = RandomSampler(
        ds) if shuffle else SequentialSampler(ds)  # type: ignore
    return DataLoader(
        ds,
        batch_size=None,
        sampler=BatchSampler(sampler, BATCH_SIZE, drop_last=False))


def params_tag(params: PriceParams):
    '''Empty for the default generator, else the non-default fields.'''
    default = PriceParams()
    diff = [
        f"{k}={v:g}" for k, v in vars(params).items()
        if v != getattr(default, k)]
    return "" if not diff else "-" + ",".join(diff)


class pt:
    @staticmethod
    def tag(name: Name, seed: int, price_seed: int,
            params: PriceParams = PriceParams()):
        # Non-default cut-offs get their own files (rolling-origin folds).
        end = "" if TRAIN_END == "2025-01-01" else f"-t{TRAIN_END[:7]}"
        end += "-ab" if AFFINE_HEAD else ""
        if name not in PRICE_DEPENDENT:
            return f"{name}-s{seed}{end}"
        return f"{name}-s{seed}-p{price_seed}{params_tag(params)}{end}"

    @staticmethod
    def short(name: Name, seed: int, price_seed: int,
              params: PriceParams = PriceParams()):
        '''Short name plus the non-default parts of the tag, e.g. cost-s1-p2.'''
        extra = pt.tag(name, seed, price_seed, params)[len(name):]
        extra = extra.replace("-s0", "", 1) if seed == 0 else extra
        extra = extra.replace("-p0", "", 1) if price_seed == 0 else extra
        return SHORT[name] + extra

    @staticmethod
    def model(name: Name, seed: int, price_seed: int,
              params: PriceParams = PriceParams(), *, sha: str):
        '''models/<short name>-<git sha>.pt'''
        return Path(f"models/{pt.short(name, seed, price_seed, params)}-{sha}.pt")

    @staticmethod
    def find(name: Name, seed: int, price_seed: int,
             params: PriceParams = PriceParams(), *, sha: str | None = None):
        '''
        The model file for this configuration: the given sha, else the newest
        one of any sha, else the pre-sha name (<name>-s<seed>[-p<ps>].pt).
        '''
        if sha is not None:
            path = pt.model(name, seed, price_seed, params, sha=sha)
            return path if path.exists() else None
        short = re.escape(pt.short(name, seed, price_seed, params))
        match = re.compile(rf"^{short}-[0-9a-f]{{7,}}(-dirty)?\.pt$")
        found = [p for p in Path("models").glob("*.pt") if match.match(p.name)]
        if found:
            return max(found, key=lambda p: p.stat().st_mtime)
        legacy = Path(f"models/{pt.tag(name, seed, price_seed, params)}.pt")
        return legacy if legacy.exists() else None


def git_sha() -> str:
    '''Short HEAD sha, with "-dirty" if tracked files have changes.'''
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")


def experiment(name: Name, seed: int, price_seed: int,
               params: PriceParams = PriceParams(), *, sha: str) -> str:
    '''
    TensorBoard run name: <YYYY-MM-DD-HH-MM>-<git sha>-<short name>, the
    short name carrying the non-default seed/price seed/fold, e.g. "-s1-p2".
    '''
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    return f"{stamp}-{sha}-{pt.short(name, seed, price_seed, params)}"


def train(
        name: Name,
        *,
        seed: int = 0,
        price_seed: int = 0,
        params: PriceParams = PriceParams(),
        force: bool = False,
        progress: bool = False):
    '''
    Trains on the train split for a fixed number of epochs and saves
    models/<short name>-<git sha>.pt. Skips if that file exists (same code,
    same configuration) unless force. progress shows a progress bar (off by
    default: noga.report trains in parallel, logging to a file).
    '''
    sha = git_sha()
    out = pt.model(name, seed, price_seed, params, sha=sha)

    if out.exists() and not force:
        print(f"Skipping training: {out} already exists")
        return

    pl.seed_everything(seed)

    train_ds, test_ds = load_data(price_seed=price_seed, params=params)
    model = Model(name=name, test=test_ds)
    logger = TensorBoardLogger(
        "logs", name="",
        version=experiment(name, seed, price_seed, params, sha=sha))

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        deterministic=True,
        log_every_n_steps=1,  # all tags are per epoch; avoids a warning
        limit_val_batches=0,
        enable_checkpointing=False,
        logger=logger,
        enable_progress_bar=progress)

    trainer.fit(model, dl(train_ds))

    out.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out)


def load_model(
        name: Name,
        *,
        seed: int = 0,
        price_seed: int = 0,
        params: PriceParams = PriceParams(),
        sha: str | None = None):
    '''The model of the given sha, else the newest one (see pt.find).'''
    path = pt.find(name, seed, price_seed, params, sha=sha)
    if path is None:
        raise FileNotFoundError(
            f"no model for {pt.short(name, seed, price_seed, params)}"
            + (f" at {sha}" if sha else ""))
    model = Model(name=name)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def predict(model: Model, ds: Data) -> np.ndarray:
    '''Predictions in MW.'''
    with torch.no_grad():
        pred = torch.cat([
            model(m, d, t, X)
            for m, d, t, X, _, _ in dl(ds, shuffle=False)])
    return pred.numpy() * Y_SCALE


def predict_prices(model: Model, ds: Data) -> np.ndarray:
    '''Predicted (p_o, p_u) in ₪/MWh, shape (n, 2).'''
    with torch.no_grad():
        pred = torch.cat([
            model.prices(m, d, t, X)
            for m, d, t, X, _, _ in dl(ds, shuffle=False)])
    return pred.numpy() * P_SCALE


if __name__ == "__main__":
    train("cost")
