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
EPOCHS = 300
BATCH_SIZE = 5_000
# Adam for every model, no weight decay.
LR = 1e-3
# First test day; training uses everything before it. 2025-07-01 is the second
# rolling-origin fold (test 2025-H2). Set with the NOGA_TRAIN_END variable.
TRAIN_END = os.environ.get("NOGA_TRAIN_END", "2025-01-01")

# MODEL
M_EMBD = 2
D_EMBD = 2
T_EMBED = 2
N_LAYERS = 2
HIDDEN_SIZE = 16
DROPOUT = 0.1
# Mean excess per slot (₪/h: ₪/MWh x MW) -> M₪ per year.
COST_PER_YEAR = 8760 / 1e6
TEST_EVERY = 25  # epochs between cost/test evaluations

DAY_IN_5_MIN = 288
# Demand history: the same slot on each of the last HISTORY_LEN days,
# HISTORY_STEP rows (one day) apart: t-7d ... t-24h, so the attention picks
# which past day today resembles (yesterday, or last week after a weekend).
HISTORY_LEN = 3
HISTORY_STEP = DAY_IN_5_MIN
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
# Models with a price head predicting p_da, p_o, p_u (multi-task training
# only; the slot's own prices are targets, never inputs).
PRICE_HEADS: set[Name] = {"cost"}
PRICE_WEIGHT = 1.0
# Day-ahead price history: PRICE_LEN prices ending 24h before t, PRICE_STEP
# rows (30 min, one settlement period) apart: t-24h-1.5h ... t-24h.
PRICE_LEN = 4
PRICE_STEP = 6


# X = [demand history | weather at t | price history].
N = HISTORY_LEN + len(FEATURES) + PRICE_LEN
# Columns of X the network sees (plus the calendar embeddings): the weather and
# the day-ahead price at t-24h (the last price-history column), so the forecast
# margin can follow the expected price level. The rest of the demand and price
# histories enter only through their attentions.
NET_COLS = [*range(HISTORY_LEN, HISTORY_LEN + len(FEATURES)), N - 1]
N_NET = len(NET_COLS)


class Model(pl.LightningModule):
    def __init__(self, name: Name, test: "Data | None" = None):
        super().__init__()

        self.name: Name = name
        self.loss = loss_fns[name]
        self.test = test  # logged as cost/test during training

        # max_norm=1 keeps every embedding vector within the unit ball, on the
        # scale of the standardised weather, so the calendar can't grow to
        # drown it out.
        self.day = nn.Embedding(7, D_EMBD, max_norm=1)
        self.month = nn.Embedding(12, M_EMBD, max_norm=1)
        self.time = nn.Embedding(DAY_IN_5_MIN, T_EMBED, max_norm=1)

        self.net = nn.Sequential(
            nn.Linear(N_NET + D_EMBD + M_EMBD + T_EMBED, HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            *[
                nn.Sequential(
                    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                    nn.LeakyReLU(),
                    nn.LayerNorm(HIDDEN_SIZE),
                    nn.Dropout(DROPOUT),
                ) for _ in range(N_LAYERS)
            ],
            # Attention logits, log a, b (see demand).
            nn.Linear(HIDDEN_SIZE, HISTORY_LEN + 2),
        )
        # Created last so the seeded init matches earlier commits.
        head = cast(nn.Linear, self.net[-1])
        with torch.no_grad():  # start from a = 1
            head.weight[HISTORY_LEN].zero_()
            head.bias[HISTORY_LEN].zero_()
        # Shares the hidden representation with the demand forecast.
        # Attention logits over the price history, log a, log(p_o/p_da),
        # log(p_u/p_o) (see price_pred).
        self.price = (
            nn.Linear(HIDDEN_SIZE, PRICE_LEN + 3)
            if name in PRICE_HEADS else None)
        if self.price is not None:
            with torch.no_grad():  # start from a = 1
                self.price.weight[PRICE_LEN].zero_()
                self.price.bias[PRICE_LEN].zero_()
        # Data is in real units (MW, ₪/MWh); these train-split scales, set by
        # train() via set_scale and saved with the model, keep the network's
        # inputs, outputs and losses ~1:
        #   x_mean, x_std  mean and std of each network input column (weather,
        #                  price at t-24h; standardised)
        #   y_scale  std of the day-over-day demand change y(t) - y(t-24h),
        #            the scale of the shift b and of the demand losses
        #   p_scale  mean day-ahead price, the price scale in the losses
        self.x_mean: Tensor
        self.x_std: Tensor
        self.y_scale: Tensor
        self.p_scale: Tensor
        self.register_buffer("x_mean", torch.zeros(N_NET))
        self.register_buffer("x_std", torch.ones(N_NET))
        self.register_buffer("y_scale", torch.tensor(1.0))
        self.register_buffer("p_scale", torch.tensor(1.0))

    def set_scale(self, ds: "Data"):
        self.x_mean.copy_(ds.X[:, NET_COLS].mean(0))
        self.x_std.copy_(ds.X[:, NET_COLS].std(0))
        self.y_scale.fill_((ds.y - ds.X[:, HISTORY_LEN - 1]).std())
        self.p_scale.fill_(ds.c[:, 0].mean())

    def scaled_loss(self, pred: Tensor, y: Tensor, c: Tensor):
        '''The training loss on scaled demand and prices.'''
        return self.loss(
            pred / self.y_scale, y / self.y_scale, c / self.p_scale)

    def hidden(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        f = torch.concat([
            self.month(month),
            self.day(day),
            self.time(time),
            (X[:, NET_COLS] - self.x_mean) / self.x_std
        ], dim=1)
        return self.net[:-1](f)

    def demand(self, z: Tensor, X: Tensor):
        '''
        Forecast (MW) = a * attention + b, a = exp(s) > 0, both per slot;
        b is in units of y_scale.
        '''
        hist = X[:, :HISTORY_LEN]

        out = self.net[-1](z)

        att = (hist * torch.softmax(out[:, :HISTORY_LEN], dim=1)).sum(dim=1)
        a = torch.exp(out[:, HISTORY_LEN])
        b = out[:, HISTORY_LEN + 1] * self.y_scale
        return a * att + b

    def forward(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        return self.demand(self.hidden(month, day, time, X), X)

    def price_pred(self, z: Tensor, X: Tensor):
        '''
        Predicted (p_da, p_o, p_u) in ₪/MWh:
            p_da = a * attention(price history),  a = exp(s) > 0
            p_o  = exp(s_o) * p_da,  p_u = exp(s_u) * p_o
        All positive, since the price history is.
        '''
        assert self.price is not None
        hist = X[:, -PRICE_LEN:]
        out = self.price(z)
        att = (hist * torch.softmax(out[:, :PRICE_LEN], dim=1)).sum(dim=1)
        p_da = torch.exp(out[:, PRICE_LEN]) * att
        p_o = torch.exp(out[:, PRICE_LEN + 1]) * p_da
        p_u = torch.exp(out[:, PRICE_LEN + 2]) * p_o
        return torch.stack([p_da, p_o, p_u], dim=1)

    def prices(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        '''Predicted (p_da, p_o, p_u) in ₪/MWh.'''
        return self.price_pred(self.hidden(month, day, time, X), X)

    def training_step(self, batch, batch_idx):
        month, day, time, X, y, c = batch

        z = self.hidden(month, day, time, X)
        pred = self.demand(z, X)
        loss = self.scaled_loss(pred, y, c)

        if self.name != "cost":
            self.log_epoch(f"loss/{self.name}", loss, len(y))
        with torch.no_grad():
            cost = loss_fns["cost"](pred, y, c) * COST_PER_YEAR
        self.log_epoch("cost/train", cost, len(y), prog_bar=True)

        if self.price is not None:
            price_loss = self.price_loss(z, X, c)
            self.log_epoch("price/train", price_loss, len(y))
            loss = loss + PRICE_WEIGHT * price_loss

        return loss

    def price_loss(self, z: Tensor, X: Tensor, c: Tensor):
        '''
        Relative squared error of (p_da, p_o, p_u): real scale, but p_u's
        spikes don't dominate.
        '''
        return torch.mean(((self.price_pred(z, X) - c) / c) ** 2)

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
        self.eval()  # no dropout, as in predict()
        with torch.no_grad():
            z = self.hidden(t.month, t.day, t.time, t.X)
            pred = self.demand(z, t.X)
            cost = loss_fns["cost"](pred, t.y, t.c) * COST_PER_YEAR
            self.log("cost/test", cost, prog_bar=True)
            if self.price is not None:
                self.log("price/test", self.price_loss(z, t.X, t.c))
        self.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


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
    Each item predicts a slot t from data at least 24h old: the demand at
    the same slot on each of the last HISTORY_LEN days and the PRICE_LEN
    day-ahead prices ending 24h before t (today's price is set only after the
    procurement decision), plus the weather forecast for t. The first slots,
    without a full history, are skipped. Both histories run oldest to newest,
    so their last column is t-24h.
    Indexable by a list of indices, so a BatchSampler yields whole batches.
    '''

    def __init__(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        start = DAY_IN_5_MIN + max(
            (HISTORY_LEN - 1) * HISTORY_STEP, (PRICE_LEN - 1) * PRICE_STEP)
        t = np.arange(start, len(df))
        lag = t - DAY_IN_5_MIN  # same slot yesterday

        # Real units throughout: MW, ₪/MWh (the model scales, see set_scale).
        y = torch.tensor(df["actual"].to_numpy(), dtype=torch.float32)
        steps = np.arange(HISTORY_LEN - 1, -1, -1) * HISTORY_STEP
        hist = y[lag[:, None] - steps]

        X = torch.tensor(df[FEATURES].to_numpy()[t], dtype=torch.float32)
        steps = np.arange(PRICE_LEN - 1, -1, -1) * PRICE_STEP
        p_hist = torch.tensor(
            df[P_DA].to_numpy()[lag[:, None] - steps], dtype=torch.float32)
        X = torch.concat([X, p_hist], dim=1)

        self.month = torch.tensor(df["month"].to_numpy()[t].astype(int))
        self.day = torch.tensor(df["day"].to_numpy()[t].astype(int))
        self.time = torch.tensor((df["time"].to_numpy()[t] / 5).astype(int))
        self.X = torch.concat([hist, X], dim=1)
        self.y = y[t]
        self.c = torch.tensor(
            df[[P_DA, P_O, P_U]].to_numpy()[t], dtype=torch.float32)

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
    model.set_scale(train_ds)
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
    return pred.numpy()


def predict_prices(model: Model, ds: Data) -> np.ndarray:
    '''Predicted (p_da, p_o, p_u) in ₪/MWh, shape (n, 3).'''
    with torch.no_grad():
        pred = torch.cat([
            model.prices(m, d, t, X)
            for m, d, t, X, _, _ in dl(ds, shuffle=False)])
    return pred.numpy()


if __name__ == "__main__":
    train("cost")
