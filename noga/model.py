from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor, nn
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torch.utils.data import RandomSampler, SequentialSampler

from noga.cost import USE_PRICE, Name, loss_fns, optims
from noga.price import P_DA, P_O, P_U, PriceParams, synth

# TRAIN
MAX_EPOCHS = 500
BATCH_SIZE = 10_000

# MODEL
M_EMBD = 2
D_EMBD = 3
T_EMBED = 16
N_LAYERS = 2
HIDDEN_SIZE = 32
Y_SCALE = 100
P_SCALE = 100

DAY_IN_5_MIN = 288
HISTORY_LEN = 10
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
# Known at decision time: log day-ahead price and the over-prediction ratio.
PRICE_FEATURES = ['log_p_da', 'p_o_ratio']

# Models whose training depends on the synthetic price draw.
PRICE_DEPENDENT: set[Name] = USE_PRICE | {"cost-noprice"}


def n_features(name: Name):
    n = len(FEATURES) + HISTORY_LEN
    return n + len(PRICE_FEATURES) if name in USE_PRICE else n


class Model(pl.LightningModule):
    def __init__(self, name: Name):
        super().__init__()

        self.name: Name = name
        self.loss = loss_fns[name]

        self.day = nn.Embedding(7, D_EMBD)
        self.month = nn.Embedding(12, M_EMBD)
        self.time = nn.Embedding(DAY_IN_5_MIN, T_EMBED)

        self.net = nn.Sequential(
            nn.Linear(n_features(name) + D_EMBD + M_EMBD + T_EMBED, HIDDEN_SIZE),
            *[
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                ) for _ in range(N_LAYERS)
            ],
            nn.LeakyReLU(),
            nn.Linear(HIDDEN_SIZE, HISTORY_LEN + 1),
        )

    def forward(self, month: Tensor, day: Tensor, time: Tensor, X: Tensor):
        f = torch.concat([
            self.month(month),
            self.day(day),
            self.time(time),
            X
        ], dim=1)

        hist = X[:, :HISTORY_LEN]

        out = self.net(f)

        hist = (hist * torch.softmax(out[:, :HISTORY_LEN], dim=1)).sum(dim=1)

        return hist + out[:, HISTORY_LEN]

    def step(self, batch, batch_idx, step='train'):
        month, day, time, X, y, c = batch

        pred = self(month, day, time, X)
        loss = self.loss(pred, y, c)

        self.log(f"{step}/{self.name}", loss, prog_bar=True)
        self.log(f"{step}/l1", torch.mean(torch.abs(pred - y)), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, step='val')

    def configure_optimizers(self):
        return optims[self.name](self.parameters())


def with_prices(
        df: pd.DataFrame,
        *,
        price_seed: int,
        params: PriceParams = PriceParams()) -> pd.DataFrame:
    '''Adds synthetic p_o, p_u (one draw per settlement period) and features.'''
    periods = df[["period", P_DA]].drop_duplicates("period")
    periods = synth(periods, seed=price_seed, params=params)

    df = df.merge(periods[["period", P_O, P_U]], on="period", how="left")
    df["log_p_da"] = np.log(df[P_DA]) - 5
    df["p_o_ratio"] = 10 * df[P_O] / df[P_DA]
    return df


class Data(Dataset):
    '''
    Item j predicts slot t = j + DAY_IN_5_MIN + HISTORY_LEN - 1 from the
    HISTORY_LEN demand readings ending 24h before t and the features at t.
    Indexable by a list of indices, so a BatchSampler yields whole batches.
    '''

    def __init__(self, df: pd.DataFrame, use_price: bool):
        df = df.reset_index(drop=True)
        n = len(df) - DAY_IN_5_MIN - HISTORY_LEN + 1
        t = np.arange(n) + DAY_IN_5_MIN + HISTORY_LEN - 1

        y = torch.tensor(df["actual"].to_numpy(), dtype=torch.float32)
        y = y / Y_SCALE
        hist = y.unfold(0, HISTORY_LEN, 1)[:n]

        cols = FEATURES + PRICE_FEATURES if use_price else FEATURES
        X = torch.tensor(df[cols].to_numpy()[t], dtype=torch.float32)

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
    train: 2023-04 .. 2024-08
    cal:   2024-09 .. 2024-12 (early stopping and post-hoc calibration)
    test:  2025
    '''
    year, month = df["year"], df["month"]  # month is 0-based
    train = (year < 2024) | ((year == 2024) & (month < 8))
    cal = (year == 2024) & (month >= 8)
    test = year == 2025
    return df[train], df[cal], df[test]


def load_data(
        *,
        use_price: bool,
        price_seed: int,
        params: PriceParams = PriceParams()):
    data = pd.read_csv("data/data.csv")
    data = with_prices(data, price_seed=price_seed, params=params)
    return tuple(Data(df, use_price) for df in splits(data))


def dl(ds: Dataset, shuffle=True):
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)  # type: ignore
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
        if name not in PRICE_DEPENDENT:
            return f"{name}-s{seed}"
        return f"{name}-s{seed}-p{price_seed}{params_tag(params)}"

    @staticmethod
    def model(name: Name, seed: int, price_seed: int,
              params: PriceParams = PriceParams()):
        return Path(f'models/{pt.tag(name, seed, price_seed, params)}.pt')

    @staticmethod
    def ckpt(name: Name, seed: int, price_seed: int,
             params: PriceParams = PriceParams()):
        return Path(f'models/{pt.tag(name, seed, price_seed, params)}.ckpt')


def train(
        name: Name,
        *,
        seed: int = 0,
        price_seed: int = 0,
        params: PriceParams = PriceParams()):
    pl.seed_everything(seed)

    out = pt.model(name, seed, price_seed, params)

    if out.exists():
        print(f"Skipping training: {out} already exists")
        return

    model = Model(name=name)
    logger = TensorBoardLogger(
        "logs", name=pt.tag(name, seed, price_seed, params))

    early_stopping = EarlyStopping(
        monitor=f"val/{name}",
        patience=20,
        mode="min")

    ckpt = pt.ckpt(name, seed, price_seed, params)
    ckpt.unlink(missing_ok=True)

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
        logger=logger,
        enable_progress_bar=False)

    train_ds, cal_ds, _ = load_data(
        use_price=name in USE_PRICE,
        price_seed=price_seed,
        params=params)

    trainer.fit(model, dl(train_ds), dl(cal_ds, shuffle=False))

    best_weights = torch.load(
        checkpoint.best_model_path,
        weights_only=True)

    out.parent.mkdir(exist_ok=True)
    torch.save(best_weights["state_dict"], out)


def load_model(
        name: Name,
        *,
        seed: int = 0,
        price_seed: int = 0,
        params: PriceParams = PriceParams()):
    model = Model(name=name)
    model.load_state_dict(torch.load(pt.model(name, seed, price_seed, params)))
    model.eval()
    return model


def predict(model: Model, ds: Data) -> np.ndarray:
    '''Predictions in MW.'''
    with torch.no_grad():
        pred = torch.cat([
            model(m, d, t, X)
            for m, d, t, X, _, _ in dl(ds, shuffle=False)])
    return pred.numpy() * Y_SCALE


if __name__ == "__main__":
    train("cost")
