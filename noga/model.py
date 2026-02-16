import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader, Dataset

UNDER = 5
EMBED_SIZE = 5
LR = 1e-3
EPOCHS = 50
HIDDEN_SIZE = 64
BATCH_SIZE = 256
DEPTH = 6
NEGATIVE_SLOPE = 0.1


def norm(data: torch.Tensor) -> torch.Tensor:
    return (data - data.mean(dim=0)) / data.std(dim=0)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.month_embedding = nn.Embedding(13, EMBED_SIZE)
        self.day_embedding = nn.Embedding(7, EMBED_SIZE)
        self.hour_embedding = nn.Embedding(24, EMBED_SIZE)

        self.wind_dir_embedding = nn.Embedding(361, EMBED_SIZE)

        self.net = nn.Sequential(
            # nn.Linear(6 * EMBED_SIZE + 15, HIDDEN_SIZE),
            nn.Linear(3 * EMBED_SIZE + 15, HIDDEN_SIZE),
            # nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Tanh(),
            *[nn.Sequential(
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
                # nn.Tanh()
                nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)
            ) for _ in range(DEPTH)]
        )

        self.out = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.net(x)
        return self.out(x)

    def step(self, batch, batch_idx):
        (
            time,
            wind_dir,

            temperature,
            humidity,
            wind_speed
        ), y = batch

        x = torch.cat([
            # self.wind_dir_embedding(wind_dir).view(len(wind_dir), -1),
            self.month_embedding(time[:, 0]),
            self.day_embedding(time[:, 1]),
            self.hour_embedding(time[:, 2]),

            temperature,
            humidity,
            wind_speed
        ], dim=1)

        return self.forward(x), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.step(batch, batch_idx)
        l1_loss = nn.functional.l1_loss(y_hat, y)

        self.log(
            "l1_loss",
            l1_loss,
            prog_bar=True)

        diff = y_hat - y
        cost = diff[diff > 0].sum() - diff[diff < 0].sum() * UNDER
        self.log("cost", cost, prog_bar=True)

        return l1_loss

    def predict_step(self, batch, batch_idx):
        y_hat, _ = self.step(batch, batch_idx)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=LR)


class Data(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.time = torch.tensor(
            data.iloc[:, 1:4].values,
            dtype=torch.int32)

        self.temperature = torch.tensor(
            data.iloc[:, 4:12].values,
            dtype=torch.float32)

        self.humidity = torch.tensor(
            data.iloc[:, 12:16].values,
            dtype=torch.float32)

        self.wind_dir = torch.tensor(
            data.iloc[:, 16:19].values,
            dtype=torch.int32)

        self.wind_speed = torch.tensor(
            data.iloc[:, 19:22].values,
            dtype=torch.float32)

        self.y = torch.tensor(
            data.iloc[:, -1].values,
            dtype=torch.float32).unsqueeze(1)

        assert self.time[:, 0].min() > 0, "Month should be non-negative"
        assert self.time[:, 0].max() <= 12, f"Month {self.time[:, 0].max()}"
        assert self.time[:, 1].min() >= 0, "Day should be non-negative"
        assert self.time[:, 1].max() < 7, "Day should be in the range [0, 6]"
        assert self.time[:, 2].min() >= 0, "Hour should be non-negative"
        assert self.time[:, 2].max() < 24, f"Hour {self.time[:, 2].max()}"

        assert self.wind_dir.min() >= 0, "Wind direction should be non-negative"
        assert self.wind_dir.max() <= 360, \
            f"Wind direction should be in the range [1, 360] {self.wind_dir.max()}"

        self.temperature = norm(self.temperature)
        self.humidity = norm(self.humidity)
        self.wind_speed = norm(self.wind_speed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            # NEED EMBEDDING
            self.time[idx],
            self.wind_dir[idx],
            #
            self.temperature[idx],
            self.humidity[idx],
            self.wind_speed[idx]
        ), self.y[idx]


if __name__ == "__main__":
    seed_everything(0)
    data = pd.read_csv("data/data.csv")
    dataset = Data(data)
    train_dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True)

    model = Model()
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",
        deterministic=True,
        devices=1)

    trainer.fit(model, train_dl)

    test_dl = DataLoader(
        dataset,
        batch_size=8000,
        shuffle=False)

    preds = trainer.predict(model, test_dl)
    y_hat = torch.cat(preds, dim=0).squeeze().detach().numpy()
    pred = data[['actual-demand', 'day-ahead-forecast']].copy()
    pred['y_hat'] = y_hat
    pred.round(0).to_csv("data/pred.csv", index=False)

    mae_baseline = (
        pred['actual-demand'] - pred['day-ahead-forecast']).abs().mean()

    mae_model = (pred['actual-demand'] - pred['y_hat']).abs().mean()
    print(f"Baseline MAE: {mae_baseline:.2f}")
    print(f"Model MAE: {mae_model:.2f}")
