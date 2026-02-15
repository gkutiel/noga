import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_fabric import seed_everything
from torch import nn
from torch.utils.data import DataLoader, Dataset

EMBED_SIZE = 8


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.month_embedding = nn.Embedding(13, EMBED_SIZE)
        self.day_embedding = nn.Embedding(7, EMBED_SIZE)
        self.hour_embedding = nn.Embedding(24, EMBED_SIZE)

        self.net = nn.Sequential(
            nn.Linear(3 * EMBED_SIZE + 15, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
        )

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        (
            time,
            wind_dir,

            temperature,
            humidity,
            wind_speed
        ), y = batch

        x = torch.cat([
            self.month_embedding(time[:, 0]),
            self.day_embedding(time[:, 1]),
            self.hour_embedding(time[:, 2]),
            # self.wind_dir_embedding(wind_dir),

            temperature,
            humidity,
            wind_speed
        ], dim=1)

        # print(x.shape, y.shape)
        # exit(0)
        y_hat = self.forward(x)
        loss = nn.functional.l1_loss(y_hat, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=1e-4)


class Data(Dataset):
    def __init__(self):
        data = pd.read_csv("data/data.csv")

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
    dataset = Data()
    dl = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True)

    model = Model()
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="cpu",
        devices=1)

    trainer.fit(model, dl)
