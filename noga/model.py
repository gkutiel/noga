import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 1. Define the System (Model + Training Logic)
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
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
            temperature,
            humidity,
            wind_speed
        ], dim=1)

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
            lr=1e-3)


class Data(Dataset):
    def __init__(self):
        data = pd.read_csv("data/data.csv")

        self.time = torch.tensor(
            data.iloc[:, 1:4].values,
            dtype=torch.int16)

        self.temperature = torch.tensor(
            data.iloc[:, 4:12].values,
            dtype=torch.float32)

        self.humidity = torch.tensor(
            data.iloc[:, 12:16].values,
            dtype=torch.float32)

        self.wind_dir = torch.tensor(
            data.iloc[:, 16:19].values,
            dtype=torch.int16)

        self.wind_speed = torch.tensor(
            data.iloc[:, 19:22].values,
            dtype=torch.float32)

        self.y = torch.tensor(
            data.iloc[:, -1].values,
            dtype=torch.float32)

        # NORMALIZE THE DATA
        self.temperature = (
            self.temperature - self.temperature.mean(0)) / self.temperature.std(0)

        self.humidity = (
            self.humidity - self.humidity.mean(0)) / self.humidity.std(0)

        self.wind_speed = (
            self.wind_speed - self.wind_speed.mean(0)) / self.wind_speed.std(0)

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
    dataset = Data()
    dl = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False)

    model = Model()
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="cpu",
        devices=1)

    trainer.fit(model, dl)
