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
            nn.Linear(10, 64),
            nn.ReLU(),
        )

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        return self.out(x)

    def training_step(self, batch, batch_idx):
        (
            # time,
            # wind_dir,

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
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=.1)


class Data(Dataset):
    def __init__(self):
        data = pd.read_csv("data/data.csv")
        print(*enumerate(data.columns), sep="\n")

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
        batch_size=32,
        shuffle=False)

    model = Model()
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        devices=1)

    trainer.fit(model, dl)
