import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# 1. Define the System (Model + Training Logic)
class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=.1)


# 2. Prepare Dummy Data (y = 2x)
x = torch.randn(128, 1)
y = 2 * x
dataset = TensorDataset(x, y)
train_loader = DataLoader(dataset, batch_size=32)

# 3. Train
model = SimpleModel()
trainer = pl.Trainer(max_epochs=10, accelerator="cpu")
trainer.fit(model, train_loader)
