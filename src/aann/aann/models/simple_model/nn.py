import torch
import warnings

from torch import nn
from torch.optim import Adam
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchmetrics import F1Score
from pytorch_lightning import LightningModule


warnings.filterwarnings("ignore", category=UserWarning)


class Model(LightningModule):
    def __init__(
        self,
        device: str,
        train_dl: DataLoader,
        num_in_features: int,
        num_hidden: int,
        num_classes: int,
        lr: float = 1e-2,
    ):
        super().__init__()

        self.device_ = device
        self.train_dl = train_dl
        self.lr = lr

        self.lin1 = nn.Linear(num_in_features, num_hidden, bias=False)
        self.sig1 = nn.Sigmoid()

        self.lin_out = nn.Linear(num_hidden, num_classes, bias=False)

        self.ce_loss = CrossEntropyLoss()
        self.f1 = F1Score(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )

    def forward(self, img_features: torch.Tensor) -> torch.Tensor:
        x = self.lin1(img_features)
        x = self.sig1(x)

        x = self.lin_out(x)

        return x

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:
        img_features = batch["img_features"].to(self.device_)
        x_out = self.forward(img_features=img_features)

        y = batch["y"]
        loss = self.ce_loss(x_out, y)
        preds = x_out.argmax(dim=1)
        self.f1(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_f1", self.f1, prog_bar=True)

        return loss

    def predict_step(
        self,
        batch: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> list[tuple]:
        img_features = batch["img_features"].to(self.device_)
        x_out = self.forward(img_features=img_features)

        preds = x_out.softmax(dim=1)
        preds = preds.argmax(dim=1)

        return preds

    def train_dataloader(self):
        return self.train_dl

    def configure_optimizers(self) -> Optimizer:
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
