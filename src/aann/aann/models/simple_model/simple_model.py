import os
import torch

import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import RichProgressBar

from torchdata.datapipes.iter import IterDataPipe, IterableWrapper

from aann.utils.path import create_path
from aann.meta.interfaces import TrainModule

from .nn import Model


class SimpleModel(TrainModule):
    def __init__(
        self,
        num_in_features: int,
        num_classes: int,
        # num_hidden: int = 8,
        device: str = "cuda",
        max_epochs: int = 50,
        save_path: str = "/resources/models/simple-model",
        model_name: str = "model",
    ):
        self.num_in_features = num_in_features
        # self.num_hidden = num_hidden
        self.num_classes = num_classes

        self.device = device
        self.max_epochs = max_epochs
        self.save_path = f"{save_path}/{model_name}.pt"

    def get_trainer(self) -> Trainer:
        trainer = Trainer(
            accelerator=self.device,
            max_epochs=self.max_epochs,
            callbacks=[RichProgressBar()],
            enable_checkpointing=False,
            logger=False,
            default_root_dir=None,
        )

        return trainer

    def train(self, train_dp_dataset: IterDataPipe):
        train_dl = self.dataset2dl(train_dp_dataset)
        self.model = Model(
            device=self.device,
            train_dl=train_dl,
            num_in_features=self.num_in_features,
            # num_hidden=self.num_hidden,
            num_classes=self.num_classes,
        )

        self.trainer = self.get_trainer()
        self.trainer.fit(self.model)

    def save(self):
        # create_path(os.path.dirname(self.save_path), overwrite=True)
        create_path(os.path.dirname(self.save_path))
        torch.save(self.model, self.save_path)

    def load(self):
        self.model = torch.load(self.save_path)
        self.trainer = self.get_trainer()

    def predict(self, dp_dataset: IterDataPipe) -> IterDataPipe:
        dl = self.dataset2dl(dp_dataset)
        preds = self.trainer.predict(self.model, dl)
        preds = torch.concat(preds)

        return preds

    def dataset_item_predict(
        self, dataset_item: dict
    ) -> tuple[int, np.ndarray]:
        dp_dataset = IterableWrapper(iter([dataset_item]), deepcopy=False)
        dp_dataset = dp_dataset.sharding_filter()
        dl = self.dataset2dl(dp_dataset)

        pred, confs = self.trainer.predict(self.model, dl)[0]
        pred = pred.item()
        confs = confs.numpy()[0]

        return pred, confs
