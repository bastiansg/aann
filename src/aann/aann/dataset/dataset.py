import torch

from more_itertools import flatten
from torchdata.datapipes.iter import IterableWrapper

from .base_images import base_images


class Dataset:
    def __init__(
        self,
        base_images: dict = base_images,
        items_per_class: int = 1_000,
        shuffle: bool = True,
    ):
        self.base_images = base_images
        self.items_per_class = items_per_class
        self.shuffle = shuffle

        self.num_in_features = len(list(flatten(base_images[0]["image"])))
        self.num_classes = len(base_images)

    def get_dataset_item(self, base_image: dict) -> dict:
        img_features = torch.tensor(base_image["image"]).flatten()
        img_features = img_features.flatten().float()
        dataset_item = {
            "img_features": img_features,
            "y": base_image["class"],
        }

        return dataset_item

    def load(self):
        dp = IterableWrapper(iter(self.base_images), deepcopy=False)
        dp = dp.map(self.get_dataset_item)
        if self.items_per_class > 1:
            dp = dp.repeat(self.items_per_class)

        if self.shuffle:
            dp = dp.shuffle()

        self.train = dp.sharding_filter()
