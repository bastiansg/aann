import multiprocessing

from abc import abstractmethod
from torch.utils.data import DataLoader

from torchdata.datapipes.iter import IterDataPipe


N_WORKERS = multiprocessing.cpu_count()


class TrainModule:
    @property
    def name(self):
        return self.__class__.__name__

    def dataset2dl(
        self,
        dataset: IterDataPipe,
        batch_size: int = 128,
        num_workers: int = N_WORKERS,
    ) -> DataLoader:
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        return dl

    @abstractmethod
    def train(self, train_dp_dataset: IterDataPipe):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def predict(self, dp_dataset: IterDataPipe) -> IterDataPipe:
        pass
