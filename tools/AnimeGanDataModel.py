import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from tools.dataset import AnimeDataSet


class AnimeGANDataModel(LightningDataModule):
    def __init__(self, dataset, data_dir, batch_size=5, num_workers=4):
        super().__init__()
        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            raise FileNotFoundError(f'Folder {anime_dir} does not exist')
        self.imageDataSet = AnimeDataSet(dataset=dataset, data_dir=data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.imageDataSet,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
