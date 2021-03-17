import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from nsp3.base import DataLoaderBase


class NSPDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """
    def __init__(self, file, dataset, batch_size, shuffle, validation_split, nworkers):

        self.train_dataset = dataset(file)

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)
