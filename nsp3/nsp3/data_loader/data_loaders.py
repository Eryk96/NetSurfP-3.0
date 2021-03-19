import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from nsp3.base import DataLoaderBase
from nsp3.data_loader.dataset_loaders import NSPData

class NSPDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """
    def __init__(self, train_path, test_path=None, batch_size, shuffle, validation_split, nworkers):
        self.train_dataset = NSPData(train_path[0])
        self.valid_dataset = NSPData(train_path[0])

        self.train_sampler = None
        self.valid_sampler = None

        self.test_path = test_path

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers,
            'shuffle': shuffle
        }

        if validation_split:
            self.split(validation_split)
            self.init_kwargs.pop('shuffle')

        super().__init__(self.train_dataset, sampler=self.train_sampler, **self.init_kwargs)

    def split(self, validation_split):
        num_train = len(self.train_dataset)
        train_indices = np.array(range(num_train))
        validation_indices = np.random.choice(train_indices, int(num_train*validation_split), replace=False)
        
        train_indices = np.delete(train_indices, validation_indices)
        
        # subset the dataset
        train_idx, valid_idx = train_indices, validation_indices
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(self.valid_dataset, sampler=self.valid_sampler, **self.init_kwargs)

    def get_test(self):
        test_data = []
        for path in self.test_path:
            test_data.append((path, DataLoader(path, **self.init_kwargs)))
        return test_data