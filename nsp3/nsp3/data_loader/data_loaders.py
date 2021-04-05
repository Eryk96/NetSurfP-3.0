import numpy as np
import nsp3.data_loader.dataset_loaders as module_dataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from nsp3.base import DataLoaderBase


class NSPDataLoader(DataLoaderBase):
    """ NSPDataLoader to load NetSurfP training data """

    def __init__(self, train_path: list, test_path: list, dataset_loader: str, batch_size: int, shuffle: bool, 
                    validation_split: float, nworkers: int):
        """ Constructor
        Args:
            train_path: path to the training dataset
            dataset_loader: dataset loader class
            batch_size: size of the batch
            shuffle: shuffles the data (only if validation data is not created)
            validation_split: decimal for the split of the validation
            nworkers: workers for the dataloader class
            test_path: path to the test dataset(s)
        """
        self.dataset_loader = getattr(module_dataset, dataset_loader)

        self.train_dataset = self.dataset_loader(train_path[0])
        self.valid_dataset = self.dataset_loader(train_path[0])

        self.train_sampler = None
        self.valid_sampler = None

        self.test_path = test_path

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers,
            'shuffle': shuffle
        }

        if validation_split:
            self._split(validation_split)
            self.init_kwargs.pop('shuffle')

        super().__init__(self.train_dataset, sampler=self.train_sampler, **self.init_kwargs)

    def _split(self, validation_split: float):
        """ Creates a sampler to extract training and validation data
        Args:
            validation_split: decimal for the split of the validation
        """
        # random indices based off the validation split
        num_train = len(self.train_dataset)
        train_indices = np.array(range(num_train))
        validation_indices = np.random.choice(train_indices, int(
            num_train * validation_split), replace=False)

        train_indices = np.delete(train_indices, validation_indices)

        # subset the dataset
        train_idx, valid_idx = train_indices, validation_indices
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler

    def split_validation(self) -> DataLoader:
        """ Returns the validation data """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(self.valid_dataset, sampler=self.valid_sampler, **self.init_kwargs)

    def get_test(self) -> list:
        """ Returns the test data """
        test_data = []
        for path in self.test_path:
            test_data.append(
                (path, DataLoader(self.dataset_loader(path), **self.init_kwargs)))
        return test_data
