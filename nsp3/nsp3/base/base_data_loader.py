import numpy as np
from torch.utils.data import DataLoader


class DataLoaderBase(DataLoader):
    """
    Base class for all data loaders
    """

    def split_validation(self) -> DataLoader:
        """ 
        Splits the dataset into train and validation
        """

        num_train = len(self.train_dataset)
        train_indices = np.array(range(num_train))
        validation_indices = np.random.choice(train_indices, int(num_train*self.validation_split), replace=False)
        
        train_indices = np.delete(train_indices, validation_indices)
        
        # subset the dataset
        train_idx, valid_idx = train_indices, validation_indices
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_dataset = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        valid_dataset = DataLoader(self.train_dataset, sampler=valid_sampler, batch_size=self.batch_size)

        return train_dataset, valid_dataset