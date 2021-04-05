import numpy as np
from torch.utils.data import DataLoader

class DataLoaderBase(DataLoader):
    """ Base class for all data loader """

    def split_validation(self) -> DataLoader:
        """ Return a `torch.utils.data.DataLoader` for validation, or None if not available. """

        raise NotImplementedError

    def get_test(self) -> list:
        """ Return a `List` containing test sets, or None if not available. """
        
        raise NotImplementedError