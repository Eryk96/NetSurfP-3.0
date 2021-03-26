import torch
from nsp3.base import DatasetBase, DatasetBaseHdf5


class NSPData(DatasetBase):
    """ NetsurfP 2.0 dataset .npz file """

    def __init__(self, *args, **kwargs):
        super(NSPData, self).__init__(*args, **kwargs)


class NSPDataOnlyEncoding(DatasetBase):
    """ NetsurfP 2.0 dataset .npz file with only input encodings"""

    def __init__(self, *args, **kwargs):
        super(NSPDataOnlyEncoding, self).__init__(*args, **kwargs)
        self.X = self.X[:, :, :20]


class NSPEmbeddingData(DatasetBaseHdf5):
    """ NetsurfP 3.0 embedding dataset hdf5 file """

    def __init__(self, *args, **kwargs):
        super(NSPEmbeddingData, self).__init__(*args, **kwargs)


class NSPOnlyEmbeddingData(DatasetBaseHdf5):
    """ NetsurfP 3.0 dataset hdf5 file with only input embeddings"""

    def __init__(self, *args, **kwargs):
        super(NSPOnlyEmbeddingData, self).__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, torch.tensor):
        """ Returns input, label and mask data
        Args:
            index: Index at the array
        """
        return self.X[index, :, 68:], self.y[index], self.lengths[index]
