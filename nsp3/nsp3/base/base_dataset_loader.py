import torch
import numpy as np
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """ Base class for NetSurfP datasets """

    def __init__(self, path: str):
        """ Constructor
        Args:
            path: file path for the dataset
        """

        data = torch.from_numpy(np.load(path)['data'])

        self.X = data[:, :, :50].clone().detach().float()
        self.y = data[:, :, 50:].clone().detach().float()
        self.lengths = torch.tensor([sum(labels[:, 0] == 1) for labels in self.y])

        self.add_unknown_nucleotide_mask()

        del data

    def add_unknown_nucleotide_mask(self):
        """ Augments the target with a unknown nucleotide mask by finding entries that don't have any residue"""

        # creates a mask based on the one hot encoding
        unknown_nucleotides = torch.max(self.X[:, :, :20], dim=2)
        unknown_nucleotides = unknown_nucleotides[0].unsqueeze(2)

        # merge the mask to first position of the targets
        self.y = torch.cat([self.y, unknown_nucleotides], dim=2)

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, torch.tensor):
        """ Returns input, label and mask data
        Args:
            index: Index at the array
        """

        return self.X[index], self.y[index], self.lengths[index]

    def __len__(self):
        """ Returns the length of the data """
        
        return len(self.X)
