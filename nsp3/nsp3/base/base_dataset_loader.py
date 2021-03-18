import torch
import numpy as np
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for all datasets
    """

    def __init__(self, data):
        """ Constructor
        Args:
            X (np.array): The array that contains the training data
            y (np.array): The array that contains the test data
        """
        data = torch.from_numpy(np.load(data)['data'])

        self.X = data[:, :, :50].clone().detach().float()
        self.y = data[:, :, 50:].clone().detach().float()
        self.lengths = torch.tensor([sum(labels[:, 0] == 1) for labels in self.y])
        
        self.unknown_nucleotide_mask()

        del data
        
    def unknown_nucleotide_mask(self):
        """ Augments the target with a unknown nucleotide mask
            by finding entries that don't have any residue
        """
        
        # creates a mask based on the one hot encoding
        unknown_nucleotides = torch.max(self.X[:, :, :20], dim=2)
        unknown_nucleotides = unknown_nucleotides[0].unsqueeze(2)
        
        # merge the mask to first position of the targets
        self.y = torch.cat([self.y, unknown_nucleotides], dim=2)
        
    def __getitem__(self, index):
        """ Returns train and test data at an index
        Args:
            index (int): Index at the array
        """
        X = self.X[index]
        y = self.y[index]
        lengths = self.lengths[index]
        
        return X, y, lengths
    
    def __len__(self):
        """Returns the length of the data"""
        return len(self.X)