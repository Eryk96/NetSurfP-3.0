import torch
from torch.utils.data import Dataset


class DatasetBase(Dataset):
    """
    Base class for all datasets
    """

    def __init__(self, dataset, indices = False):
        """ Constructor
        Args:
            X (np.array): The array that contains the training data
            y (np.array): The array that contains the test data
        """

        self.data = torch.tensor(dataset[:, :, :50]).float()
        self.targets = torch.tensor(dataset[:, :, 50:68]).float()
        self.lengths = torch.tensor([sum(target[:, 0] == 1) for target in self.targets])
        
        self.unknown_nucleotide_mask()
        
    def unknown_nucleotide_mask(self):
        """ Augments the target with a unknown nucleotide mask
            by finding entries that don't have any residue
        """
        
        # creates a mask based on the one hot encoding
        unknown_nucleotides = torch.max(self.data[:, :, :20], dim=2)
        unknown_nucleotides = unknown_nucleotides[0].unsqueeze(2)
        
        # merge the mask to first position of the targets
        self.targets = torch.cat([self.targets, unknown_nucleotides], dim=2)
        
    def __getitem__(self, index):
        """ Returns train and test data at an index
        Args:
            index (int): Index at the array
        """
        X = self.data[index]
        y = self.targets[index]
        lengths = self.lengths[index]
        
        return X, y, lengths
    
    def __len__(self):
        """Returns the length of the data"""
        return len(self.data)