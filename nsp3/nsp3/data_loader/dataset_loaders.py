import torch
import numpy as np
from nsp3.base import DatasetBase


class NSPData(DatasetBase):
    """ NetsurfP 2.0 dataset .npz file """

    def __init__(self, *args, **kwargs):
        super(NSPData, self).__init__(*args, **kwargs)


class NSPDataOnlyEncoding(DatasetBase):
    """ NetsurfP 2.0 dataset .npz file with only input encodings"""
    
    def __init__(self, *args, **kwargs):
        super(NSPDataOnlyEncoding, self).__init__(*args, **kwargs)
        self.X = self.X[:, :, :20]
