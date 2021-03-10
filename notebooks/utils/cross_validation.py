import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def partitions(number, k):
    '''
    Distribution of the folds
    Args:
        number: length of the sequence
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions

def get_indices(n_splits, sequences):
    '''
    Indices of the set test
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    fold_sizes = partitions(sequences, n_splits)
    indices = np.arange(sequences).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])

def k_folds(n_splits, sequences):
    '''
    Generates folds for cross validation
    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    indices = np.arange(sequences).astype(int)
    for val_idx in get_indices(n_splits, sequences):
        train_idx = np.setdiff1d(indices, val_idx)
        yield SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)