"""
Data-related functions

"""

import numpy as np


def batch_generator(data, batch_size, seqlen):
    indices = np.arange(data['x'].shape[0])
    np.random.shuffle(indices)

    for i in range(0, indices.shape[0], batch_size):
        batch_indices = indices[i:i + batch_size]
        x_batch = data['x'][batch_indices]

        real_bs = batch_indices.shape[0]

        batch_len = int(np.max(np.sum(x_batch[:, :, -1], axis=-1)))
        # Trim batch
        batch_data = {}
        for key, dat in data.items():
            batch_data[key] = dat[batch_indices, :batch_len, :]
            # if real_bs < batch_size:
                # bd_ = np.zeros((batch_size,) + batch_data[key].shape[1:])
                # bd_[:real_bs, :, :] = batch_data[key]
                # batch_data[key] = bd_

        if seqlen:
            reset = True
            for j in range(0, batch_len, seqlen):
                sub_batch = {}
                for key, dat in batch_data.items():
                    sub_batch[key] = dat[:, j:j + seqlen, :]
                yield sub_batch, reset
                reset = False
        else:
            yield batch_data, True


def split(n_samples, testdata_size):
    """Split a dataset into a train and test set."""
    indices = np.arange(n_samples)
    np.random.seed(13)
    np.random.shuffle(indices)

    tst_mask = indices < testdata_size
    trn_mask = indices >= testdata_size

    assert np.sum(trn_mask) + np.sum(tst_mask) == n_samples

    return trn_mask, tst_mask    

    train_idx = indices[:-testdata_size]
    test_idx  = indices[-testdata_size:]

    data_trn = {}
    data_tst = {}

    for name, dat in data.items():
        data_trn[name] = dat[train_idx]
        data_tst[name] = dat[test_idx]

    return data_trn, data_tst


def cv_split(n_samples, n_layer, n_folds, tst_fold):
    """Split data into train and test for CV."""
    trn_folds = set(range(n_folds - n_layer))
    if tst_fold not in trn_folds:
        raise Exception('Invalid tst_fold={}'.format(tst_fold))
    trn_folds = sorted(trn_folds - {tst_fold, })

    #n_samples = raw.shape[0] #n_samples
    samples_per_fold = (n_samples + n_folds - 1) // n_folds
    sample_idx = np.tile(np.arange(n_folds), samples_per_fold)[:n_samples]
    np.random.seed(13)
    np.random.shuffle(sample_idx)

    trn_mask = np.in1d(sample_idx, trn_folds)
    tst_mask = sample_idx == tst_fold

    return trn_mask, tst_mask