#!/usr/bin/env python
"""
DOC
"""

import numpy as np

# import netsurfp.objectives
# from . import objectives

#
# Metrics
#


def remove_masked(y_true, y_pred):
    mask = y_true[:, :, -1] == 1
    return y_true[:, :, :-1][mask], y_pred[mask]


def f1_score(y_true, y_pred):
    import sklearn.metrics
    y_true, y_pred = remove_masked(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    return sklearn.metrics.f1_score(y_true, y_pred)


def acc(y_true, y_pred):
    y_true, y_pred = remove_masked(y_true, y_pred)
    assert y_true.shape == y_pred.shape
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    return np.average(y_true == y_pred)


def pcc(y_true, y_pred):
    import scipy.stats
    y_true, y_pred = remove_masked(y_true, y_pred)
    return scipy.stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]


def mcc(y_true, y_pred):
    from sklearn.metrics import matthews_corrcoef
    y_true, y_pred = remove_masked(y_true, y_pred)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    return matthews_corrcoef(y_true.flatten(), y_pred.flatten())


def fpr(y_true, y_pred):
    y_true, y_pred = remove_masked(y_true, y_pred)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    N = np.sum(1 - y_true)  # Total Negatives
    FP = np.sum((y_true != y_pred).astype(int) * (1 - y_true))

    return FP / N


#


def q3acc(y_true_q8, y_pred_q8):
    """Transforms Q8 to Q3 and assesses the accuracy of the predictions."""
    y_true_q8, y_pred_q8 = remove_masked(y_true_q8, y_pred_q8)

    # old: HGBECTSI
    # cnv: HHEECCCC
    # new: GHIBESTC
    # cnv: HHHEECCC

    def q8toq3(y_q8_prob):
        y_cat = np.argmax(y_q8_prob, axis=-1)
        # 0, 1, 2 -> 0
        y_cat[y_cat <= 2] = 0
        # 3 and 4 -> 1
        y_cat[y_cat == 3] = 1
        y_cat[y_cat == 4] = 1
        # 5, 6, 7 -> 2
        y_cat[5 <= y_cat] = 2
        return y_cat

    y_true_q3 = q8toq3(y_true_q8)
    y_pred_q3 = q8toq3(y_pred_q8)
    return np.average(y_true_q3 == y_pred_q3)


#
#
#


def q3(raw_data):
    """3-category secondary structure."""
    metrics = [acc]

    # data
    mask = raw_data[:, :, 50]
    dat_ = np.stack((np.sum(raw_data[:, :, 57:60], axis=-1),
                     np.sum(raw_data[:, :, 60:62], axis=-1),
                     np.sum(raw_data[:, :, 62:65], axis=-1),
                     mask), axis=2) #yapf: disable

    return 'clf', metrics, dat_


def q8(raw_data):
    """8-category secondary structure."""
    metrics = [acc, q3acc]

    # data
    mask = raw_data[:, :, 50:51]
    y = np.concatenate((raw_data[:, :, 57:65], mask), axis=2)

    return 'clf', metrics, y


#


def rsa(raw_data, mode='iso'):
    """Relative solvent surface accessibility."""
    name = 'y_{}rsa'.format(mode)
    metrics = [pcc]

    seq_mask = raw_data[:, :, 50:51]
    dis_mask = raw_data[:, :, 51:52]
    ukn_mask = np.any(
        raw_data[:, :, 0:20], axis=-1, keepdims=True).astype('float')

    mask = seq_mask * dis_mask * ukn_mask

    if mode.lower() == 'iso':
        y = np.concatenate((raw_data[:, :, 55:56], mask), axis=2)
    elif mode.lower() == 'cpx':
        y = np.concatenate((raw_data[:, :, 56:57], mask), axis=2)
    else:
        raise ValueError('Unknown RSA: {} (valid: iso or cpx)'.format(mode))

    return 'reg', metrics, y


#


def asa(raw_data, mode='iso'):
    """Absolute solvent surface accessibility."""
    name = 'y_{}asa'.format(mode)
    metrics = [pcc]

    seq_mask = raw_data[:, :, 50:51]
    dis_mask = raw_data[:, :, 51:52]
    ukn_mask = np.any(
        raw_data[:, :, 0:20], axis=-1, keepdims=True).astype('float')

    mask = seq_mask * dis_mask * ukn_mask

    if mode.lower() == 'iso':
        y = np.concatenate((raw_data[:, :, 53:54], mask), axis=2)
    elif mode.lower() == 'cpx':
        y = np.concatenate((raw_data[:, :, 54:55], mask), axis=2)
    else:
        raise ValueError('Unknown RSA: {} (valid: iso or cpx)'.format(mode))

    return 'reg', metrics, y


def isoasa(raw_data):
    return asa(raw_data, 'iso')


def cpxasa(raw_data):
    return asa(raw_data, 'cpx')


#


def isorsa(raw_data):
    """Relative solvent surface accessibility.

    Calculated on the isolated molecule (as opposed to complexed in biounit)

    """
    return rsa(raw_data, 'iso')


def cpxrsa(raw_data):
    """Relative solvent surface accessibility.

    Calculated on the isolated molecule (as opposed to complexed in biounit)

    """
    return rsa(raw_data, 'cpx')


def disorder(raw_data):
    """Disordered residue or not"""
    name = 'y_disorder'
    #metrics = {name: objectives.get_categorical_accuracy(masked='apply')}
    metrics = [mcc, acc, fpr]

    # data
    mask = raw_data[:, :, 50:51]
    y = np.concatenate(
        (raw_data[:, :, 51:52], 1.0 - raw_data[:, :, 51:52], mask), axis=2)

    return 'clf', metrics, y


def interface(raw_data):
    """Protein interface."""
    metrics = [acc, mcc]

    mask = raw_data[:, :, 50:51] * raw_data[:, :, 51:52]

    asa_iso = raw_data[:, :, 53:54]
    asa_cpx = raw_data[:, :, 54:55]

    ifc = ((asa_iso - asa_cpx) >= 1).astype('int')
    y = np.concatenate((ifc, 1 - ifc, mask), axis=2)

    # ifc = 1 - ((asa_iso - asa_cpx) / 240.5)
    # y = np.concatenate((np.clip(ifc, 0., 1.), mask), axis=2)

    return 'clf', metrics, y


#


def mae(y_true, y_pred):
    """Minimum absolute error for angles."""
    y_true, y_pred = remove_masked(y_true, y_pred)

    #Convert back to angle in degrees
    y_true = np.arctan2(y_true[:, 0], y_true[:, 1]) * (180 / np.pi)
    y_pred = np.arctan2(y_pred[:, 0], y_pred[:, 1]) * (180 / np.pi)

    err = np.abs(y_true - y_pred)
    return np.mean(np.fmin(err, 360 - err))


def angle(raw_data, aname, mask_boundary=True):
    name = 'y_' + aname
    metrics = [mae]

    if aname == 'phi':
        dat_pp = np.copy(raw_data[:, :, 65:66])
    elif aname == 'psi':
        dat_pp = np.copy(raw_data[:, :, 66:67])
    else:
        raise Exception()

    seq_mask = raw_data[:, :, 50:51]
    dis_mask = raw_data[:, :, 51:52]
    mask = seq_mask * dis_mask
    if mask_boundary:
        mask *= (dat_pp != 360).astype('float')

    #Convert to radian
    dat_pp = dat_pp * (np.pi / 180)
    dat_sin = np.sin(dat_pp)
    dat_cos = np.cos(dat_pp)

    y = np.concatenate((dat_sin, dat_cos, mask), axis=2)
    return 'reg_tanh', metrics, y


def phi(raw_data):
    """Phi torsion angle"""
    return angle(raw_data, 'phi')


def psi(raw_data):
    """Phi and psi torsion angles"""
    return angle(raw_data, 'psi')


def phi_nomask(raw_data):
    """Phi torsion angle"""
    return angle(raw_data, 'phi', mask_boundary=False)


def psi_nomask(raw_data):
    """Phi and psi torsion angles"""
    return angle(raw_data, 'psi', mask_boundary=False)


#
# --
#


def get_output(name):
    return globals()[name]
