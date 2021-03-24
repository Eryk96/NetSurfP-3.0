
"""
Objectives for keras with masks.

"""

import keras.objectives
import keras.metrics
import keras.backend as K


def get_categorical_crossentropy(masked='ignore'):
    """Categorical crossentropy."""
    masked = masked.lower()
    if masked not in ('ignore', 'apply', 'remove'):
        raise ValueError('masked must be one of "ignore", "apply", "remove"')

    def categorical_crossentropy_(y_true, y_pred):
        if masked == 'apply':
            mask = y_true[:, :, -1]
            y_true = y_true[:, :, :-1]
            return keras.objectives.categorical_crossentropy(y_true, y_pred) * mask
        elif masked == 'remove':
            y_true = y_true[:, :, :-1]

        return keras.objectives.categorical_crossentropy(y_true, y_pred)

    return categorical_crossentropy_


def get_categorical_accuracy(masked='ignore'):
    """Accuracy for 1 of k classes predictions."""
    masked = masked.lower()
    if masked not in ('ignore', 'apply', 'remove'):
        raise ValueError('masked must be one of "ignore", "apply", "remove"')

    def categorical_accuracy_(y_true, y_pred):
        if masked == 'apply':
            mask = y_true[:, :, -1]
            y_true = K.argmax(y_true[:, :, :-1], axis=-1)
            y_pred = K.argmax(y_pred, axis=-1)
            hits = K.cast(K.equal(y_true, y_pred), 'float32')
            return K.sum(hits * mask) / K.sum(mask)
        elif masked == 'remove':
            y_true = y_true[:, :, :-1]

        return keras.metrics.categorical_accuracy(y_true, y_pred)

    return categorical_accuracy_


def get_mse(masked='ignore', n_out=1):
    """Mean squared error."""
    masked = masked.lower()
    if masked not in ('ignore', 'apply', 'remove'):
        raise ValueError('masked must be one of "ignore", "apply", "remove"')

    def mse_(y_true, y_pred):
        if masked == 'apply':
            
            mask = y_true[:, :, -1:]
            y_true = y_true[:, :, :-1]
            
            #n_out = K.shape(y_true)[2]
            #if K.all(K.greater(n_out, 1)):
            #    mask = K.repeat_elements(mask, n_out, axis=2)
            if n_out > 1:
               mask = K.repeat_elements(mask, n_out, axis=2)

            v = K.square(y_pred - y_true) * mask
            return K.sum(v) / K.sum(mask)
        elif masked == 'remove':
            y_true = y_true[:, :, :-1]

        return K.mean(K.square(y_pred - y_true), axis=-1)

    return mse_


def get_pcc(masked='ignore'):
    """Pearson correlation coefficient."""
    masked = masked.lower()
    if masked not in ('ignore', 'apply', 'remove'):
        raise ValueError('masked must be one of "ignore", "apply", "remove"')

    def pcc(y_true, y_pred):
        if masked == 'apply':
            mask = K.flatten(y_true[:, :, -1:])
            y_pred = K.flatten(y_pred[:, :, :1]) * mask
            y_true = K.flatten(y_true[:, :, :-1]) * mask
            Sxy = K.sum(K.prod(K.stack((y_pred, y_true)), axis=0))
            Sx = K.sum(y_pred)
            Sx2 = K.sum(y_pred**2)
            Sy = K.sum(y_true)
            Sy2 = K.sum(y_true**2)
            N = K.sum(mask)
            r = (Sxy - Sx * Sy / N) / K.sqrt((Sx2 - Sx**2 / N) * (Sy2 - Sy**2 / N))
            return r
        elif masked == 'remove':
            y_true = y_true[:, :, :-1]

        Sxy = K.sum(K.prod(K.stack((y_pred, y_true)), axis=0))
        Sx = K.sum(y_pred)
        Sx2 = K.sum(y_pred**2)
        Sy = K.sum(y_true)
        Sy2 = K.sum(y_true**2)
        N = K.sum(K.ones_like(y_pred))
        r = (Sxy - Sx * Sy / N) / K.sqrt((Sx2 - Sx**2 / N) * (Sy2 - Sy**2 / N))
        return r

    return pcc
