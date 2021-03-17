"""
Evaluation utils.

"""

import netsurfp2_dev.targets


def eval_preds(raw_data, y_pred, target):
    """Evaluate predictions."""
    target = netsurfp2.targets.get_output(target)
    _, metrics, y_true = target(raw_data)

    #Add in the evaluation mask
    y_true[:, :, -1:] *= raw_data[:, :, 52:53]

    return {m.__name__: m(y_true, y_pred) for m in metrics}