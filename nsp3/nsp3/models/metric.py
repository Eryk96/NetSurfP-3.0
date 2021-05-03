import torch
import torch.nn.functional as F
import numpy as np


def get_mask(labels: torch.tensor, use_disorder_mask: bool = False, use_unknown_mask: bool = False) -> torch.tensor:
    """ Returns mask from labels
    Args:
        labels: tensor containing labels
        use_disorder_mask: apply disorder mask
        use_unknown_mask: apply unknown nucleotide mask
    """
    labels = labels.clone()

    evaluation_mask = labels[:, :, 2]
    zero_mask = labels[:, :, 0] * evaluation_mask
    disorder_mask = labels[:, :, 1]
    unknown_mask = labels[:, :, -1]

    if use_disorder_mask:
        zero_mask = zero_mask * disorder_mask
    if use_unknown_mask:
        zero_mask = zero_mask * unknown_mask

    return zero_mask


def dihedral_to_radians(angle: torch.tensor) -> torch.tensor:
    """ Converts angles to radians
    Args:
        angles: tensor containing angle values
    """
    return angle * np.pi / 180


def arctan_dihedral(sin: torch.tensor, cos: torch.tensor) -> torch.tensor:
    """ Converts sin and cos back to diheral angles
    Args:
        sin: tensor with sin values 
        cos: tensor with cos values
    """
    result = torch.where(cos >= 0, torch.arctan(sin / cos),
                         torch.arctan(sin / cos) + np.pi)
    result = torch.where((sin <= 0) & (cos <= 0), result - np.pi * 2, result)

    return result * 180 / np.pi


def fpr(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns false positive rate
    Args:
        inputs: tensor with binary values
        labels: tensor with binary values
    """
    fp = sum((pred == 1) & (labels == 0))
    tn = sum((pred == 0) & (labels == 0))

    return (fp / (fp + tn)).item()


def fnr(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns false negative rate
    Args:
        inputs: tensor with binary values
        labels: tensor with binary values
    """
    fn = sum((pred == 0) & (labels == 1))
    tp = sum((pred == 1) & (labels == 1))

    return (fn / (fn + tp)).item()


def mcc(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns mathews correlation coefficient
    Args:
        inputs: tensor with binary values
        labels: tensor with binary values
    """
    fp = sum((pred == 1) & (labels == 0))
    tp = sum((pred == 1) & (labels == 1))
    fn = sum((pred == 0) & (labels == 1))
    tn = sum((pred == 0) & (labels == 0))

    mcc = (tp * tn - fp * fn) / torch.sqrt(((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)).float())

    if torch.isnan(mcc):
        return 0

    return mcc.item()


def pcc(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns pearson correlation coefficient
    Args:
        inputs: tensor with predicted values
        labels: tensor with correct values
    """
    x = pred - torch.mean(pred)
    y = labels - torch.mean(labels)

    return (torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))).item()


def mae(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns mean absolute error
    Args:
        inputs: tensor with predicted values
        labels: tensor with correct values
    """
    err = torch.abs(labels - pred)
    return torch.mean(torch.fmin(err, 360 - err)).item()


def accuracy(pred: torch.tensor, labels: torch.tensor) -> float:
    """ Returns accuracy
    Args:
        inputs: tensor with predicted values
        labels: tensor with correct values
    """

    return (sum((pred == labels)) / len(labels)).item()


def metric_ss8(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns SS8 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 7:15], dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)


def metric_ss3(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns SS3 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)

def metric_ss3_from_ss8(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns SS3 metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()[mask == 1]
    
    outputs = F.one_hot(torch.argmax(outputs, dim=2), num_classes=8)
    outputs = torch.max(outputs * structure_mask, dim=2)[0].long()[mask == 1]

    return accuracy(outputs, labels)

def metric_dis_mcc(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns mathews correlation coefficient disorder metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """

    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1.0 - labels], dim=2), dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return mcc(outputs, labels)


def metric_dis_fpr(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns false positive rate disorder metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1.0 - labels], dim=2), dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return fpr(outputs, labels)


def metric_rsa(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns relative surface accesibility metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 5].unsqueeze(2)[mask == 1]
    outputs = outputs[mask == 1]

    return pcc(outputs, labels)


def metric_asa(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns absolute surface accesibility metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    outputs = (outputs * labels[:, :, 17].unsqueeze(2))[mask == 1]
    labels = labels[:, :, 3].unsqueeze(2)[mask == 1]

    return pcc(outputs, labels)


def metric_phi(outputs: torch.tensor, labels: torch.tensor) -> float:
    """ Returns phi angle metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 15]
    mask = mask * (labels != 360).int()
    labels = labels[mask == 1]

    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]

    return mae(outputs, labels)


def metric_psi(outputs, labels) -> float:
    """ Returns psi angle metric
    Args:
        outputs: tensor with predicted values
        labels: tensor with correct values
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 16]
    mask = mask * (labels != 360).int()
    labels = labels[mask == 1]

    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]

    return mae(outputs, labels)
