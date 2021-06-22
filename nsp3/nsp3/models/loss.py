import torch
import torch.nn as nn

from nsp3.models.metric import dihedral_to_radians, arctan_dihedral, get_mask


def mse(outputs: torch.tensor, labels: torch.tensor, mask: torch.tensor) -> torch.tensor:
    """ Returns mean squared loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """
    loss = torch.square(outputs - labels) * mask
    return torch.sum(loss) / torch.sum(mask)


def cross_entropy(outputs: torch.tensor, labels: torch.tensor, mask: torch.tensor, weights: torch.tensor = None) -> torch.tensor:
    """ Returns cross entropy loss using masking
    Args:
        outputs: tensor with predictions
        labels: tensor with labels
        mask: tensor with masking
    """

    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)*mask
    return torch.sum(loss) / torch.sum(mask)


def ss8(outputs: torch.tensor, labels: torch.tensor, weights: torch.tensor = None) -> torch.tensor:
    """ Returns SS8 loss
    Args:
        outputs: tensor with SS8 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 7:15], dim=2)
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def ss3(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns SS3 loss
    Args:
        outputs: tensor with SS3 predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    # convert ss8 to ss3 class
    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(labels.device)
    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def disorder(outputs: torch.tensor, labels: torch.tensor, weights: torch.tensor = None) -> torch.tensor:
    """ Returns disorder loss
    Args:
        outputs: tensor with disorder predictions
        labels: tensor with labels
    """
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1.0 - labels], dim=2), dim=2)

    outputs = outputs.permute(0, 2, 1)

    return cross_entropy(outputs, labels, mask)


def rsa(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 5].unsqueeze(2)

    return mse(outputs, labels, mask)

def rsa_iso(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 5].unsqueeze(2)

    return mse(outputs, labels, mask)

def rsa_cpx(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns relative surface accesibility loss
    Args:
        outputs: tensor with rsa predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 6].unsqueeze(2)

    return mse(outputs, labels, mask)


def phi(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns phi loss
    Args:
        outputs: tensor with phi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 15].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(
        labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def psi(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns psi loss
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 16].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(
        labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def multi_task_loss(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted multi task loss. 
        Combines ss8, ss3, disorder, rsa, phi and psi loss.
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _ss3 = ss3(outputs[1], labels) * 5
    _dis = disorder(outputs[2], labels) * 5
    _rsa = rsa(outputs[3], labels) * 100
    _phi = phi(outputs[4], labels) * 5
    _psi = psi(outputs[5], labels) * 5

    loss = torch.stack([_ss8, _ss3, _dis, _rsa, _phi, _psi])

    return loss.sum()


def multi_task_extended(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted multi task loss. 
        Combines ss8, ss3, disorder, rsa_iso, rsa_cpx, phi and psi loss.
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _dis = disorder(outputs[1], labels) * 5
    _rsa_iso = rsa_iso(outputs[2], labels) * 100
    _rsa_cpx = rsa_cpx(outputs[3], labels) * 100
    _phi = phi(outputs[4], labels) * 5
    _psi = psi(outputs[5], labels) * 5

    loss = torch.stack([_ss8, _dis, _rsa_iso, _rsa_cpx, _phi, _psi])

    return loss.sum()


def secondary_structure_loss(outputs: torch.tensor, labels: torch.tensor) -> torch.tensor:
    """ Returns a weighted double task loss for secondary structure. 
    Args:
        outputs: tensor with psi predictions
        labels: tensor with labels
    """
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _ss3 = ss3(outputs[1], labels) * 5

    loss = torch.stack([_ss8, _ss3])

    return loss.sum()
