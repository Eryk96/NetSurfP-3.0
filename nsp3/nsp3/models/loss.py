import torch
import torch.nn as nn

from nsp3.models.metric import dihedral_to_radians, arctan_dihedral, get_mask

def mse(outputs, labels, mask):
    loss = torch.square(outputs - labels) * mask
    return torch.sum(loss) / torch.sum(mask)
    

def cross_entropy(outputs, labels, mask):
    labels = labels.clone()
    labels[mask == 0] = -999
        
    return nn.CrossEntropyLoss(ignore_index=-999)(outputs, labels.long())


def ss8(outputs, labels):
    mask = get_mask(labels)
    
    labels = torch.argmax(labels[:, :, 7:15], dim=2)
    outputs = outputs.permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def ss3(outputs, labels):
    mask = get_mask(labels)
    
    structure_mask = torch.tensor([0,0,0,1,1,2,2,2]).to(labels.device)
    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
    outputs = outputs.permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def disorder(outputs, labels):
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1-labels], dim=2), dim=2)
        
    outputs = outputs.permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def rsa(outputs, labels):
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)
    mask = mask.unsqueeze(2)

    labels = labels[:, :, 5].unsqueeze(2)
        
    return mse(outputs, labels, mask)


def phi(outputs, labels):
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 15].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2*[mask.unsqueeze(2)], dim=2)
        
    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def psi(outputs, labels):
    mask = get_mask(labels, use_disorder_mask=True, use_unknown_mask=True)

    labels = labels[:, :, 16].unsqueeze(2)
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2*[mask.unsqueeze(2)], dim=2)
        
    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def multi_task_loss(outputs, labels):
    # filters
    zero_mask = labels[:, :, 0]
    disorder_mask = labels[:, :, 1]
    unknown_mask = labels[:, :, -1]
        
    # weighted losses
    _ss8 = ss8(outputs[0], labels) * 1
    _ss3 = ss3(outputs[1], labels) * 5
    _dis = disorder(outputs[2], labels) * 5
    _rsa = rsa(outputs[3], labels) * 100
    _phi = phi(outputs[4], labels) * 5
    _psi = psi(outputs[5], labels) * 5
        
    loss = torch.stack([_ss8, _ss3, _dis, _rsa, _phi, _psi])
        
    return loss.sum()

