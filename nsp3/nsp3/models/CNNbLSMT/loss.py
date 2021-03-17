import torch
import torch.nn as nn

from nsp3.models.CNNbLSMT import dihedral_to_radians, arctan_dihedral

def mse(self, outputs, labels, mask):
    loss = torch.square(outputs - labels) * mask
    return torch.sum(loss) / torch.sum(mask)
    

def cross_entropy(self, outputs, labels, mask):
    labels = labels.clone()
    labels[mask == 0] = -999
        
    return nn.CrossEntropyLoss(ignore_index=-999)(outputs, labels.long())


def ss8(self, outputs, labels, mask):
    labels = torch.argmax(labels[:, :, 7:15], dim=2)
    outputs = outputs[0].permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def ss3(self, outputs, labels, mask):
    structure_mask = torch.tensor([0,0,0,1,1,2,2,2]).to(device)

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
    outputs = outputs[1].permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def disorder(self, outputs, labels, mask):
    # apply the disorder loss
    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1-labels], dim=2), dim=2)
        
    outputs = outputs[2].permute(0, 2, 1)
        
    return cross_entropy(outputs, labels, mask)


def rsa(self, outputs, labels, mask):
    labels = labels[:, :, 5].unsqueeze(2)
    outputs = outputs[3]
        
    mask = mask.unsqueeze(2)
        
    return mse(outputs, labels, mask)


def phi(self, outputs, labels, mask):
    labels = labels[:, :, 15].unsqueeze(2)
    outputs = outputs[4]
        
    mask = mask * (labels != 360).squeeze(2).int()
    mask = torch.cat(2*[mask.unsqueeze(2)], dim=2)
        
    loss = mse(outputs.squeeze(2), torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))), dim=2).squeeze(2), mask)
    return loss


def psi(self, outputs, labels, mask):
    labels = labels[:, :, 16].unsqueeze(2)
    outputs = outputs[5]
        
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
    _ss8 = ss8(outputs, labels, zero_mask) * 1
    _ss3 = ss3(outputs, labels, zero_mask) * 5
    _dis = disorder(outputs, labels, zero_mask) * 5
    _rsa = rsa(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 100
    _phi = phi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5
    _psi = psi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5
        
    loss = torch.stack([_ss8, _ss3, _dis, _rsa, _phi, _psi])
        
    return loss.sum()

