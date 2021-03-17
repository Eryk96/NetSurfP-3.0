import torch

def get_mask(labels, disorder_mask=False, unknown_mask=False):
    evaluation_mask = labels[:, :, 2]
    zero_mask = labels[:, :, 0] * evaluation_mask
    disorder_mask = labels[:, :, 1]
    unknown_mask = labels[:, :, -1]

    if disorder_mask:
        zero_mask = zero_mask * disorder_mask
    if unknown_mask:
        zero_mask = zero_mask * unknown_mask

    return zero_mask

def dihedral_to_radians(angle):
    """ Converts angles to radians
    Args:
        angles (1D Tensor): vector with angle values
    """
    return angle*np.pi/180
    
def arctan_dihedral(sin, cos):
    """ Converts sin and cos back to diheral angles
    Args:
        sin (1D Tensor): vector with sin values 
        cos (1D Tensor): vector with cos values
    """
    result = torch.where(cos >= 0, torch.arctan(sin/cos), torch.arctan(sin/cos)+np.pi)
    result = torch.where((sin <= 0) & (cos <= 0), result-np.pi*2, result)
    
    return result*180/np.pi


def fpr(pred, labels):
    """ False positive rate
    Args:
        inputs (1D Tensor): vector with predicted binary numeric values
        labels (1D Tensor): vector with correct binary numeric values
    """
    fp = sum((pred == 1) & (labels == 0))
    tn = sum((pred == 0) & (labels == 0))
    
    return (fp/(fp+tn)).item()


def fnr(pred, labels):
    """ False negative rate
    Args:
        inputs (1D Tensor): vector with predicted binary numeric values
        labels (1D Tensor): vector with correct binary numeric values
    """
    fn = sum((pred == 0) & (labels == 1))
    tp = sum((pred == 1) & (labels == 1))
    
    return (fn/(fn+tp)).item()


def mcc(pred, labels):
    """ Mathews correlation coefficient
    Args:
        inputs (1D Tensor): vector with predicted binary numeric values
        labels (1D Tensor): vector with correct binary numeric values
    """
    fp = sum((pred == 1) & (labels == 0))
    tp = sum((pred == 1) & (labels == 1))
    fn = sum((pred == 0) & (labels == 1))
    tn = sum((pred == 0) & (labels == 0))
    
    return ((tp*tn-fp*fn)/torch.sqrt(((tp+fp)*(fn+tn)*(tp+fn)*(fp+tn)).float())).item()


def pcc(pred, labels):
    """ Pearson correlation coefficient
    Args:
        inputs (1D Tensor): vector with predicted numeric values
        labels (1D Tensor): vector with correct numeric values
    """
    x = pred - torch.mean(pred)
    y = labels - torch.mean(labels)
    
    return (torch.sum(x * y) / (torch.sqrt(torch.sum(x ** 2)) * torch.sqrt(torch.sum(y ** 2)))).item()


def mae(pred, labels):
    """ Mean absolute error
    Args:
        inputs (1D Tensor): vector with predicted numeric values
        labels (1D Tensor): vector with correct numeric values
    """
    err = torch.abs(labels - pred)
    return torch.mean(torch.fmin(err, 360-err)).item()


def accuracy(pred, labels):
    """ Accuracy coefficient
    Args:
        inputs (1D Tensor): vector with predicted integer values
        labels (1D Tensor): vector with correct integer values
    """
    
    return (sum((pred == labels)) / len(labels)).item()

def metric_ss8(outputs, labels, mask):
    mask = get_mask(labels)

    labels = torch.argmax(labels[:, :, 7:15], dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]
        
    return accuracy(outputs, labels)
    

def metric_ss3(outputs, labels, mask):
    mask = get_mask(labels)

    structure_mask = torch.tensor([0,0,0,1,1,2,2,2])

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]
        
    return accuracy(outputs, labels)


def metric_dis_mcc(outputs, labels):
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1-labels], dim=2), dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return mcc(outputs, labels)


def metric_dis_fpr(outputs, labels):
    mask = get_mask(labels)

    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1-labels], dim=2), dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return fpr(outputs, labels)


def metric_rsa(outputs, labels):
    mask = get_mask(labels, disorder_mask=True, unknown_mask=True)

    labels = labels[:, :, 5].unsqueeze(2)[mask == 1]
    outputs = outputs[mask == 1]
        
    return pcc(outputs, labels)


def metric_asa(outputs, labels):
    mask = get_mask(labels, disorder_mask=True, unknown_mask=True)

    outputs = (outputs * labels[:, :, 17].unsqueeze(2))[mask == 1]
    labels = labels[:, :, 3].unsqueeze(2)[mask == 1]
    
    return pcc(outputs, labels)


def metric_phi(outputs, labels):    
    mask = get_mask(labels, disorder_mask=True, unknown_mask=True)

    labels = labels[:, :, 15]
    mask = mask * (labels != 360).int()
    labels = labels[mask == 1]
    
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]
    
    return mae(outputs, labels)
    

def metric_psi(outputs, labels):
    mask = get_mask(labels, disorder_mask=True, unknown_mask=True)

    labels = labels[:, :, 16]
    mask = mask * (labels != 360).int()
    labels = labels[mask == 1]
    
    outputs = arctan_dihedral(outputs[:, :, 0], outputs[:, :, 1])[mask == 1]
    
    return mae(outputs, labels)