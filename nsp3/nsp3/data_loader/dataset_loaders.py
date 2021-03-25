from nsp3.base import DatasetBase

class NSPData(DatasetBase):
    """ NetsurfP 2.0 dataset .npz file """
    def __init__(self, *args, **kwargs):
        super(NSPData, self).__init__(*args, **kwargs)