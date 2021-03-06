
"""
Definitions of models.

"""

import importlib

def get(modelname):
    return importlib.import_module('netsurfp2_dev.models.' + modelname).make_model


def get_filename(modelname):
    return importlib.import_module('netsurfp2_dev.models.' + modelname).__file__

