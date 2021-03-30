import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsp3.base import ModelBase
from nsp3.utils import setup_logger
from nsp3.embeddings import ESM1bEmbedding


log = setup_logger(__name__)


class ESM1b(ModelBase):
<<<<<<< HEAD
    def __init__(self, in_features: int, language_model: str, **kwargs):
=======
    def __init__(self, in_features: int, language_model: str, feature_extract: bool = True):
>>>>>>> 13c3a6a9dda1c2d9bf175cbfd44b7f126cdec174
        """ Initializes the model
        Args:
            in_features: size of the embedding features
            language_model: path to the language model weights
            feature_extracting: finetune or do feature extraction from language model
        """
        super(ESM1b, self).__init__()

<<<<<<< HEAD
        self.embedding = ESM1bEmbedding(language_model, **kwargs)
=======
        self.feature_extract = feature_extract
        self.embedding = ESM1bEmbedding(language_model, self.feature_extract)
>>>>>>> 13c3a6a9dda1c2d9bf175cbfd44b7f126cdec174

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=2),
            nn.Tanh()
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True) -> list:
        print("Params to learn:")
        if self.feature_extract:
            for name, param in self.named_parameters(recurse=recurse):
                if param.requires_grad == True:
                    print("\t",name)
                    yield param
        else:
            for name, param in self.named_parameters(recurse=recurse):
                if param.requires_grad == True:
                    print("\t",name)
                    yield param

    def forward(self, x, mask):
        x = self.embedding(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, dis, rsa, phi, psi]
