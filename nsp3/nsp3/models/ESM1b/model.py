import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsp3.base import ModelBase
from nsp3.utils import setup_logger
from nsp3.embeddings import ESM1bEmbedding


log = setup_logger(__name__)


class ESM1b(ModelBase):
    def __init__(self, in_features: int, language_model: str, **kwargs):
        """ Constructor
        Args:
            in_features: size of the embedding features
            language_model: path to the language model weights
        """
        super(ESM1b, self).__init__()

        self.embedding = ESM1bEmbedding(language_model, **kwargs)

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
        """ Returns the parameters to learn """

        log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                log.info("\t" + name)
                yield param

    def forward(self, x: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """ Forwarding logic """

        x = self.embedding(x, max(mask))

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, dis, rsa, phi, psi]