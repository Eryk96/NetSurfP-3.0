import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsp3.base import ModelBase
from nsp3.utils import setup_logger
from nps3.embeddings import ESM1bEmbedding


log = setup_logger(__name__)


class CNNbLSTM(ModelBase):
    def __init__(self, init_n_channels, out_channels, cnn_layers, kernel_size, padding, n_hidden, dropout, lstm_layers):
        """ Initializes the model with the required layers
        Args:
            init_n_channels [int]: size of the incoming feature vector
            out_channels [int]: amount of hidden neurons in the bidirectional lstm
            cnn_layers [int]: amount of cnn layers
            kernel_size [tuple]: kernel sizes of the cnn layers
            padding [tuple]: padding of the cnn layers
            n_hidden [int]: amount of hidden neurons
            dropout [float]: amount of dropout
            lstm_layers [int]: amount of bidirectional lstm layers
        """
        super(CNNbLSTM, self).__init__()

        self.embedding = ESM1bEmbedding("../../models/esm1b_t33_650M_UR50S.pt")

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=3),
        ])     
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden*2, out_features=2),
            nn.Tanh()
        ])

        log.info(f'<init>: \n{self}')
        
    def forward(self, x, mask):
        max_length = x.size(1)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]
