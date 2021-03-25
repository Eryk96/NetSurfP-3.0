import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsp3.base import ModelBase
from nsp3.utils import setup_logger


log = setup_logger(__name__)


class CNNbLSTM(ModelBase):
    def __init__(self, init_n_channels, out_channels, cnn_layers, kernel_size, padding, n_hidden, dropout, lstm_layers):
        """ Initialization of the CNNbLSTM model
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

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels, kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels+(out_channels*2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels+(out_channels*2), hidden_size=n_hidden, batch_first=True, \
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

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

        # calculate the residuals
        x = x.permute(0,2,1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0,2,1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        dis = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, dis, rsa, phi, psi]
