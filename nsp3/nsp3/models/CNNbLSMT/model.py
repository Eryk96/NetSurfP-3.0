import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsp3.base import ModelBase
from nsp3.utils import setup_logger

from nsp3.embeddings import ESM1bEmbedding

log = setup_logger(__name__)


class CNNbLSTM(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int):
        """ Baseline model for CNNbLSTM
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super(CNNbLSTM, self).__init__()

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])

        log.info(f'<init>: \n{self}')

    def forward(self, x, mask) -> list:
        """ Forwarding logic """

        max_length = x.size(1)
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
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


class CNNbLSTM_ESM1b_SecondaryStructure(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, language_model, **kwargs):
        """ Baseline model with ESM1b and only secondary structure predictions
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super(CNNbLSTM_ESM1b_SecondaryStructure, self).__init__()

        self.embedding = ESM1bEmbedding(language_model, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
        ])

        log.info(f'<init>: \n{self}')

    def parameters(self, recurse: bool = True) -> list:
        """ Returns the parameters to learn """

        log.info("Params to learn:")
        for name, param in self.named_parameters(recurse=recurse):
            if param.requires_grad == True:
                log.info("\t" + name)
                yield param

    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """

        max_length = x.size(1)

        x = self.embedding(x, max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)

        return [ss8, ss3]


class CNNbLSTM_ESM1b_Complete(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ Constructor
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
            language_model: path to language model weights
        """
        super(CNNbLSTM_ESM1b_Complete, self).__init__()

        # ESM1b block
        self.embedding = ESM1bEmbedding(embedding_args, embedding_pretrained, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
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

    def forward(self, x: torch.tensor, mask: torch.tensor) -> list:
        """ Forwarding logic """
        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
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


class CNNbLSTM_Extended(ModelBase):
    def __init__(self, init_n_channels: int, out_channels: int, cnn_layers: int, kernel_size: tuple, padding: tuple, n_hidden: int, dropout: float, lstm_layers: int, embedding_args: dict, embedding_pretrained: str = None, **kwargs):
        """ CNNbLSTM with removed Q3 prediction which instead is predicted by remapping Q8
        Args:
            init_n_channels: size of the incoming feature vector
            out_channels: amount of hidden neurons in the bidirectional lstm
            cnn_layers: amount of cnn layers
            kernel_size: kernel sizes of the cnn layers
            padding: padding of the cnn layers
            n_hidden: amount of hidden neurons
            dropout: amount of dropout
            lstm_layers: amount of bidirectional lstm layers
        """

        super(CNNbLSTM_Extended, self).__init__()

        # ESM1b block
        self.embedding = ESM1bEmbedding(embedding_args, embedding_pretrained, **kwargs)

        # CNN blocks
        self.conv = nn.ModuleList()
        for i in range(cnn_layers):
            self.conv.append(nn.Sequential(*[
                nn.Dropout(p=dropout),
                nn.Conv1d(in_channels=init_n_channels, out_channels=out_channels,
                          kernel_size=kernel_size[i], padding=padding[i]),
                nn.ReLU(),
            ]))

        self.batch_norm = nn.BatchNorm1d(init_n_channels + (out_channels * 2))

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_n_channels + (out_channels * 2), hidden_size=n_hidden, batch_first=True,
                            num_layers=lstm_layers, bidirectional=True, dropout=dropout)
        self.lstm_dropout_layer = nn.Dropout(p=dropout)

        # Task block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
        ])
        self.rsa_iso = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.rsa_cpx = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
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

    def forward(self, x, mask) -> list:
        """ Forwarding logic """

        # remove start and end token from length
        max_length = x.size(1) - 2

        x = self.embedding(x, max(mask))
        x = x.permute(0, 2, 1)

        # concatenate channels from residuals and input + batch norm
        r = x
        for layer in self.conv:
            r = torch.cat([r, layer(x)], dim=1)

        x = self.batch_norm(r)

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)
        x = pack_padded_sequence(x, mask, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=max_length, batch_first=True)
        x = self.lstm_dropout_layer(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        dis = self.disorder(x)
        rsa_iso = self.rsa_iso(x)
        rsa_cpx = self.rsa_iso(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, dis, rsa_iso, rsa_cpx, phi, psi]