
import numpy as np

from .. import objectives, data
import netsurfp2_dev.data
from .CNN_bLSTM import CNNbLSTM

class bLSTM(CNNbLSTM):

    def build(self, data, outputs, lstm_dim, lstm_dropout, lstm_levels,
              dense_dim, dense_dropout, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        # from keras.layers import Input, Masking, Bidirectional, LSTM, Dropout
        # from keras.layers import Dense, TimeDistributed
        import keras.layers as kl

        n_features = data['x'].shape[-1]

        l_in = kl.Input(batch_shape=(None, None, n_features), name='x')
        # l_mk = kl.Masking(mask_value=0.)(l_in)
        l_mk = l_in

        lstm_cmmn = dict(
            # dropout=lstm_dropout,
            # recurrent_dropout=lstm_dropout,
            return_sequences=True,
            stateful=False,
            # implementation=2,
        )

        l_bi = l_mk
        for _ in range(lstm_levels):
            # l_bi = Bidirectional(LSTM(lstm_dim, **lstm_cmmn))(l_bi)
            l_bi = kl.Dropout(lstm_dropout, noise_shape=[None, 1, None])(l_bi)
            l_bi = kl.Bidirectional(kl.CuDNNLSTM(lstm_dim, **lstm_cmmn))(l_bi)

        l_dp = kl.Dropout(dense_dropout)(l_bi)
        l_de = kl.TimeDistributed(kl.Dense(dense_dim, activation='relu'))(l_dp)
        l_pn = kl.Dropout(dense_dropout)(l_de)

        loss = {}
        loss_weights = {}
        l_outputs = []

        for name, w, ltype in outputs:
            yname = 'y_' + name
            n_out = data[yname].shape[-1] - 1

            if ltype == 'reg':
                loss[yname] = objectives.get_mse(masked='remove', n_out=n_out)
                dense_layer = kl.Dense(n_out, activation='sigmoid')
            elif ltype == 'reg_tanh':
                loss[yname] = objectives.get_mse(masked='apply', n_out=n_out)
                dense_layer = kl.Dense(n_out, activation='tanh')
            elif ltype == 'clf':
                loss[yname] = objectives.get_categorical_crossentropy(masked='remove')
                dense_layer = kl.Dense(n_out, activation='softmax')
            else:
                raise Exception(f'Unknown loss type: {ltype}')

            loss_weights[yname] = w

            l_outputs.append(kl.TimeDistributed(dense_layer, name=yname)(l_pn))

        import keras.models
        self.model = keras.models.Model(l_in, l_outputs)
        self.model.compile('adam', loss=loss, loss_weights=loss_weights)
        self.model.summary()

        return data


make_model = bLSTM