
import os

import numpy as np

from .. import objectives
import netsurfp.data

from .CNN_bLSTM import CNNbLSTM


class ResNet(CNNbLSTM):
    
    def build(self, data_full, outputs,
              filter_sizes, layers, cnn_drop, cnn_channels,
              epochs, batch_size,
              pretrained_weights=None):

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_outputs = len(outputs)

        import keras
        from keras.layers import Input, Dropout, Dense, TimeDistributed
        from keras.layers import Conv1D, Concatenate, Activation, Add, BatchNormalization

        import sys
        sys.setrecursionlimit(2147483647)

        n_features = data_full['x'].shape[-1]

        l_in = Input(batch_shape=(None, None, n_features), name='x')

        def convmodule(i, incoming):
            convs = []
            for fs in filter_sizes:
                fs = 2**fs + 1 #filter size
                #Add dropout to the conv layer
                l_dp0 = Dropout(cnn_drop, name='drop_l{}_cn{}'.format(i, fs))(incoming)
                #Convolution layer
                convs.append(Conv1D(cnn_channels, fs,
                                    activation='linear',
                                    padding='same',
                                    name='l{}_cn{}'.format(i, fs))(l_dp0)) #yapf: disable
            return Concatenate(axis=-1)(convs)

        l_cnn = convmodule(0, l_in)

        for i in range(0, layers * 2, 2):
            l_cnn_1 = convmodule(i+1, l_cnn)
            l_cnn_1 = Activation('relu')(l_cnn_1)
            l_cnn_2 = convmodule(i+2, l_cnn_1)
            l_cnn = Add()([l_cnn_1, l_cnn_2])
            l_cnn = Activation('relu')(l_cnn)
            l_cnn = BatchNormalization()(l_cnn)

        l_pn = l_cnn

        loss = {}
        loss_weights = {}
        l_outputs = []

        for name, w, ltype in outputs:
            yname = 'y_' + name
            if ltype == 'reg':
                loss[yname] = objectives.get_mse(masked='remove')
            elif ltype == 'clf':
                loss[yname] = objectives.get_categorical_crossentropy(masked='remove')

            loss_weights[yname] = w

            n_out = data_full[yname].shape[-1] - 1
            dense_layer = Dense(n_out, activation='sigmoid')
            l_outputs.append(TimeDistributed(dense_layer, name=yname)(l_pn))

        import keras.models
        self.model = keras.models.Model(l_in, l_outputs)

        self.model.compile('adam', loss=loss, loss_weights=loss_weights) #WHAT!
        self.model.summary()

make_model = ResNet
