
import os

import numpy as np

from .. import objectives
import netsurfp2_dev.data

class CNNbLSTM:

    def build(self, data_full, outputs,
              cnn_min, cnn_max, cnn_l2, cnn_drop, cnn_passthrough,
              lstm_dim, lstm_dropout, lstm_levels, lstm_l2,
              dense_dim, dense_dropout, epochs, batch_size,
              cnn_filters=32, pretrained_weights=None):

        self.epochs = epochs
        self.batch_size = batch_size
        self.n_outputs = len(outputs)

        import keras
        from keras.layers import Input, Masking, Bidirectional, LSTM, Dropout
        from keras.layers import Dense, TimeDistributed
        from keras.layers import Concatenate, Multiply, Conv1D

        n_features = data_full['x'].shape[-1]

        l_in = Input(batch_shape=(None, None, n_features), name='x')
        l_mask = Input(batch_shape=(None, None, 1), name='x_mask')
        
        conv_reg = keras.regularizers.l2(cnn_l2)

        def conv_layer(incoming):
            convs = []
            for i in range(cnn_min, cnn_max + 1):
                fs = 2**i + 1  #filter size
                #Add dropout to the conv layer
                l_dp0 = Dropout(cnn_drop, name='drop_cn{}'.format(fs))(incoming)
                #Convolution layer
                convs.append(Conv1D(cnn_filters, fs,
                                    activation='relu',
                                    padding='same',
                                    kernel_regularizer=conv_reg,
                                    name='cn{}'.format(fs))(l_dp0)) #yapf: disable
            p = []
            if cnn_passthrough:
                p = [incoming]
            return Concatenate(axis=-1)(convs + p)

        l_conv1 = conv_layer(l_in)
        l_conv1 = keras.layers.BatchNormalization()(l_conv1)

        mask_size = l_conv1._keras_shape[-1]
        l_pm = keras.layers.Concatenate()([l_mask] * mask_size)
        l_mn = keras.layers.Multiply()([l_conv1, l_pm])

        l_mk = keras.layers.Masking(mask_value=0.)(l_mn)

        rnn_reg = keras.regularizers.l2(lstm_l2)

        lstm_cmmn = dict(
            dropout=lstm_dropout,
            recurrent_dropout=lstm_dropout,
            return_sequences=True,
            stateful=False,
            kernel_regularizer=rnn_reg,
            recurrent_regularizer=rnn_reg,
            implementation=2)

        l_bi = l_mk
        for _ in range(lstm_levels):
            l_bi = Bidirectional(LSTM(lstm_dim, **lstm_cmmn))(l_bi)

        l_dn = l_bi
        if dense_dim:
            l_dn = keras.layers.Dropout(dense_dropout)(l_dn)
            l_dn = TimeDistributed(Dense(dense_dim, activation='relu'))(l_dn)
        
        l_pn = keras.layers.Dropout(dense_dropout)(l_dn)

        loss = {}
        loss_weights = {}
        l_outputs = []

        for name, w, ltype in outputs:
            yname = 'y_' + name
            n_out = data_full[yname].shape[-1] - 1
            
            if ltype == 'reg':
                loss[yname] = objectives.get_mse(masked='apply', n_out=n_out)
                dense_layer = Dense(n_out, activation='sigmoid')
            elif ltype == 'reg_tanh':
                loss[yname] = objectives.get_mse(masked='apply', n_out=n_out)
                dense_layer = Dense(n_out, activation='tanh')
            elif ltype == 'clf':
                loss[yname] = objectives.get_categorical_crossentropy(masked='apply')
                dense_layer = Dense(n_out, activation='softmax')

            l_outputs.append(TimeDistributed(dense_layer, name=yname)(l_pn))
            loss_weights[yname] = w

        import keras.models
        self.model = keras.models.Model([l_in, l_mask], l_outputs)

        if pretrained_weights:
            self.model.load_weights(pretrained_weights, by_name=True)

        self.model.compile('adam', loss=loss, loss_weights=loss_weights)
        self.model.summary()

        return data_full

    def predict(self, data_full):
        out = self.model.predict(data_full)
        if self.n_outputs > 1:
            out_ = []
            for p in out:
                if len(p.shape) == 2:
                    p = p[:, :, np.newaxis]
                print(p.shape)
                out_.append(p)
            out = np.concatenate(out_, axis=2)

        return out

    def fit(self, data_trn, data_tst, metrics, logger=None, save=False):
        epochs = self.epochs
        bs = self.batch_size

        import tqdm
        import time
        import tempfile

        metrics_max = {}
        metrics_min = {}
        best_loss_age = 0

        epoch_prog = tqdm.trange(1, epochs + 1, ascii=True)
        for epoch in epoch_prog:
            start = time.time()
            epoch_loss = {k: [] for k in self.model.metrics_names}
            all_batches = list(netsurfp2_dev.data.batch_generator(data_trn, bs, None))

            if epoch == 1:
                all_batches.sort(key=lambda b: b[0]['x'].shape[1], reverse=True)

            for batch_data, rst in tqdm.tqdm(all_batches, leave=False, ascii=True):
                batch_out = self.model.train_on_batch(batch_data, batch_data)

                if len(self.model.metrics_names) == 1:
                    batch_out = [batch_out]

                for v, k in zip(batch_out, self.model.metrics_names):
                    epoch_loss[k].append(v)

                # STATUS_FILE = '/home/mskl/PhD/BI000-NetSurfP-2.0/02_Models/status.txt'
                # while True:
                #     with open(STATUS_FILE) as sf:
                #         if sf.read().strip() == 'running':
                #             break
                #     time.sleep(5)

            for key, val in epoch_loss.items():
                m = float(np.mean(val))
                if np.isnan(m):
                    m = 0.
                fkey = 'train.{}'.format(key)
                if logger:
                    logger(fkey, m, epoch)
                if fkey not in metrics_max:
                    metrics_max[fkey] = m
                    metrics_min[fkey] = m
                else:
                    metrics_max[fkey] = max([m, metrics_max[fkey]])
                    metrics_min[fkey] = min([m, metrics_min[fkey]])

            #
            # Evaluate loss
            #

            val_out = self.model.evaluate(data_tst, data_tst, batch_size=bs, verbose=0)
            if len(self.model.metrics_names) == 1:
                val_out = [val_out]

            for k, v in zip(self.model.metrics_names, val_out):
                fkey = 'test.{}'.format(k)
                if logger:
                    logger(fkey, v, epoch)
                if fkey not in metrics_max:
                    metrics_max[fkey] = v
                    metrics_min[fkey] = v
                else:
                    if fkey == 'test.loss':
                        if np.isnan(v):
                            raise Exception("NaN'ed after {} epochs".format(epoch))
                        if v < metrics_min[fkey]:
                            best_loss_age = 0
                            if save:
                                self.model.save(save)
                        else:
                            best_loss_age += 1
                    metrics_max[fkey] = max([v, metrics_max[fkey]])
                    metrics_min[fkey] = min([v, metrics_min[fkey]])

            #
            # Evaluate metrics
            #

            postfix = {'age': best_loss_age}

            #for data_, mode_ in [(data_trn, 'train.'), (data_tst, 'test.')]:
            for data_, mode_ in [(data_tst, 'test.')]:
                out_ = self.model.predict(data_, batch_size=bs)
                if len(self.model.output_names) == 1:
                    out_ = [out_]

                for outname, y_pred in zip(self.model.output_names, out_):
                    y_true = data_[outname]
                    for func in metrics[outname]:
                        v = func(y_true, y_pred)
                        fkey = mode_ + outname + '_' + func.__name__
                        if logger:
                            logger(fkey, v, epoch)
                        if fkey not in metrics_max:
                            metrics_max[fkey] = v
                            metrics_min[fkey] = v
                        else:
                            metrics_max[fkey] = max([v, metrics_max[fkey]])
                            metrics_min[fkey] = min([v, metrics_min[fkey]])

                        postfix[fkey] = metrics_max[fkey]

            #
            # Early stopping (sorta)
            #

            if logger:
                logger('epoch.time', time.time() - start, epoch)

            epoch_prog.set_postfix(postfix)

            if best_loss_age >= 50:
                break

        if save and os.path.isfile(save):
            try:
                self.model.load_weights(save)
            except Exception:
                print('FAILED SAVING MODEL')

        res = []
        for k, v in sorted(metrics_min.items()):
            res.append('{}={:.4f}<{:.4f}'.format(k, v, metrics_max[k]))
        res.append('best_loss_age={}'.format(best_loss_age))

        return '\n'.join(res)

make_model = CNNbLSTM
