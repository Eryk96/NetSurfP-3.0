
import numpy as np

from .. import objectives, data
import netsurfp.data

class bLSTM:

    def build(self, data, outputs, lstm_dim, lstm_dropout, lstm_levels,
              dense_dim, dense_dropout, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        from keras.layers import Input, Masking, Bidirectional, LSTM, Dropout
        from keras.layers import Dense, TimeDistributed

        n_features = data['x'].shape[-1]

        l_in = Input(batch_shape=(None, None, n_features), name='x')
        l_mk = Masking(mask_value=0.)(l_in)

        lstm_cmmn = dict(
            dropout=lstm_dropout,
            recurrent_dropout=lstm_dropout,
            return_sequences=True,
            stateful=False,
            implementation=2)

        l_bi = l_mk
        for _ in range(lstm_levels):
            l_bi = Bidirectional(LSTM(lstm_dim, **lstm_cmmn))(l_bi)

        l_dp = Dropout(dense_dropout)(l_bi)
        l_de = TimeDistributed(Dense(dense_dim, activation='relu'))(l_dp)
        l_pn = Dropout(dense_dropout)(l_de)

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

            n_out = data[yname].shape[-1] - 1
            dense_layer = Dense(n_out, activation='sigmoid')
            l_outputs.append(TimeDistributed(dense_layer, name=yname)(l_pn))

        import keras.models
        self.model = keras.models.Model(l_in, l_outputs)
        self.model.compile('adam', loss=loss, loss_weights=loss_weights)
        self.model.summary()

        return data

    def predict(self, data_full):
        return self.model.predict(data_full)

    def fit(self, data_trn, data_tst, metrics, logger=None, save=False):
        epochs = self.epochs
        bs = self.batch_size

        import tqdm
        import time
        import tempfile

        metrics_max = {}
        metrics_min = {}
        best_loss_age = 0

        for epoch in tqdm.trange(1, epochs + 1, ascii=True):
            start = time.time()
            epoch_loss = {k: [] for k in self.model.metrics_names}
            all_batches = list(netsurfp.data.batch_generator(data_trn, bs, None))

            if epoch == 1:
                all_batches.sort(key=lambda b: b[0]['x'].shape[1], reverse=True)

            for batch_data, rst in tqdm.tqdm(all_batches, leave=False, ascii=True):
                batch_out = self.model.train_on_batch(batch_data, batch_data)

                if len(self.model.metrics_names) == 1:
                    batch_out = [batch_out]

                for v, k in zip(batch_out, self.model.metrics_names):
                    epoch_loss[k].append(v)

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

            #
            # Early stopping (sorta)
            #

            if logger:
                logger('epoch.time', time.time() - start, epoch)

            if best_loss_age >= 50:
                break

        if save and os.path.isfile(save):
            self.model.load_weights(save)

        res = []
        for k, v in sorted(metrics_min.items()):
            res.append('{}={:.4f}<{:.4f}'.format(k, v, metrics_max[k]))
        res.append('best_loss_age={}'.format(best_loss_age))

        return '\n'.join(res)

make_model = bLSTM