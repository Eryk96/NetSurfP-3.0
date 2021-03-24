
import numpy as np

from .. import objectives, data
import netsurfp2_dev.data

class bLSTM:

    def build(self, data, outputs, lstm_dim, lstm_dropout, lstm_levels,
              dense_dim, dense_dropout, epochs, batch_size):

        self.epochs = epochs
        self.batch_size = batch_size

        import keras.layers as kl
        import keras.backend as K

        self.n_features = data['x'].shape[-1]

        netG_input = kl.Input(batch_shape=(None, None, self.n_features), name='x')

        lstm_cmmn = dict(
            # dropout=lstm_dropout,
            # recurrent_dropout=lstm_dropout,
            return_sequences=True,
            stateful=False,
            # implementation=2
        )

        l_bi = netG_input
        for _ in range(lstm_levels):
            l_bi = kl.Dropout(lstm_dropout, noise_shape=[None, 1, None])(l_bi)
            l_bi = kl.Bidirectional(kl.LSTM(lstm_dim, **lstm_cmmn))(l_bi)

        l_dp = kl.Dropout(dense_dropout)(l_bi)
        l_de = kl.TimeDistributed(kl.Dense(dense_dim, activation='relu'))(l_dp)
        l_pn = kl.Dropout(dense_dropout)(l_de)

        loss = {}
        loss_weights = {}
        l_outputs = []
        # self.n_outputs = {}
        n_out_total = 0

        self.netD_input_names = []

        for name, w, ltype in outputs:
            yname = 'y_' + name
            n_out = data[yname].shape[-1] - 1
            # self.n_outputs[yname] = n_out
            n_out_total += n_out

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

            # netD_inputs_names.append(kl.Input(shape=(None, n_out), name=f'Dr_{name}'))
            self.netD_input_names.append([yname, n_out])

        cat_output = [kl.Concatenate()(l_outputs)]

        import keras.models
        self.model = keras.models.Model(netG_input, l_outputs + cat_output, name='generator')
        self.model.summary()
        self.model.compile('sgd', loss=loss, loss_weights=loss_weights)
        print('\n' * 2)

        #
        # -- Discriminator
        #

        # netD_inputs = [kl.Input(shape=(None, n), name=f'D_{name}') for name, n in netD_input_names]
        netD_input = kl.Input(shape=(None, n_out_total))

        # l_bi = kl.Concatenate()(netD_inputs)
        # l_bi = netD_input
        # for _ in range(lstm_levels):
            # l_bi = kl.Dropout(lstm_dropout, noise_shape=[None, 1, None])(l_bi)
            # l_bi = kl.Bidirectional(kl.LSTM(lstm_dim, **lstm_cmmn))(l_bi)

        filter_sizes = (3, 5, 13)
        layers = 5
        cnn_drop = 0.5
        cnn_channels = 32

        def convmodule(i, incoming):
            convs = []
            for fs in filter_sizes:
                #fs = 2**fs + 1 #filter size
                #Add dropout to the conv layer
                l_dp0 = kl.Dropout(cnn_drop, name='drop_l{}_cn{}'.format(i, fs))(incoming)
                #Convolution layer
                convs.append(kl.Conv1D(cnn_channels, fs,
                                    activation='linear',
                                    padding='same',
                                    name='l{}_cn{}'.format(i, fs))(l_dp0)) #yapf: disable
            return kl.Concatenate(axis=-1)(convs)

        l_cnn = convmodule(0, netD_input)

        for i in range(0, layers * 2, 2):
            l_cnn_1 = convmodule(i+1, l_cnn)
            l_cnn_1 = kl.Activation('elu')(l_cnn_1)
            l_cnn_2 = convmodule(i+2, l_cnn_1)
            l_cnn = kl.Add()([l_cnn_1, l_cnn_2])
            l_cnn = kl.Activation('elu')(l_cnn)
            l_cnn = kl.BatchNormalization()(l_cnn)

        l_dp = kl.Dropout(dense_dropout)(l_cnn)
        l_de = kl.TimeDistributed(kl.Dense(dense_dim, activation='relu'))(l_dp)
        l_pn = kl.Dropout(dense_dropout)(l_de)

        l_do = kl.TimeDistributed(kl.Dense(1, activation='sigmoid'))(l_pn)

        self.netD = keras.models.Model(netD_input, l_do, name='discriminator')
        self.netD.summary()
        print('\n' * 2)

        #
        # -- Discriminator loss
        #

        netD_real_input = kl.Input(shape=(None, n_out_total), name='netD_real_input')
        # netD_fake_input = kl.Input(shape=(None, n_out_total), name='netD_fake_input')
        netD_fake_input = self.model(netG_input)[-1]
        #
        loss_real = K.mean(self.netD(netD_real_input))
        loss_fake = K.mean(self.netD(netD_fake_input))

        ϵ_input = K.placeholder(shape=(None, 1))
        netD_mixed_input = kl.Input(shape=(None, n_out_total), name='netD_mixed_input',
            tensor=ϵ_input * netD_real_input + (1-ϵ_input) * netD_fake_input)

        grad_mixed = K.gradients(self.netD(netD_mixed_input), [netD_mixed_input])[0]
        norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1,2]))
        grad_penalty = K.mean(K.square(norm_grad_mixed -1))

        λ = 10
        d_loss = loss_fake - loss_real + λ * grad_penalty

        from keras.optimizers import Adam

        training_updates = Adam().get_updates(self.netD.trainable_weights, [], d_loss)
        self.netD_train = K.function(
            [netD_real_input, netG_input, ϵ_input],
            [loss_real, loss_fake],    
            training_updates
        )

        g_out = self.model(netG_input)
        combined_output = self.netD(g_out[-1])

        d_loss_w = kl.Input(shape=(1, ))
        g_loss = -K.mean(combined_output) * d_loss_w
        lossw_total = sum(loss_weights.values())

        y_true_placeholders = []
        y_losses = []
        for i, (yname, nout) in enumerate(self.netD_input_names):
            ph = kl.Input(shape=(None, nout + 1))
            lossfn = loss[yname]
            y_loss = lossfn(ph, g_out[i])
            y_losses.append(y_loss)
            g_loss += (loss_weights[yname] / lossw_total) * y_loss
            y_true_placeholders.append(ph)

        training_updates = Adam().get_updates(self.model.trainable_weights, [], g_loss)
        self.netG_train = K.function(
            [netG_input, d_loss_w] + y_true_placeholders,
            g_out + y_losses + [combined_output, g_loss],    
            training_updates
        )

        return data

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
            # epoch_loss = {k: [] for k in self.model.metrics_names}
            all_batches = list(netsurfp2_dev.data.batch_generator(data_trn, bs, None))

            if epoch == 1:
                all_batches.sort(key=lambda b: b[0]['x'].shape[1], reverse=True)

            d_iters = 1
            # if epoch < 10 or epoch % 10 == 9:
                # d_iters = 1

            for bi, (batch_data, rst) in enumerate(tqdm.tqdm(all_batches, leave=False, ascii=True)):
                d_weight = 0.
                if epoch >= 3:
                    d_weight = 1
                
                batch_netD_real_input = np.concatenate([batch_data[yn][:, :, :-1] for yn, _ in self.netD_input_names], axis=-1)
                bloss_real, bloss_fake = self.netD_train([
                    batch_netD_real_input,
                    batch_data['x'],
                    np.random.random(),
                ])
                
                netG_inp = [batch_data['x'], d_weight]
                for yn, _ in self.netD_input_names:
                    netG_inp.append(batch_data[yn])

                if bi % d_iters == d_iters - 1:
                    g_out = self.netG_train(netG_inp)

            #
            # Evaluate loss
            #

            val_out = self.model.evaluate(data_tst, data_tst, batch_size=bs, verbose=0)
            if len(self.model.metrics_names) == 1:
                val_out = [val_out]

            for k, v in zip(self.model.metrics_names, val_out):
                if not k.startswith('y_'):
                    continue
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
                    if not outname.startswith('y_'):
                        continue
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

        res = []
        for k, v in sorted(metrics_min.items()):
            res.append('{}={:.4f}<{:.4f}'.format(k, v, metrics_max[k]))
        res.append('best_loss_age={}'.format(best_loss_age))

        return '\n'.join(res)


make_model = bLSTM