import numpy as np

import xgboost


class XGB:
    def _preprocess_data(self, data):
        window_size = self.window_size
        wb = window_size // 2
        wa = wb - (1 - window_size % 2)

        x_data = data['x']

        #Add some 0's at either end
        padded_shape = list(x_data.shape)
        padded_shape[1] += wb + wa
        padded_data = np.zeros(padded_shape)
        padded_data[:, wb:-wa, :] = x_data

        ykeys = [dk for dk in data if dk.startswith('y_')]
        if len(ykeys) > 1:
            raise NotImplementedError('More than one target')
        ykey = ykeys[0]

        processed_data = {'x': [], 'y': [], 'map2seq': []}

        for i in range(x_data.shape[0]):
            seqlen = int(np.sum(padded_data[i, :, 50]))
            for j in range(seqlen):
                if not data[ykey][i, j, -1]:
                    continue

                processed_data['x'].append(
                    padded_data[i, j:j + window_size, :])
                processed_data['y'].append(data[ykey][i, j, :-1])
                processed_data['map2seq'].append((i, j))

        for datkey in ('x', 'y'):
            processed_data[datkey] = np.array(processed_data[datkey])

        n_samples = processed_data['x'].shape[0]
        processed_data['x'] = processed_data['x'].reshape(n_samples, -1)

        # if self.ltype == 'clf':
            # processed_data['y'] = np.argmax(processed_data['y'], axis=-1)
        # else:
            # processed_data['y'] = processed_data['y'].flatten()

        return processed_data

    def build(self, data_full, outputs, n_estimators, max_depth, subsample,
              colsample_bytree, tree_method, grow_policy, window_size):
        self.window_size = window_size
        self.n_estimators = n_estimators

        if len(outputs) > 1:
            raise NotImplementedError('Only one output supported')

        name, w, ltype = outputs[0]
        self.ykey = 'y_' + name
        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            grow_policy=grow_policy,
            n_jobs=10, )

        self.ltype = ltype
        if ltype == 'reg':
            self.model = xgboost.XGBRegressor(**kwargs)
        elif ltype == 'clf':
            self.model = xgboost.XGBClassifier(**kwargs)
        else:
            raise ValueError('Unknown modeltype: "{}"'.format(ltype))

    def fit(self, data_trn, data_tst, metrics, logger=None, save=False):
        """..."""
        data_trn_w = self._preprocess_data(data_trn)
        data_tst_w = self._preprocess_data(data_tst)

        y_true = data_tst_w['y']#[:, np.newaxis]
        y_true_mask = np.ones((y_true.shape[0], 1))
        y_true = np.concatenate([y_true, y_true_mask], axis=1)[:, np.newaxis, :]

        if self.ltype == 'clf':
            y_trn = np.argmax(data_trn_w['y'], axis=-1)
        else:
            y_trn = data_trn_w['y']

        self.model.fit(data_trn_w['x'], y_trn)

        res = []
        ykey, mlist = metrics.popitem()
        for i in range(1, self.n_estimators + 1):
            if self.ltype == 'reg':
                y_pred = self.model.predict(data_tst_w['x'], ntree_limit=i)
                y_pred = y_pred[:, np.newaxis, np.newaxis]
            else:
                y_pred = self.model.predict_proba(data_tst_w['x'], ntree_limit=i)
                y_pred = y_pred[:, np.newaxis, :]

            for metric in mlist:
                val = metric(y_true, y_pred)
                logger('test.' + metric.__name__, val, i)

        res.append('{} = {:.4f}'.format(metric.__name__, val))

        return '\n'.join(res)

    def predict(self, data_full):
        data_w = self._preprocess_data(data_full)
        if self.ltype == 'reg':
            preds = self.model.predict(data_w['x'])
            preds = preds[:, np.newaxis]
        else:
            preds = self.model.predict_proba(data_w['x'])

        pred_seq = np.zeros(data_full[self.ykey].shape[:2] + (preds.shape[-1], ))

        for coord, val in zip(data_w['map2seq'], preds):
            pred_seq[coord] = val

        return pred_seq


make_model = XGB
