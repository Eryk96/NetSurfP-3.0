"""
DOC
"""

import os
import glob

import numpy as np

import netsurfp2_dev.models
import netsurfp2_dev.data
import netsurfp2_dev.targets


def main(_run, modeltype, modelargs, outputs, datafile, prev_layer,
         n_layer, n_folds, tst_fold, tst_samples=None, save_preds=False,
         dry_run=False): #yapf: disable

    print('Loading:', datafile)
    datafp = _run.open_resource(datafile, 'rb')
    raw = np.load(datafp)['data']

    x_data = raw[:, :, 0:51]
    if prev_layer:
        x_files = sorted(glob.glob(os.path.join(prev_layer, '*.npz')))
        new_x = [x_data]
        for x_file in x_files:
            with open(x_file) as fp:
                prev_preds = np.load(x_file)['preds']
                if len(prev_preds.shape) == 3:
                    prev_preds = [prev_preds]

                for pp in prev_preds:
                    new_x.append(pp)

        x_data = np.concatenate(new_x, axis=-1)


    data_full = {'x': x_data, 'x_mask': raw[:, :, 50:51]}

    metrics = {}
    outputs_ = []
    transfers = {}
    transfer_temps = {}

    for odict in outputs:
        name = odict['name']
        w = odict['w']
        ltype, metcs, y = netsurfp2_dev.targets.get_output(name)(raw)
        metrics['y_' + name] = metcs
        data_full['y_' + name] = y
        outputs_.append([name, w, ltype])

        if 'transfer_target' in odict:
            transfers['y_' + name] = odict['transfer_target']
        if 'transfer_temp' in odict:
            transfer_temps['y_' + name] = odict['transfer_temp']

    model = netsurfp2_dev.models.get(modeltype)()
    model.build(data_full, outputs_, **modelargs)

    if tst_samples:
        trn_mask, tst_mask = netsurfp2_dev.data.split(raw.shape[0], tst_samples)
    else:
        trn_mask, tst_mask = netsurfp2_dev.data.cv_split(raw.shape[0], n_layer, n_folds,
                                                    tst_fold)

    data_trn = {}
    data_tst = {}

    for dataname, dat_ in data_full.items():
        if dataname in transfers:
            trf = np.load(transfers[dataname])[dataname[2:]]
            trf_trn = trf[trn_mask]
            if dataname in transfer_temps:
                T = transfer_temps[dataname]
                trf_trn = np.exp(trf_trn / T) / np.sum(np.exp(trf_trn / T), axis=-1, keepdims=True)
            data_trn[dataname] = np.concatenate([trf_trn, dat_[trn_mask][:, :, -1:]], axis=-1)
            print('TRANSFER:', dataname, transfers[dataname], data_trn[dataname].shape, 'ORIG', dat_[trn_mask].shape)
        else:
            data_trn[dataname] = dat_[trn_mask]
        
        data_tst[dataname] = dat_[tst_mask]

    print("X FEATURES =", x_data.shape[2])

    print('TRN SAMPLES = {:>6,}'.format(data_trn['x'].shape[0]))
    print('TST SAMPLES = {:>6,}'.format(data_tst['x'].shape[0]))
    print()

    print('Longest sequence = {:,}'.format(raw.shape[1]))
    max_diso = np.max(np.sum(raw[:, :, 50], axis=1) - np.sum(raw[:, :, 51], axis=1))
    print('Max disorder = {:.0f}'.format(max_diso))

    if dry_run:
        return

    modelfile = None
    if save_preds:
        modelfile = save_preds.replace('.npz', '.h5')
    output = model.fit(data_trn, data_tst, metrics,
                       logger=_run.log_scalar, save=modelfile) #yapf: disable

    if save_preds:
        preds = model.predict(data_full)
        np.savez_compressed(save_preds, preds=preds)

    return output


def run(cfg, exname, dbname):
    """Create a Sacred experiment and run it."""
    import sacred.observers
    import sacred.utils

    ex = sacred.Experiment(exname)
    ex.add_config(cfg)
    ex.main(main)

    ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

    import netsurfp2_dev.models
    ex.add_source_file(netsurfp2_dev.models.get_filename(cfg['modeltype']))

    observer = sacred.observers.MongoObserver.create(db_name=dbname)
    ex.observers.append(observer)

    slack_cfg = '/home/mskl/PhD_external/NetSurfP2/00_Lib/slack.json'
    slack_obs = sacred.observers.SlackObserver.from_config(slack_cfg)
    ex.observers.append(slack_obs)

    ex.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('cfg', help='Experiment configuration json file')
    parser.add_argument('-m', help='MongoDB database', default='sacred')
    parser.add_argument(
        '--now', action='store_true', help='Run immediately, no queue.')
    parser.add_argument(
        '--right-now',
        action='store_true',
        help='Run immediately, no queue, and no sacred registration.')
    parser.add_argument(
        '--test', help='Run minimal model', action='store_true')
    parser.add_argument(
        '--skip-check', help='Skip Check', action='store_true')
    args = parser.parse_args()

    import os
    import json
    with open(args.cfg) as f:
        cfg = json.load(f)

    cfg['datafile'] = os.path.abspath(cfg['datafile'])
    exname = os.path.splitext(os.path.split(args.cfg)[1])[0]

    if cfg['save_preds']:
        cfg['save_preds'] = os.path.join(cfg['save_preds'], exname + '.npz')

    class Decoy:
        open_resource = open

        def log_scalar(self, key, val, t):
            return

    if args.right_now:
        del cfg['queue']
        print(main(Decoy(), **cfg))
    else:
        if args.now:
            run(cfg, exname, args.m)
        else:
            import rq
            import redis
            import os

            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            queue_name = cfg['queue']
            del cfg['queue']

            if not args.skip_check:
                #Load the model to see if anything goes wrong
                main(_run=Decoy(), dry_run=True, **cfg)

            q = rq.Queue(queue_name, connection=redis.Redis())
            import netsurfp2_dev.run
            q.enqueue(netsurfp2_dev.run.run, cfg, exname, args.m, timeout='168h')
