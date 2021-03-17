#!/usr/bin/env python

"""
Preprocess data from a list of PDB files.
"""

import os
import re
import sys
import json
import argparse
import collections

import tqdm

from . import dssp
from . import mmseqs
from . import hhsuite
from . import vectorize_profile


def parse_pdblist(fp, seqres):
    """Read a PDB list."""
    pdbnames = dict()
    eval_masks = collections.defaultdict(list)

    # errors = False

    next(fp)
    for line in fp:
        line = line.split('#')[0]
        if not line.strip():
            continue

        line = line.split()[0]
        pdbid = line[:4].lower()
        chainid = line[4]

        m = re.match(r'\w{5}\.(-?\d+)-(-?\d+)', line)
        if m:
            eval_mask = (int(m.group(1)), int(m.group(2)))
            molid = pdbid.lower() + '-' + chainid
            eval_masks[molid].append(eval_mask)

        # try:
        sequence = seqres.get('{}:{}:sequence'.format(pdbid.upper(), chainid))
        disorder = seqres.get('{}:{}:disorder'.format(pdbid.upper(), chainid))
        pdbnames[(pdbid, chainid)] = (sequence, disorder)
        # except KeyError:
        # print('Not in ss_dis.txt: {}:{}'.format(pdbid, chainid))
        # errors = True

    # if errors:
    #     raise Exception('Errors reading PDB list')

    return pdbnames, dict(eval_masks)


def run_blast(molid, seq, db, cov=0.8, ident=25, cache=True):
    import subprocess

    molid = molid.upper()
    prefiltered = {}
    if cache:
        cachefile = db + '.netsurfp_cache'

        if os.path.isfile(cachefile):
            with open(cachefile) as f:
                prefiltered = json.load(f)
                if molid in prefiltered:
                    return prefiltered[molid]

    prefiltered[molid] = False

    stdin = '>{}\n{}'.format(molid, seq)
    p = subprocess.Popen(
        ['blastp', '-db', db, '-outfmt', '10'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
    out, err = p.communicate(stdin)

    if p.returncode != 0 or err:
        raise Exception(err)

    for hit in out.splitlines():
        hit = hit.split(',')
        ident = float(hit[2])
        alnlen = int(hit[3])
        if alnlen >= len(seq) * 0.8 and ident >= 25:
            prefiltered[molid] = True

    if cache:
        with open(cachefile, 'w') as f:
            json.dump(prefiltered, f)

    return prefiltered[molid]


if __name__ == '__main__':
    # yapf: disable
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title='search method', dest='method')

    hh_parser = subparsers.add_parser('hhblits', aliases=['hh'],
        description=__doc__ + 'Searching with HHBlits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    hh_parser.add_argument('--db', help='HHBlits database',
        default='/data/Databases/hhsuite/uniclust30_2017_04/uniclust30_2017_04')
    hh_parser.add_argument('-t', help='Number of threads to start',
        default=None, type=int)
    hh_parser.add_argument('-n', help='Number of hhblits iterations',
        default=2, type=int)
    hh_parser.add_argument('--mact', help='hhblits -mact',
        default=0.35, type=float)
    hh_parser.add_argument('--use-msa',
        help='Use the generated MSA instead of HHM profile',
        action='store_true')

    mm_parser = subparsers.add_parser('mmseqs', aliases=['mm'],
        description=__doc__ + 'Searching with MMSeqs2.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mm_parser.add_argument('--db', help='MMSeqs curated database',
        default='/data/Databases/mmseqs/uniclust90_2017_04')
    mm_parser.add_argument('--tmpdir', help='MMSeqs temporary directory',
        default=None)

    parser.add_argument('pdblist', help='Input PISCES pdblist')
    parser.add_argument('--minlen', help='Minimum seqlen',
        type=int, default=20)
    parser.add_argument('--blastdb', help='CB513 blast database',
        default=None)
    parser.add_argument('--asa-max', help='ASA[max] json file',
        default='ASA_max_Sarai2003.json')

    args = parser.parse_args()
    # yapf: enable

    # import gzip
    # with gzip.open('ss_dis.txt.gz', 'rt') as f:
    #     seqres = {}
    #     is_secstr = False
    #     for line in f:
    #         if line[0] == '>':
    #             line = line.strip()
    #             if line.endswith(':secstr'):
    #                 is_secstr = True
    #             else:
    #                 current_id = line[1:]
    #                 seqres[current_id] = ''
    #                 is_secstr = False
    #         elif not is_secstr:
    #             seqres[current_id] += line.strip('\n\r')
    seqres = {}

    #
    # Parse pdblist
    #
    with open(args.pdblist) as fp:
        raw_pdblist, eval_masks = parse_pdblist(fp, seqres)

    #
    # Run DSSP
    #
    dssp_out, err, filt = dssp.run_pdblist(raw_pdblist, minlen=args.minlen)

    print('{} chains, {} filtered, {} errors.'.format(len(dssp_out), filt, err),
        file=sys.stderr)

    dssp_filtered = {}
    filtered = 0

    # raise Exception

    #
    # Remove validation set
    #
    if args.blastdb:
        desc = 'Removing validation set'
        for molid, dat in tqdm.tqdm(sorted(dssp_out.items()), desc=desc, ncols=80):
            seq = ''.join(r['restype'] for r in dat)
            is_filtered = run_blast(molid, seq, args.blastdb)
            if not is_filtered:
                dssp_filtered[molid] = dat
            else:
                filtered += 1
    else:
        dssp_filtered = dssp_out

    print('{} chains, {} filtered.'.format(len(dssp_filtered), filtered),
        file=sys.stderr) #yapf: disable

    #
    # Perform Search
    #
    outfile = args.pdblist.rsplit('.', 1)[0]
 
    if args.method in ['mm', 'mmseqs']:
        outfile +=  '.mmseqs'
        profiles_out = mmseqs.run_pdblist(dssp_filtered, args.db, tmpdir=args.tmpdir)
    elif args.method in ['hh', 'hhblits']:
        db_short = os.path.split(args.db)[-1]
        mact = '{:.2f}'.format(float(args.mact))
        outfile +=  '.hhblits.{}.N{}.M{}'.format(db_short, args.n, mact)
        if args.use_msa:
            outfile += '.msa'

        profiles_out = hhsuite.run_pdblist(dssp_filtered, args.db,
            hh_iters=args.n, n_threads=args.t, use_msa=args.use_msa, mact=args.mact)

    outfile += '.npz'

    #
    # Vectorize Data
    #
    import numpy as np

    data = None
    hh_header = None
    molidlist = []
    AAs = 'ACDEFGHIKLMNPQRSTVWY'

    with open(args.asa_max) as f:
        asa_max = json.load(f)

    #
    # Set eval masks
    if eval_masks:
        new_profiles_out = {}
        for molid, hh_dat in sorted(profiles_out.items()):
            for mask in eval_masks.get(molid, [None]):
                new_profiles_out[molid, mask] = hh_dat
    else:
        new_profiles_out = {(mid, None): hh_dat for mid, hh_dat in profiles_out.items()}

    profiles_out = new_profiles_out

    #
    # Do the actual vectorization
    profmats = {}

    for molid, eval_mask in sorted(profiles_out):
        hhm_data = profiles_out[(molid, eval_mask)]
        dssp_data = dssp_out[molid]

        profmats[molid, eval_mask] = vectorize_profile(molid, hhm_data, dssp_data, asa_max, eval_mask)

    #
    # Stack profiles with padding
    longest = max(profmats.values(), key=lambda p: p.shape[0])

    data = np.zeros((len(profmats), ) + longest.shape)

    print('Data: {}'.format(data.shape))

    molidlist = []
    for i, ((molid, _), profmat) in enumerate(sorted(profmats.items())):
        data[i, :profmat.shape[0], :] = profmat
        molidlist.append(molid)

    print('Saving to {}..'.format(outfile))

    np.savez_compressed(outfile, pdbids=molidlist, data=data)