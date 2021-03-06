#!/usr/bin/env python

"""
Preprocess data from a list of PDB files.
"""

import os
import re
import sys
# import json
import argparse
import collections

import tqdm

from . import dssp
# from . import mmseqs
# from . import hhsuite


def parse_pdblist(fp):
    """Read a PDB list."""
    pdbnames = set()
    eval_masks = collections.defaultdict(list)

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

        pdbnames.add((pdbid, chainid))

    return pdbnames, dict(eval_masks)


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

    mm_parser = subparsers.add_parser('mmseqs', aliases=['mm'],
        description=__doc__ + 'Searching with MMSeqs2.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    mm_parser.add_argument('--db', help='MMSeqs curated database',
        default='/data/Databases/mmseqs/uniclust90_2017_04')
    mm_parser.add_argument('--meta-db', help='MMSeqs metagenomics database',
        default=None)
    mm_parser.add_argument('--tmpdir', help='MMSeqs temporary directory',
        default=None)

    parser.add_argument('pdblist', help='Input PISCES pdblist')
    parser.add_argument('--minlen', help='Minimum seqlen',
        type=int, default=30)
    parser.add_argument('--blastdb', help='CB513 blast database',
        default=None)
    parser.add_argument('--asa-max', help='ASA[max] json file',
        default='ASA_max_Sarai2003.json')

    args = parser.parse_args()
    # yapf: enable

    #
    # Parse pdblist
    #
    with open(args.pdblist) as fp:
        raw_pdblist, eval_masks = parse_pdblist(fp)

    #
    # Run DSSP
    #
    dssp_out, err, filt = dssp.run_pdblist(raw_pdblist, minlen=args.minlen)

    # print('{} chains, {} filtered, {} errors.'.format(len(dssp_out), filt, err),
    #     file=sys.stderr)