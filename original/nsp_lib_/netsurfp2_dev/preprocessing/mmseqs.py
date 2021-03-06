"""
MMSeqs2 wrappers.

"""

import os
import sys
import tempfile
import contextlib
import subprocess
import multiprocessing

import tqdm

from . import hhsuite


@contextlib.contextmanager
def nottempdir(tmpdir):
    yield tmpdir


def run_pdblist(pdblist, targetdb, tmpdir=None):
    if tmpdir is None:
        tmpcontext = tempfile.TemporaryDirectory()
    else:
        tmpcontext = nottempdir(tmpdir)

    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with tmpcontext as tmpdir:
        print('Running MMSeqs2 in {}..'.format(tmpdir), file=sys.stderr)

        input_faa = os.path.join(tmpdir, 'in.fasta')
        querydb = os.path.join(tmpdir, 'in.mm')

        if not os.path.isfile(querydb):
            with open(input_faa, 'w') as tmpfaa:
                for molid, dat in sorted(pdblist.items()):
                    print('>' + molid, file=tmpfaa)
                    print(''.join(r['restype'] for r in dat), file=tmpfaa)

            subprocess.run(['mmseqs', 'createdb', input_faa, querydb], **skw)

        mm_log = os.path.join(tmpdir, 'log.mm_search')

        with open(mm_log, 'a') as logfp:
            alns = search(querydb, targetdb, tmpdir,
                max_seqs=2000, prefix='', logfp=logfp) #yapf: disable

        pool = multiprocessing.Pool()

        promises = {}
        for molid in sorted(pdblist):
            aln = alns[molid]
            promises[molid] = pool.apply_async(make_profile,
                                               (molid, aln, tmpdir))

        profiles_out = {}

        desc = 'Generating profiles'
        kwds = dict(desc=desc, smoothing=0, ncols=80)
        prog = tqdm.tqdm(sorted(pdblist), **kwds)
        for molid in prog:
            try:
                hhout = promises[molid].get()
                profiles_out[molid] = hhout
            except Exception:
                prog.write('Error: ' + molid)

    return profiles_out


def make_profile(molid, aln, tmpdir):
    tmpaln = os.path.join(tmpdir, molid + '_tmpaln.fasta')
    with open(tmpaln, 'w') as f:
        f.write(aln)

    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    proc = subprocess.run(['hhmake', '-i', tmpaln, '-v', '0',
                           '-o', 'stdout', '-M', 'first'],
                          universal_newlines=True, **skw)

    return hhsuite.parse_hhm(iter(proc.stdout.splitlines()))


def search(querydb, targetdb, tmpdir, max_seqs=300, prefix='', logfp=subprocess.PIPE):
    """Run MMSeqs2 search."""
    msadb = os.path.join(tmpdir, prefix + 'out.mm_msa')
    mm_result = os.path.join(tmpdir, prefix + 'out.mm_search')

    if not os.path.isfile(msadb):
        print('mmseqs search ...', file=sys.stderr)

        tmpdir2 = os.path.join(tmpdir, prefix + 'tmp2')
        os.mkdir(tmpdir2)

        subprocess.run(['mmseqs', 'search', querydb, targetdb,
                        mm_result, tmpdir2, '--num-iterations', '2',
                        '--max-seqs', str(max_seqs)],
                       stdout=logfp, stderr=logfp, check=True) #yapf: disable

        subprocess.run(['mmseqs', 'result2msa', querydb,
                        targetdb, mm_result, msadb],
                       stdout=logfp, stderr=logfp, check=True) #yapf: disable

    alns = {}
    with open(msadb) as fdat, open(msadb + '.index') as fidx:
        for line in fidx:
            msa_id, start, length = line.strip().split('\t')
            fdat.seek(int(start))
            msa = fdat.read(int(length) - 1)
            molid = msa.split(None, 1)[0][1:]

            alns[molid] = msa

    return alns