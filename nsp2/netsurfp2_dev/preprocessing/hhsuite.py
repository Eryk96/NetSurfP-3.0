"""
HHSuite wrappers.

"""

import os
import json
import gzip
import shutil
import multiprocessing

import tqdm


def run_pdblist(pdblist, db, n_threads=None, hh_iters='2', use_msa=False, mact=0.35):
    if n_threads is None:
        n_threads = multiprocessing.cpu_count() // 2
    
    pool = multiprocessing.Pool(n_threads)
    promises = {}

    for molid, dat in sorted(pdblist.items()):
        seq = ''.join(r['restype'] for r in dat)
        promises[molid] = pool.apply_async(hhblits, (molid, seq, db, hh_iters), {'use_msa': use_msa, 'mact': mact}), seq

    longest_seq = 0
    profiles_out = {}

    desc = 'Running HHBlits'
    for molid, (promise, seq) in tqdm.tqdm(sorted(promises.items()), ncols=80,
                                           desc=desc, leave=True, smoothing=0):
        try:
            hhout = promise.get()
        except Exception:
            raise Exception('Error:' + molid)

        if hhout['seq'] != seq:
            # print(molid.upper() + '.*')
            raise Exception('Error:' + molid)

        profiles_out[molid] = hhout

    return profiles_out


def freq(freqstr):
    if freqstr == '*':
        return 0.
    p = 2**(int(freqstr) / -1000)
    assert 0 <= p <= 1.0
    return p


def parse_hhm(hhmfp):
    neff = None
    for line in hhmfp:
        if line[0:4] == 'NEFF' and neff is None:
            neff = float(line[4:].strip())
        if line[0:8] == 'HMM    A':
            header1 = line[7:].strip().split('\t')
            break

    header2 = next(hhmfp).strip().split('\t')
    next(hhmfp)

    seq = []
    profile = []
    for line in hhmfp:
        if line[:2] == '//':
            break
        aa = line[0]
        seq.append(aa)

        freqs = line[7:].split('\t')[:20]
        features = {h: freq(i) for h, i in zip(header1, freqs)}
        assert len(freqs) == 20

        mid = next(hhmfp)[7:].strip().split('\t')

        features.update({h: freq(i) for h, i in zip(header2, mid)})

        profile.append(features)
        next(hhmfp)

    return {
        'seq': ''.join(seq),
        'profile': profile,
        'neff': neff,
        'header': header1 + header2,
    }


def hhblits(molid, seq, db, hh_iters='2', cachedir='/data/Cache/NetSurfP-2.0/hhblits', use_msa=False, mact='0.35'):

    mact = '{:.2f}'.format(float(mact))

    db_short = os.path.split(db)[-1]
    cachedir = os.path.join(cachedir, db_short, 'N{}M{}'.format(hh_iters, mact))
    if not os.path.isdir(cachedir):
        try:
            os.mkdir(cachedir)
        except FileExistsError:
            pass

    molid = molid.upper()
    cachebase = os.path.join(cachedir, molid)
    cachefile = cachebase + '.hhm'

    import gzip
    import tempfile
    import subprocess

    if os.path.isfile(cachefile):
        if not use_msa:
            with open(cachefile) as hhmfp:
                return parse_hhm(hhmfp)
        else:
            raise NotImplementedError
            
        # with gzip.open(cachefile, 'rt') as f:
        #     if not use_msa:
        #         hhmfilename = cachebase + '.hhm'
        #         with open(hhmfilename) as hhmfp:
        #             parsed_hhm = 
        #         return json.load(f)
        #     else:
        #         outa3m = cachebase + '.a3m'
        #         proc = subprocess.run(['hhmake', '-i', outa3m, '-v', '0',
        #                                '-o', 'stdout'],
        #                                universal_newlines=True,
        #                                stdout=subprocess.PIPE,
        #                                stderr=subprocess.PIPE)
        #         with open(cachebase + '.a3m.hhm', 'w') as f:
        #             f.write(proc.stdout)

        #         return parse_hhm(iter(proc.stdout.splitlines()))


    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta') as tmpf, \
            tempfile.NamedTemporaryFile(suffix='.hhm', mode='rt') as outf, \
            tempfile.NamedTemporaryFile(suffix='.a3m', mode='rt') as outa3m:
        print('>' + molid, file=tmpf)
        print(seq, file=tmpf)
        tmpf.flush()

        cmd = [
            'hhblits', '-i', tmpf.name, '-o', '/dev/null', '-ohhm', outf.name,
            '-n', str(hh_iters), '-d', db, '-cpu', '2', '-oa3m', outa3m.name,
            '-mact', mact,
        ] # yapf: disable

        try:
            # os.environ['HHLIB'] = '/opt/hhsuite'
            o = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            print(exc.returncode, exc.output)
            raise

        outfname = outf.name
        outf = iter(outf)
        dat = parse_hhm(outf)

        # with gzip.open(cachefile, 'wt') as f:
        #     json.dump(dat, f)
        
        shutil.copyfile(outfname, cachebase + '.hhm')
        shutil.copyfile(outa3m.name, cachebase + '.a3m')

        return dat