#!/usr/bin/env python
"""
Functions for running DSSP on PDB biounits
"""

import os
import glob
import json
import functools
import argparse

CONVERT3_1 = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}


def run_pdblist(pdblist, minlen):
    """Run DSSP on pdblist."""
    import tqdm

    errors = 0
    filtered = 0

    dssp_out = {}

    desc = 'Running DSSP'
    pdblist_iter = tqdm.tqdm(sorted(pdblist.items()), desc=desc, ncols=80)
    for (pdbid, chainid), (sequence, disorder) in pdblist_iter:
        try:
            do = run_biounit(pdbid, chainid, sequence, disorder)
        except RunDsspError as err:
            tqdm.tqdm.write('{}: {} -- {}-{}'.format(
                type(err).__name__, err, pdbid, chainid))
            errors += 1
            continue

        if do:
            ordered = len([r for r in do if not r['disordered']])
            if ordered >= minlen:
                dssp_out[pdbid + '-' + chainid] = do
            else:
                filtered += 1
        else:
            errors += 1
            msg = 'Nothing: {}-{}'.format(pdbid, chainid)
            tqdm.tqdm.write(msg, file=sys.stderr)

    return dssp_out, errors, filtered


def parse_pdbunit(filehandle, target_chain):
    """Parse a PDB biounit.

    Returns the full biounit and the isolated chain. (only ATOM records) as
    well as the seqres.

    Lot's of gymnastics to make sure the right target chain is extracted.

    """
    models = []  #Full lines
    chains = []  #sets of chains
    seqres = {}  #sequences
    for line in filehandle:
        if line[:5] == 'MODEL':
            models.append([])
            chains.append(set())
        elif line[:6] in ('ATOM  ', 'TER   ', 'HETATM'):
            if not models:
                models.append([])
                chains.append(set())

            chains[-1].add(line[21])
            models[-1].append(line)
        elif line[:6] == 'SEQRES':
            chain_ = line[11]
            seqres[chain_] = seqres.get(chain_, [])
            seqres[chain_].extend(
                [CONVERT3_1.get(aa3, 'X') for aa3 in line[19:].split()])

    if not models:
        raise Exception('Empty PDB')

    # print(filehandle.name, sum(len(c) for c in chains))

    # If the same pdb chain is present multiple times in an assembly, the
    # different chains are put in different models in the pdb file.
    # So we rename extra chains and save the mapping so we can recover the
    # right chain later.

    used_chains = set()

    def _increment_chain(chain):
        new_chain = chain
        for _ in range(65):
            if new_chain not in used_chains:
                break
            if new_chain == 'Z':
                new_chain = 'a'
            elif new_chain == 'z':
                new_chain = '0'
            elif new_chain == '9':
                new_chain = 'A'
            else:
                new_chain = chr(ord(new_chain) + 1)
        else:
            new_chain = 'Z' if target_chain != 'Z' else 'X'

        used_chains.add(new_chain)
        return new_chain

    # Assign a unique chain (if possible, some assemblies have more than 65
    # chains) to any chain in the PDB file.

    chain_mapping = []
    for mdl_chains in chains:
        chain_mapping.append({})
        for chain in mdl_chains:
            chain_mapping[-1][chain] = _increment_chain(chain)

    # Reduce all the chain sets down to one set
    unique_chains = functools.reduce(lambda a, b: a | b, chains)

    # PISCES puts `0` if the pdb only contains one chain.
    if target_chain == '0' and len(unique_chains) == 1:
        target_chain = unique_chains.pop()
    # If the chain we are looking for is not present, we are looking at the
    # wrong file and return nothing.
    elif not any([target_chain in m for m in chain_mapping]):
        return None

    complex_out = []
    single_out = []

    for model, mdl_chain_map in zip(models, chain_mapping):
        for line in model:
            chain = line[21]
            if chain != mdl_chain_map[chain]:
                chain = mdl_chain_map[chain]
                line = line[:21] + chain + line[22:]
            complex_out.append(line)
            if chain == target_chain:
                single_out.append(line)

    return complex_out, single_out, ''.join(seqres[target_chain])


def run_biounit(pdbid, chainid, sequence, disorder,
                # subset=None,
                pdbdir='/data/Databases/pdb/data/biounit/PDB/',
                cachedir='/data/Cache/NetSurfP-2.0/dssp_biounit',
                cifdir='/data/Databases/pdb/data/structures'):
    """Main function to parse biounits and run DSSP."""
    # cachedir = None
    if cachedir:
        cachefile = os.path.join(cachedir, '{}_{}.json.gz'.format(pdbid, chainid))

        import gzip
        if os.path.isfile(cachefile):
            with gzip.open(cachefile, 'rt') as f:
                return json.load(f)

    pdbpath = os.path.join(pdbdir, 'divided', pdbid[1:3], pdbid + '.pdb*.gz')
    pdbfiles = glob.glob(pdbpath)

    if not pdbfiles:
        raise PdbNotFound(pdbpath)

    pdbfile = None
    parsed = {}
    for pdbfile_ in pdbfiles:
        import gzip
        import re
        assembly_id = re.search(pdbid + r'\.pdb(\d+)\.gz', pdbfile_).group(1)
        with gzip.open(pdbfile_, 'rt') as f:
            parsed_ = parse_pdbunit(f, chainid)
            if parsed_:
                parsed[assembly_id] = parsed_

    import shlex
    if not parsed:
        return
    # If there is more than one candidate, we have to pick out the author
    # assembly over the predicted assembly. This information is in the CIF
    # files.
    elif len(parsed) > 1:
        ciffile = os.path.join(cifdir, 'divided', 'mmCIF', pdbid[1:3],
                               pdbid + '.cif.gz')
        with gzip.open(ciffile, 'rt') as f:
            cifdata = {}
            for line in f:
                if line.startswith('loop_'):
                    header0 = next(f).strip()
                    if not header0.startswith('_pdbx_struct_assembly.'):
                        continue
                    headers = [header0.strip().split('.')[-1]]
                    for header in f:
                        if not header.startswith('_'):
                            break
                        headers.append(header.strip().split('.')[-1])
                    # Add the first line of the value list to the output dict
                    d_ = {k: v for k, v in zip(headers, shlex.split(header))}
                    cifdata[d_['id']] = d_
                    #Go through each value list, stop at '#'
                    for dataline in f:
                        if dataline.startswith('#'):
                            break
                        d_ = {
                            k: v
                            for k, v in zip(headers, shlex.split(dataline))
                        }
                        cifdata[d_['id']] = d_

        author_ass = None
        for assid, parsed_ in parsed.items():
            if 'author' in cifdata[assid]['details']:
                author_ass = parsed_
                break

        if not author_ass:
            author_ass = parsed[sorted(parsed)[0]]

        complex_lines, single_lines, seqres = author_ass
    else:
        complex_lines, single_lines, seqres = parsed[sorted(parsed)[0]]

    # assert complex_lines and single_lines, pdbid
    import tempfile

    with tempfile.NamedTemporaryFile(mode='tw') as f:
        #print(*single_lines)
        f.writelines(single_lines)
        f.flush()
        # single_rec, single_seq = dssp(f.name, subset=subset)
        single_rec, single_seq = dssp(f.name)

    single_seqres = ''.join(single_rec[k]['restype'].strip('!') for k in single_seq)

    if single_seqres != seqres:
        import Bio.pairwise2

        #Extreme high penalties on opening gaps
        aln = Bio.pairwise2.align.globalms(single_seqres, seqres, 5, -4, -10,
                                           -.1)[0]
        cov = 0
        tml = 0
        for a, b in zip(*aln[:2]):
            if a != '-':
                tml += 1
                if a == b:
                    cov += 1

        if cov / tml < 0.8:
            print(pdbid, chainid)
            print(Bio.pairwise2.format_alignment(*aln))
            raise Exception()

        import re
        aln = list(aln)
        aln[0], n = re.subn(r'(.+)(\w)(\-+)(\w)$', r'\1\2\4\3', aln[0])

        assert n <= 1, (pdbid, chainid)

        clean_dsspseq = [r for r in single_seq if r]
        idx = 0
        disidx = 1
        new_dsspseq = []
        for query, target in zip(*aln[:2]):
            if query == '-':
                dis_id = 'DISORDER {}'.format(disidx)
                new_dsspseq.append(dis_id)
                single_rec[dis_id] = {
                    'restype': target,
                    'disordered': True,
                    'asa_single': 0,
                    'asa_complex': 0,
                    'phi': 0.0,
                    'psi': 0.0,
                    'Ca-x': 0.0,
                    'Ca-y': 0.0,
                    'Ca-z': 0.0,
                    'ss': 'C',
                    'masked': False,
                }
                disidx += 1
            else:
                new_dsspseq.append(clean_dsspseq[idx])
                idx += 1

        single_seq = new_dsspseq

    with tempfile.NamedTemporaryFile(mode='tw') as f:
        f.writelines(complex_lines)
        f.flush()
        complex_rec, complex_seq = dssp(f.name)

    # print()

    out = []
    for resid in single_seq:
        rec = single_rec[resid]

        if rec['restype'] == '-':
           continue

        rec['resid'] = resid

        if not single_rec[resid]['disordered']:
            rec['asa_single'] = rec['asa']
            rec['asa_complex'] = complex_rec[resid]['asa']
            del rec['asa']

        # if rec['restype'] == '!':
        #     raise DsspFail('Missed disordered residue in alignment')
        
        out.append(rec)

    if cachedir:
        with gzip.open(cachefile, 'wt') as f:
            json.dump(out, f)

    return out


class RunDsspError(Exception):
    pass


class DsspFail(RunDsspError):
    pass


class PdbNotFound(RunDsspError):
    pass


def dssp(filename, subset=None):
    import subprocess
    p = subprocess.Popen(
        ['mkdssp', '-i', filename],
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
        # print('timeout:', pdbfile, file=sys.stderr)
        raise DsspFail('DSSP timeout')

    if p.returncode != 0:
        # print(err, file=sys.stderr)
        # print(out, file=sys.stderr)
        raise DsspFail(err)

    dssp_seq = []
    records = {}

    # print(out)

    parsing = False
    for line in out.strip().splitlines():
        if '#  RESIDUE AA STRUCTURE' in line:
            parsing = True
        elif parsing:
            record = {}
            resid = line[5:12].lstrip()
            restype = line[13]
            # if restype == '!':
            #     continue

            dssp_seq.append(resid)
            records[resid] = record

            # if subset:
            #     resnum = line[5:10].strip()
            #     if resnum:
            #         resnum = int(line[5:10])
            #         if resnum == subset[0]:
            #             before_subset = False
            #         if resnum == subset[1]:
            #             this_one = True
            #             after_subset = True

            #     if before_subset or (after_subset and not this_one):
            #         record['masked'] = False
            #     else:
            #         record['masked'] = True
            #     this_one = False
            # else:
            #     record['masked'] = False

            # DSSP detects breaks in chains.
            if restype == '!':
               record['restype'] = '-'
               record['disordered'] = True
            else:
                record['restype'] = restype
                record['disordered'] = False

            record['asa'] = float(line[34:38])
            record['phi'] = float(line[103:109])
            record['psi'] = float(line[109:115])
            record['Ca-x'] = float(line[115:122])
            record['Ca-y'] = float(line[122:129])
            record['Ca-z'] = float(line[129:136])

            this_ss = 'C' if line[16] == ' ' else line[16]
            record['ss'] = this_ss

    return records, dssp_seq
