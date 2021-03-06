
"""
Functions for running DSSP on PDB biounits
"""

import os
import gzip

from . import mmcif


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
    'TYR': 'Y'
}


def run_pdblist(pdblist, minlen):
    """Run DSSP on pdblist."""
    import tqdm 
    desc = 'Running DSSP'
    for pdbid, chainid in tqdm.tqdm(sorted(pdblist), desc=desc, ncols=80):
        try:
            do = run_biounit(pdbid, chainid)
        except RunDsspError as err:
            tqdm.tqdm.write('{}: {} -- {}-{}'.format(
                type(err).__name__, err, pdbid, chainid))
            errors += 1
            continue


def run_biounit(pdbid, chainid, cachedir='/data/Cache/NetSurfP-2.0/dssp_biounit2/'):
    """.."""
    dbdir = '/data/Databases/pdb/data/structures/divided/mmCIF/'
    filename = os.path.join(dbdir, pdbid[1:3], pdbid.lower() + '.cif.gz')
    if not os.path.isfile(filename):
        raise PdbNotFound(pdbid + '-' + chainid)

    with gzip.open(filename, 'rt') as f:
        cif_full = mmcif.mmCIF.parse(f)

    target_label_id = cif_full.auth2label[chainid]
    cif_single = cif_full.isolate_chains(target_label_id)

    fname = os.path.join(cachedir, '{}-{}.cif'.format(pdbid, chainid))
    with open(fname, 'w') as f:
        cif_single.write(f)

    # print(len(cif_full.records['_atom_site.label_asym_id']))
    # print(len(cif_single.records['_atom_site.label_asym_id']))
    # print(target_label_id)

    # print(auth2label)
    #for k in ('_pdbx_struct_assembly_gen.assembly_id', '_pdbx_struct_assembly_gen.oper_expression', '_pdbx_struct_assembly_gen.asym_id_list', ):
    #    print(parsed[k])

    raise Exception


def dssp(filename):
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
        raise DsspFail('DSSP timeout')

    if p.returncode != 0:
        raise DsspFail(err)

    return out


class RunDsspError(Exception):
    pass


class DsspFail(RunDsspError):
    pass


class PdbNotFound(RunDsspError):
    pass