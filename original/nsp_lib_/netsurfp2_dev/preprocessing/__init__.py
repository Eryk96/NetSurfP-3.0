
import numpy as np

HH_HEADER = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', 'M->M', 'M->I', 'M->D', 'I->M', 'I->I', 'D->M',
    'D->D', 'Neff', 'Neff_I', 'Neff_D'
]

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def zipalign(hhm_data, dssp_data):
    if len(hhm_data['profile']) == len(dssp_data):
        for i, (hh_res, hh_datum, ss_datum) in enumerate(
                zip(hhm_data['seq'], hhm_data['profile'], dssp_data)):
            assert hh_res == ss_datum['restype']
            yield i, hh_datum, ss_datum
    else:
        hh_index = -1
        for i, ss_datum in enumerate(dssp_data):
            if ss_datum['restype'] != 'X':
                hh_index += 1
                assert ss_datum['restype'] == hhm_data['seq'][hh_index]

            hh_datum = hhm_data['profile'][hh_index] if hh_index >= 0 else None
            yield i, hh_datum, ss_datum


def vectorize_profile(molid, hhm_data, dssp_data, asa_max, eval_mask, hh_header=None):
    hh_header = hh_header or HH_HEADER
    num_feat = 20 + len(hh_header) + 18

    matrix = np.zeros((len(dssp_data), num_feat))

    current_eval_mask = 1
    switch_next = False
    if eval_mask:
        current_eval_mask = 0

    # print(molid)

    for i, hh_datum, ss_datum in zipalign(hhm_data, dssp_data):
        residx = -1

        # [0:20] Amino Acids (sparse encoding)
        # Unknown residues are stored as an all-zero vector
        if ss_datum['restype'] in AMINO_ACIDS:
            residx = AMINO_ACIDS.index(ss_datum['restype'])
            matrix[i, residx] = 1.

        # [20:50] hmm profile
        if hh_datum is not None:
            for j, h in enumerate(hh_header):
                matrix[i, j + 20] = hh_datum[h]

        # [50] Seq mask (1 @ seq, 0 @ empty)
        # [51] Disordered mask (0 = disordered, 1 = ordered)
        j = 20 + len(hh_header)
        matrix[i, j + 0] = 1.
        matrix[i, j + 1] = int(not ss_datum['disordered'])

        # [52] Evaluation mask (CB513)
        if eval_mask:
            if switch_next:
                current_eval_mask = 0
                switch_next = False
            #
            try:
                resint = int(ss_datum['resid'][:-1])
                if resint == eval_mask[0]:
                    current_eval_mask = 1
                if resint == eval_mask[1]:
                    switch_next = True
            except ValueError:
                pass

        matrix[i, j + 2] = current_eval_mask

        # [53] ASA (isolated)
        # [54] ASA (complexed)
        matrix[i, j + 3] = ss_datum['asa_single']
        matrix[i, j + 4] = ss_datum['asa_complex']

        # [55] RSA (isolated)
        # [56] RSA (complexed)
        this_am = asa_max.get(ss_datum['restype'], -1)
        matrix[i, j + 5] = max([min([ss_datum['asa_single'] / this_am, 1.]), 0.])
        matrix[i, j + 6] = max([min([ss_datum['asa_complex'] / this_am, 1.]), 0.])

        # OLD!!! ==> [57:65] Q8 HGBELTSI (Q8 -> Q3: HHEECCCC)
        # [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
        Q8 = 'GHIBESTC'
        j = 20 + len(hh_header) + 7
        assert ss_datum['ss'] in Q8
        ss_idx = Q8.index(ss_datum['ss'])
        matrix[i, j + ss_idx] = 1.

        # [65:67] Phi+Psi
        j = 20 + len(hh_header) + 15
        matrix[i, j + 0] = ss_datum['phi']
        matrix[i, j + 1] = ss_datum['psi']
        # [67] ASA_max
        matrix[i, j + 2] = this_am

    return matrix