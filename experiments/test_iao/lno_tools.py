import numpy as np
from pyscf.data import elements
from pyscf import lo, scf
from pyscf.lno.tools import autofrag_iao
from collections.abc import Iterable

from pyscf.lno import lnoccsd
from pyscf.lno import ulnoccsd

import mod_lnoccsd

def split_lno(mlno, lno_coeff, lno_frozen):
    mf = mlno._scf
    mol = mf.mol
    mo_occ = mlno.mo_occ

    if isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mo_occ)

        idx_act = np.array([i for i in range(mol.nao) if i not in lno_frozen])
        idx_frzocc = np.array([i for i in range(nocc) if i not in idx_act])
        idx_actocc = np.array([i for i in range(nocc) if i in idx_act])
        idx_actvir = np.array([i for i in range(nocc, mol.nao) if i in idx_act])
        idx_frzvir = np.array([i for i in range(nocc, mol.nao) if i not in idx_act])

        lno_frzocc = lno_coeff[:,idx_frzocc]
        lno_actocc = lno_coeff[:,idx_actocc]
        lno_actvir = lno_coeff[:,idx_actvir]
        lno_frzvir = lno_coeff[:,idx_frzvir]

        nfrzocc = len(idx_frzocc)
        nactocc = len(idx_actocc)
        nactvir = len(idx_actvir)
        nfrzvir = len(idx_frzvir)

        lno_split = [lno_frzocc, lno_actocc, lno_actvir, lno_frzvir]
    
    elif isinstance(mf, scf.uhf.UHF):
        nocc_a = np.count_nonzero(mo_occ[0])
        nocc_b = np.count_nonzero(mo_occ[1])

        idx_act_a = np.array([i for i in range(mol.nao) if i not in lno_frozen[0]])
        idx_act_b = np.array([i for i in range(mol.nao) if i not in lno_frozen[1]])

        idx_frzocc_a = np.array([i for i in range(nocc_a) if i not in idx_act_a])
        idx_actocc_a = np.array([i for i in range(nocc_a) if i in idx_act_a])
        idx_actvir_a = np.array([i for i in range(nocc_a, mol.nao) if i in idx_act_a])
        idx_frzvir_a = np.array([i for i in range(nocc_a, mol.nao) if i not in idx_act_a])
        idx_frzocc_b = np.array([i for i in range(nocc_b) if i not in idx_act_b])
        idx_actocc_b = np.array([i for i in range(nocc_b) if i in idx_act_b])
        idx_actvir_b = np.array([i for i in range(nocc_b, mol.nao) if i in idx_act_b])
        idx_frzvir_b = np.array([i for i in range(nocc_b, mol.nao) if i not in idx_act_b])

        lno_frzocc_a = lno_coeff[0][:,idx_frzocc_a]
        lno_actocc_a = lno_coeff[0][:,idx_actocc_a]
        lno_actvir_a = lno_coeff[0][:,idx_actvir_a]
        lno_frzvir_a = lno_coeff[0][:,idx_frzvir_a]
        lno_frzocc_b = lno_coeff[1][:,idx_frzocc_b]
        lno_actocc_b = lno_coeff[1][:,idx_actocc_b]
        lno_actvir_b = lno_coeff[1][:,idx_actvir_b]
        lno_frzvir_b = lno_coeff[1][:,idx_frzvir_b]

        nfrzocc = [len(idx_frzocc_a), len(idx_frzocc_b)]
        nactocc = [len(idx_actocc_a), len(idx_actocc_b)]
        nactvir = [len(idx_actvir_a), len(idx_actvir_b)]
        nfrzvir = [len(idx_frzvir_a), len(idx_frzvir_b)]

        lno_split_a = [lno_frzocc_a, lno_actocc_a, lno_actvir_a, lno_frzvir_a]
        lno_split_b = [lno_frzocc_b, lno_actocc_b, lno_actvir_b, lno_frzvir_b]
        lno_split = [lno_split_a, lno_split_b]

    print('LAS info')
    print(f'nfrozen occupied orbitals:  {nfrzocc}')
    print(f'nactive occupied orbitals:  {nactocc}')
    print(f'nactive virtual orbitals:   {nactvir}')
    print(f'nfrozen virtual orbitals:   {nfrzvir}')
    
    return  lno_split

def mo_span(mo1, s1e, mo2):
    '''
    Measure subspace containment between mo1 and mo2.
    Returns (span12, span21):
      span12 = max-abs residual for  span(mo2) ⊆ span(mo1)   ("mo1 spans mo2")
      span21 = max-abs residual for  span(mo1) ⊆ span(mo2)   ("mo2 spans mo1")
    Small residual => containment holds. Assumes mo1, mo2 orthonormal in the s1e metric.
    '''
    olp11 = mo1.T.conj() @ s1e @ mo1
    olp12 = mo1.T.conj() @ s1e @ mo2
    olp22 = mo2.T.conj() @ s1e @ mo2

    p12 = np.abs(olp12.T.conj() @ olp12 - olp22).max()
    p21 = np.abs(olp12 @ olp12.T.conj() - olp11).max()
    return p12, p21


def check_rspan(lo, s1e, mo):
    # lo passed as mo1 -> span12 tests  mo ⊆ span(lo)  (lo spans at least mo)
    p12, p21 = mo_span(lo, s1e, mo)
    return p12, p21


def check_uspan(lo, s1e, mo):
    p12a, p21a = mo_span(lo[0], s1e, mo[0])
    p12b, p21b = mo_span(lo[1], s1e, mo[1])
    return (p12a, p12b), (p21a, p21b)


def check_span(mf, lo_coeff_occ, frozen=0, thresh=1e-8):
    s1e = mf.get_ovlp()

    if isinstance(mf, scf.uhf.UHF):
        if isinstance(frozen, int):
            frozen = (frozen, frozen)
        nocc = (np.count_nonzero(mf.mo_occ[0]),
                np.count_nonzero(mf.mo_occ[1]))
        mo_coeff_occ = (mf.mo_coeff[0][:, frozen[0]:nocc[0]],
                        mf.mo_coeff[1][:, frozen[1]:nocc[1]])
        p12, p21 \
            = check_uspan(lo_coeff_occ, s1e, mo_coeff_occ)
        span12 = p12[0] < thresh and p12[1] < thresh
        span21 = p21[0] < thresh and p21[1] < thresh
    elif isinstance(mf, scf.rhf.RHF):
        nocc = np.count_nonzero(mf.mo_occ)
        mo_coeff_occ = mf.mo_coeff[:, frozen:nocc]
        p12, p21 = check_rspan(lo_coeff_occ, s1e, mo_coeff_occ)
        span12 = p12 < thresh
        span21 = p21 < thresh
    else:
        raise TypeError(f'unsupported mean-field type: {type(mf)}')

    print(f'LO span the MO occ space - {span12}.\n'
          f'MO occ span the LO space - {span21}.')
    
    if not span12:
        raise ValueError(f"LO DO NOT SPAN MO occ, CHECK THE LOCALIZATION!!!"
                         f"the projection lost are {p12} {p21}")
    
    return None

def riao_localization(mf, lo_file=None):
    # IAO localization
    mol = mf.mol
    frozen = elements.chemcore(mol)
    orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
    lo_coeff = lo.iao.iao(mol, orbocc)
    lo_coeff = lo.orth.vec_lowdin(lo_coeff, mf.get_ovlp())
    moliao = lo.iao.reference_mol(mol)
    frag_lolist = autofrag_iao(moliao)
    if lo_file is not None:
        np.savez('./lo_coeff.npz', lo_coeff=lo_coeff)
    return lo_coeff, frag_lolist, moliao.elements

def uiao_localization(mf, lo_file=None):
    # IAO localization alpha beta separate
    mol = mf.mol
    frozen = elements.chemcore(mol)
    orbocc_a = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
    orbocc_b = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
    lo_coeff_a = lo.iao.iao(mol, orbocc_a)
    lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, mf.get_ovlp())
    lo_coeff_b = lo.iao.iao(mol, orbocc_b)
    lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, mf.get_ovlp())
    lo_coeff = [lo_coeff_a, lo_coeff_b]
    moliao = lo.iao.reference_mol(mol)
    frag_lolist = autofrag_iao(moliao)
    frag_lolist = [[i,i] for i in frag_lolist]
    if lo_file is not None:
        np.savez('./lo_coeff.npz', lo_coeff_a=lo_coeff_a,lo_coeff_b=lo_coeff_b)
    return lo_coeff, frag_lolist, moliao.elements

# def uiao_localization2(mf, lo_file=None):
#     # IAO localization alpha beta combined 
#     # NOTE the resulting IAOs doesn't necessarily
#     # span both alpha occ and beta occ!!!
#     mol = mf.mol
#     frozen = elements.chemcore(mol)
#     moliao = lo.iao.reference_mol(mol)

#     orbocca = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
#     orboccb = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
#     orboccc = np.hstack([orbocca, orboccb])

#     # svd combined orbitals
#     s = mf.get_ovlp()                       # AO overlap, (nao, nao)
#     G = orboccc.T @ s @ orboccc             # MO–MO Gram matrix, (n, n), symmetric PSD
#     w, V = np.linalg.eigh(G)                # eigh == SVD for symmetric PSD
#     w, V = w[::-1], V[:, ::-1]              # descending
#     # tol  = 1e-10 * w[0]                      # relative threshold
#     keep = w > 1e-6
#     print(f"union dimension = {keep.sum()} of {len(w)}")
#     # S-orthonormal orbitals spanning span(moa) ∪ span(mob)
#     orboccc_orth = orboccc @ (V[:, keep] / np.sqrt(w[keep]))
#     # sanity check: physically orthonormal
#     assert np.allclose(orboccc_orth.T @ s @ orboccc_orth, np.eye(keep.sum()), atol=1e-8)
#     iao_coeff = lo.iao.iao(mol, orboccc_orth)
#     iao_coeff = lo.orth.vec_lowdin(iao_coeff, mf.get_ovlp())
#     lo_coeff = [iao_coeff, iao_coeff]
#     #moliao = lo.iao.reference_mol(mol)
#     frag_lolist = autofrag_iao(moliao)
#     frag_lolist = [[i,i] for i in frag_lolist]

#     if lo_file is not None:
#         np.savez('./lo_coeff.npz', lo_coeff_a=lo_coeff[0],lo_coeff_b=lo_coeff[0])
      
#     return lo_coeff, frag_lolist, moliao.elements

def iao_localization(mf, lo_file=None):
    if isinstance(mf, scf.rhf.RHF):
        return riao_localization(mf, lo_file)
    elif isinstance(mf, scf.uhf.UHF):
        return uiao_localization(mf, lo_file)
    else:
        raise TypeError(f'unsupported mean-field type: {type(mf)}')


def get_lnoparam(mlno, lo_coeff, lno_thresh, lno_pct_occ, lno_norb, loidx, ifrag):

    if isinstance(mlno._scf, scf.rhf.RHF):
        orbloc = lo_coeff[:,loidx]
        lno_param = [{
            'thresh': lno_thresh[i],
            'pct_occ': lno_pct_occ[i],
            'norb': lno_norb[ifrag][i]
            } for i in [0,1]]
    
    elif isinstance(mlno._scf, scf.uhf.UHF):
        orbloc = [lo_coeff[0][:,loidx[0]], lo_coeff[1][:,loidx[1]]]
        lno_param = [
            [
                {
                    'thresh': (
                        lno_thresh[i][s] if isinstance(lno_thresh[i], Iterable)
                        else lno_thresh[i]
                    ),
                    'pct_occ': (
                        lno_pct_occ[i][s] if isinstance(lno_pct_occ[i], Iterable)
                        else lno_pct_occ[i]
                    ),
                    'norb': (
                        lno_norb[ifrag][i][s] if isinstance(lno_norb[ifrag][i], Iterable)
                        else lno_norb[ifrag][i]
                    ),
                } for i in [0, 1]
            ] for s in range(2)
        ]

    else:
        raise TypeError(f'unsupported mean-field type: {type(mlno._scf)}')
        
    return orbloc, lno_param


def lnomp2_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    # give canonicalized orbitals to MP2
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.rlnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        emp_frag = mod_lnoccsd.ulnomp2_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else: 
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    return emp_frag

def lnoccsd_kernel(mlno, lno_coeff, lno_frozen, uocc_loc, maskact, verbose=3):
    mf = mlno._scf
    if isinstance(mf, scf.rhf.RHF):
        mcc = lnoccsd.CCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.rlnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    elif isinstance(mf, scf.uhf.UHF):
        mcc = ulnoccsd.UCCSD(mf, mo_coeff=lno_coeff, frozen=lno_frozen).set(verbose=verbose)
        mcc.conv_tol = 1e-6
        mcc.conv_tol_normt = 3e-5
        ecc_frag, t1, t2 = \
            mod_lnoccsd.ulnoccsd_solver(mcc, lno_coeff, uocc_loc, mlno.mo_occ, maskact)
    else: 
        raise NotImplementedError('LNO Only Support Restricted and Unrestricted Orbitals!')
    return ecc_frag, t1, t2