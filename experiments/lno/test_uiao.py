import numpy as np
from pyscf import gto, scf, lo, mp, cc
mol = gto.Mole()
mol.verbose = 4
mol.atom = '''
O   -1.485163346097   -0.114724564047    0.000000000000
H   -1.868415346097    0.762298435953    0.000000000000
H   -0.533833346097    0.040507435953    0.000000000000
O    1.416468653903    0.111264435953    0.000000000000
H    1.746241653903   -0.373945564047   -0.758561000000
# H    1.746241653903   -0.373945564047    0.758561000000
'''
mol.basis = 'cc-pvdz'
mol.precision = 1e-10
mol.spin = 3
mol.build()
mf = scf.UHF(mol).density_fit()
mf = mf.newton()
mf.kernel()

frozen = 2
mymp = mp.MP2(mf, frozen=frozen)
mymp.kernel()
efull_mp2 = mymp.e_corr
print(f'MP2 Corr = {efull_mp2:.8f}')

mycc = cc.CCSD(mf, frozen=frozen)
mycc.kernel()
efull_ccsd = mycc.e_corr
print(f'CCSD Corr = {efull_ccsd:.8f}')

efull_t = mycc.ccsd_t()
efull_ccsd_t = efull_ccsd + efull_t
print(f'CCSD(T) Corr = {efull_ccsd_t:.8f}')

from pyscf.lno.tools import autofrag_iao

# def test_lno_iao_by_thresh(self):
mol = mycc.mol
mf = mycc._scf
frozen = mycc.frozen

# IAO localization
orbocc_a = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
orbocc_b = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
lo_coeff_a = lo.iao.iao(mol, orbocc_a)
lo_coeff_b = lo.iao.iao(mol, orbocc_b)
lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, mf.get_ovlp())
lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, mf.get_ovlp())
lo_coeff = [lo_coeff_a, lo_coeff_b]
moliao = lo.iao.reference_mol(mol)

frag_lolist = autofrag_iao(moliao)
frag_lolist = [[frag,frag] for frag in frag_lolist]

from afqmc.lno_afqmc import lno_afqmc
options = {
           'n_eql': 3,
           'n_prop_steps': 50,
           'n_blocks': 100,
           'n_walkers': 100,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'uccsd_pt2',
           'dt':0.005,
           'use_gpu': True,
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = 2,
              thresh = 1e-6,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 1e-4,
              atom_group = moliao.elements,
              )
