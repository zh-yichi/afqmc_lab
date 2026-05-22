import numpy as np
from pyscf import gto, scf, lo

lno_num = 1
lno_list = [3e-4,1e-4,3e-5,1e-5]
lno_thresh = lno_list[lno_num-1]

####  test H2 monomers ####
a = 2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 10 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
unit = 'B'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol = gto.M(atom=atoms,
            basis="ccpvdz",
            verbose=4,
            unit='angstrom',
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=40000,
            )

mf = scf.UHF(mol).density_fit()
# mf.chkfile = '../../../pyscf/c3/mf.chk'
# mf.init_guess = 'chk'
mf.kernel()

from pyscf.lno.tools import autofrag_iao
from pyscf import lo
import numpy as np
from pyscf.data import elements

frozen = elements.chemcore(mol)

# IAO localization
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

# np.savez('../lo_coeff.npz', lo_coeff=lo_coeff)

from afqmc.lno_afqmc import lno_afqmc
options = {
           'n_eql': 10,
           'n_prop_steps': 50,
           'n_blocks': 50,
           'n_walkers': 300,
           'nchol_chunk': 120,
           'mix_precision': False,
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
              nfrozen = frozen,
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-6,
              target_sto_error = 5e-4,
              run_frg_list = [0],
              atom_group = moliao.elements,
              )
