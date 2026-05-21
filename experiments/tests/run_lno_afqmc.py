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
basis = 'ccpvdz'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol = gto.M(atom=atoms,
            basis="ccpvdz",
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=40000,
            )

mf = scf.RHF(mol).density_fit()
# mf.chkfile = '../../../pyscf/c3/mf.chk'
# mf.init_guess = 'chk'
mf.kernel()

from pyscf.lno.tools import autofrag_iao
from pyscf import lo
import numpy as np
from pyscf.data import elements

frozen = elements.chemcore(mol)

# IAO localization
orbocc = mf.mo_coeff[:,frozen:np.count_nonzero(mf.mo_occ)]
lo_coeff = lo.iao.iao(mol, orbocc)
lo_coeff = lo.orth.vec_lowdin(lo_coeff, mf.get_ovlp())
moliao = lo.iao.reference_mol(mol)
frag_lolist = autofrag_iao(moliao)

np.savez('../lo_coeff.npz', lo_coeff=lo_coeff)

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
           'walker_type': 'rhf',
           'trial': 'ccsd_pt2',
           'dt':0.005,
           'use_gpu': False,
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = frozen,
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 5e-4,
              run_frg_list = [0],
              atom_group = moliao.elements,
              )
