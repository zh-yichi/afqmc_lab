import numpy as np
from pyscf import gto, scf, lo

lno_num = 4
lno_list = [3e-4,1e-4,3e-5,1e-6]
lno_thresh = lno_list[lno_num-1]

####  test H2 monomers ####
a = 2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 5 # set as integer multiple of monomers
spin = 2 # spin per monomer
elmt = 'O'
unit = 'B'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol = gto.M(atom=atoms,
            basis="sto6g",
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=spin*nc,
            max_memory=40000,
            )

mf = scf.UHF(mol).density_fit()
mf.kernel()

stable = False
while not stable:
    print(f'mean-field stability test')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'UHF Energy: {mf.e_tot}, stability {stable}')
        break

from afqmc.lno_afqmc import lno_afqmc, tools
from pyscf.data import elements
lo_coeff, frag_lolist, atm_center = tools.iao_localization(mf)


from afqmc.lno_afqmc import lno_afqmc
options = {
           'n_prop_steps': 50,
           'n_blocks': 600,
           'n_walkers': 300,
           'max_memory': 2000,
           'mix_precision': False,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = elements.chemcore(mol),
              thresh = lno_thresh,
              qmc_options = options,
              target_sto_error = 1e-6,
              run_frag_list = [0,1],
              atom_group = atm_center,
              )
