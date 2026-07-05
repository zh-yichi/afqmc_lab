import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

from afqmc.corr_sample import integral, launch_afqmc

####  test monomers ####
a = 1.20577 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'A' # unit of length
na = 2 # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
spin = 2 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'O'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol1 = gto.M(atom=atoms,
            basis=basis,
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=spin*nc,
            max_memory=40000,
            )

mf1 = scf.UHF(mol1).density_fit()
mf1.kernel()

stable = False
while not stable:
    print(f'mean-field stability test')
    if not stable:
        mo_i, _, stable,_ = mf1.stability(return_status=True)
        dm = mf1.make_rdm1(mo_i,mf1.mo_occ)
        mf1.kernel(dm0=dm)
    elif stable:
        print(f'HF Energy: {mf1.e_tot}, stability {stable}')
        break


# mycc1 = cc.CCSD(mf1)
# mycc1.set_frozen()
# mycc1.kernel()

mol2 = gto.M(atom=atoms,
            basis=basis,
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=spin*nc,
            max_memory=40000,
            )

mf2 = scf.UHF(mol2).density_fit()
mf2.kernel()

stable = False
while not stable:
    print(f'mean-field stability test')
    if not stable:
        mo_i, _, stable,_ = mf2.stability(return_status=True)
        dm = mf2.make_rdm1(mo_i,mf2.mo_occ)
        mf2.kernel(dm0=dm)
    elif stable:
        print(f'HF Energy: {mf2.e_tot}, stability {stable}')
        break

print(f"mf1 energy = {mf1.e_tot:.8f} | mf2 energy = {mf2.e_tot:.8f}")
print("before Procruste")
print("Alpha Norm : ", np.linalg.norm(mf1.mo_coeff[0] - mf2.mo_coeff[0]))
print(" Beta Norm : ", np.linalg.norm(mf1.mo_coeff[1] - mf2.mo_coeff[1]))
mf2.mo_coeff = integral.match_mo(mf1, mf2)
print("Alpha Norm : ", np.linalg.norm(mf1.mo_coeff[0] - mf2.mo_coeff[0]))
print(" Beta Norm : ", np.linalg.norm(mf1.mo_coeff[1] - mf2.mo_coeff[1]))



# mf2.kernel()

# mycc2 = cc.CCSD(mf2)
# mycc2.set_frozen()
# mycc2.kernel()

# options =  {'eql_time': 50,
#             'n_prop_step': 30,
#             'n_blocks': 20,
#             'n_walkers': 300,
#             'nchol_chunk': 30,
#             'max_memory': 3000,
#             'seed': 17,
#             'trial': 'uhf',
#             'mix_precision': False,
#             }

# from afqmc.corr_sample import integral, launch_afqmc
# integral.prep_integral(mycc1, mycc2, chol_cut=1e-5)
# launch_afqmc.ph_afqmc(options)
# os.system('mv afqmc.out cs_afqmc.out')


