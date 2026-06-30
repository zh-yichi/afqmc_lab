import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

####  test H2 monomers ####
a = 2.2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 1 # set as integer multiple of monomers
spin = 2 # spin per monomer
frozen = 0 # frozen orbital per monomer
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


mycc = cc.CCSD(mf)
mycc.set_frozen()
mycc.kernel()

options = {'eql_time': 10,
            'n_blocks': 50,
            'n_walkers': 10,
            'max_error': 0.0,
            'nchol_chunk': 30,
            'max_memory': 3000,
            'seed': 1,
            'walker_type': 'uhf',
            'trial': 'upt2ccsd',
            'mix_precision': False,
            }

from afqmc import integral, launch_afqmc
integral.prep_integral(mycc, chol_cut=1e-5)
launch_afqmc.ph_afqmc(options)
