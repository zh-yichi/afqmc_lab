import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

atoms = '''
C          0.000000       0.000000       0.620000
C          0.000000       0.000000      -0.620000
'''

mol = gto.M(
        atom=atoms,
        basis="ccpvdz",
        spin=0,
        max_memory=40000,
        verbose=4
        )
mol.build()


mf = scf.UHF(mol)
mf.level_shift = 0.4
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

#et = mycc.ccsd_t()
#print(f"CCSD Energy = {mycc.e_tot+et}")

options =  {'n_blocks': 100,
            'n_walkers': 300,
            'max_memory': 8000,
            'seed': 17,
            'trial': 'upt2ccsd_wrong',
            'mix_precision': False,
            }

from afqmc import integral, launch_afqmc
integral.prep_integral(mycc, chol_cut=1e-5)
launch_afqmc.ph_afqmc(options)
