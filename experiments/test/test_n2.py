import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

d = 100

m_list = [1]
for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'N {0.0+shift} 0.0 0.0     \n'
        atoms += f'N {0.0+shift} 0.0 2.0 \n'

    # nfrozen = 2*nc
    spin = 0
    mol = gto.M(atom=atoms, 
                unit='b',
                basis="ccpvdz", 
                spin=spin, 
                verbose=4)
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    stable = False
    for i in range(10):
        print(f'mf stability test {i+1}')
        if not stable:
            mo_i, _, stable,_ = mf.stability(return_status=True)
            dm = mf.make_rdm1(mo_i,mf.mo_occ)
            mf.kernel(dm0=dm)
        elif stable:
            print(f'mf energy: {mf.e_tot}, stability {stable}')
            break

    mycc = cc.CCSD(mf)
    mycc.set_frozen()
    mycc.kernel()

    options = {'n_blocks': 300,
               'n_walkers': 300,
               'nchol_chunk': 30,
               'max_memory': 3000,
               'seed': 17,
               'walker_type': 'rhf',
               'trial': 'pt2ccsd',
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc)
    launch_afqmc.ph_afqmc(options)
