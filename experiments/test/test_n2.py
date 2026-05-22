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
        atoms += f'N {0.0+shift} 0.0 1.20577 \n'

    mol = gto.M(atom=atoms, basis="sto6g", verbose=4)
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.set_frozen()
    mycc.kernel()
    
    options = {'n_eql': 160,
               'n_prop_steps': 50,
               'n_blocks': 300,
               'n_walkers': 300,
               'max_error': 0.0,
               'mix_precision': False,
               'seed': 1,
               'walker_type': 'rhf',
               'trial': 'ccsd_pt2',
               'dt': 0.005,
               'use_gpu': True,
               'free_projection': False,
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc, chol_cut=1e-7)
    launch_afqmc.run_afqmc(options)
