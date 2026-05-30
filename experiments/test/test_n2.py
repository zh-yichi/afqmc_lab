import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

d = 100

o2 = '''
N 0.0 0.0 0.0
N 0.0 0.0 1.20577
'''

m_list = [1]
for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'O {0.0+shift} 0.0 0.0     \n'
        atoms += f'O {0.0+shift} 0.0 1.20577 \n'

    # nfrozen = 2*nc
    spin = 0
    mol = gto.M(atom=atoms, basis="ccpvdz", spin=spin, verbose=4)
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mo1 = mf.stability()[0]
    mf = mf.newton().run(mo1, mf.mo_occ)
    mf.stability()

    mycc = cc.CCSD(mf)
    mycc.set_frozen()
    mycc.kernel()

    options = {'eql_time': 10,
               'n_blocks': 100,
               'n_walkers': 300,
               'max_error': 0.0,
               'nchol_chunk': 30,
               'max_memory': 3000,
               'seed': 17,
               'walker_type': 'rhf',
               'trial': 'pt2ccsd_ad',
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc)
    #launch_afqmc.ph_afqmc(options)

    options = {'eql_time': 10,
               'n_blocks': 100,
               'n_walkers': 300,
               'max_error': 0.0,
               'mix_precision': False,
               'nchol_chunk': 30,
               'max_memory': 3000,
               'seed': 17,
               'walker_type': 'rhf',
               'trial': 'pt2ccsd',
               }
    #launch_afqmc.ph_afqmc(options)
