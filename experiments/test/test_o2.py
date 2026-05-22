import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

d = 100

o2 = '''
O 0.0 0.0 0.0
O 0.0 0.0 1.20577
'''

m_list = [1]
for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'O {0.0+shift} 0.0 0.0     \n'
        atoms += f'O {0.0+shift} 0.0 1.20577 \n'

    # nfrozen = 2*nc
    spin = 2*nc
    mol = gto.M(atom=atoms, basis="6-31g", spin=spin, verbose=4)
    mol.build()

    mf = scf.UHF(mol)
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
    print('number of O2:', nc)
    print('max t1a: ', np.abs(mycc.t1[0]).max())
    print('max t1b: ', np.abs(mycc.t1[1]).max())
    print('max t2aa: ', np.abs(mycc.t2[0]).max())
    print('max t2ab: ', np.abs(mycc.t2[1]).max())
    print('max t2bb: ', np.abs(mycc.t2[2]).max())

    options = {'eql_time': 10,
               'n_prop_steps': 50,
               'n_blocks': 300,
               'n_walkers': 300,
               'max_error': 0.0,
               'nchol_chunk': 30,
               'max_memory': 10000,
               'mix_precision': False,
               'seed': 1,
               'walker_type': 'uhf',
               'trial': 'upt2ccsd',
               'dt': 0.005,
               'free_projection': False,
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc,chol_cut=1e-7)
    launch_afqmc.run_afqmc(options)

    options = {'eql_time': 10,
               'n_prop_steps': 50,
               'n_blocks': 300,
               'n_walkers': 300,
               'max_error': 0.0,
               'mix_precision': False,
               'seed': 1,
               'walker_type': 'uhf',
               'trial': 'upt2ccsd_ad',
               'dt': 0.005,
               'free_projection': False,
               }

   # launch_afqmc.run_afqmc(options)

