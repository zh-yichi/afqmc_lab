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
    mol = gto.M(atom=atoms, basis="sto6g", spin=spin, verbose=4)
    mol.build()

    mf = scf.UHF(mol)#.density_fit()
    mf.kernel()
    
    stable = False
    while not stable:
        print(f'mean-field stability test')
        if not stable:
            mo_i, _, stable,_ = mf.stability(return_status=True)
            dm = mf.make_rdm1(mo_i,mf.mo_occ)
            mf.kernel(dm0=dm)
        elif stable:
            print(f'UHF Energy: {umf.e_tot}, stability {stable}')
            break


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
               'walker_type': 'uhf',
               'trial': 'upt2ccsd',
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc)
    launch_afqmc.ph_afqmc(options)
