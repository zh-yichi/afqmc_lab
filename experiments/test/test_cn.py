import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np
from scipy.linalg import block_diag


d = 100

m_list = [1,3,5,7,9,11,13,15]

dm_mono = None
e_mono = None

for nc in m_list:
    atoms = ""
    for n in range(nc):
        shift = n*d
        atoms += f'C {0.0+shift} 0.00000 0.00000 \n'
        atoms += f'N {0.0+shift} 0.00000 1.16739 \n'

    spin = 1 * nc
    mol = gto.M(atom=atoms, 
                unit='A',
                basis="sto6g", 
                spin=spin, 
                verbose=4)
    mol.build()

    mf = scf.UHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-10
    if dm_mono is None:
        mf.kernel()
    else:
        dm0 = np.array([block_diag(*([dm_mono[0]] * nc)),
                        block_diag(*([dm_mono[1]] * nc))])
        mf.kernel(dm0=dm0)

    skip_stability = (e_mono is not None
                      and mf.converged
                      and abs(mf.e_tot - nc * e_mono) < 1e-6 * nc)
    if skip_stability:
        print(f'mf energy: {mf.e_tot}, matches {nc} x monomer '
              f'({nc * e_mono}), skipping stability analysis')
    else:
        stable = False
        for i in range(10):
            print(f'mf stability test {i+1}')
            if not stable:
                mo_i, _, stable,_ = mf.stability(return_status=True)
                dm = mf.make_rdm1(mo_i,mf.mo_occ)
                mf = mf.newton()
                mf.kernel(dm0=dm)
            elif stable:
                print(f'mf energy: {mf.e_tot}, stability {stable}')
                break

    if nc == 1:
        dm_mono = np.asarray(mf.make_rdm1())
        e_mono = mf.e_tot

    print(f'nc = {nc}: E_scf = {mf.e_tot}, '
          f'{nc} * E_scf(1) = {nc * e_mono}, '
          f'diff = {mf.e_tot - nc * e_mono}')

    mycc = cc.CCSD(mf).set_frozen()
    mycc.conv_tol = 1e-6
    mycc.conv_tol_normt = 1e-5
    mycc.kernel()

    # --- swap with your AFQMC ---
    # options =  {'eql_time': 50,
    #             'n_blocks': 600,
    #             'n_walkers': 300,
    #             'max_memory': 10000,
    #             'seed': 27,
    #             'trial': 'upt2ccsd_cisd',
    #             'mix_precision': False,
    #             }

    # from afqmc import integral, launch_afqmc
    # integral.prep_integral(mycc, chol_cut=1e-5)
    # launch_afqmc.ph_afqmc(options)
    # os.system(f"mv afqmc.out cisd_pt2.out{nc}")

