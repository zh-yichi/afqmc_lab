import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import os
import numpy as np

d = 2 # distance between each cluster
unit = 'bohr' # unit of length
na = 2  # size of a cluster (monomer)
nc = 10 # set as integer multiple of monomers
spin = 0 # spin per monomer
#frozen = 0 # frozen orbital per monomer
elmt = 'H'
bond_list = [2]
for a in bond_list:
    atoms = ""
    for n in range(nc*na):
        shift = ((n - n % na) // na) * (d-a)
        atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

    mol = gto.M(atom=atoms, basis="sto6g",spin=spin*nc, unit=unit, verbose=4)
    mol.build()

    mf = scf.UHF(mol)#.density_fit()
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
    
    options = {'eql_time': 10,
               'n_blocks': 100,
               'n_walkers': 300,
               'max_error': 0.0,
               'mix_precision': False,
               'seed': 17,
               'walker_type': 'uhf',
               'trial': 'upt2ccsd',
               'free_projection': False,
               }

    from afqmc import integral, launch_afqmc
    integral.prep_integral(mycc, chol_cut=1e-5)
    launch_afqmc.ph_afqmc(options)
