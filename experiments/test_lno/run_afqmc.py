import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

from pyscf import gto, scf, cc
import numpy as np

####  test H2 monomers ####
a = 2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc_list = [1] # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
unit = 'B'
basis = 'sto6g'
for nc in nc_list:
    atoms = ""
    for n in range(nc*na):
        shift = ((n - n % na) // na) * (d-a)
        atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

    mol = gto.M(atom=atoms, basis="sto6g", spin=spin*nc, unit=unit, verbose=4)
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    # scf stability
stable = False
while not stable:
    print(f'mf stability test')
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

    from afqmc import prep, launch_afqmc
    prep.prep_afqmc(mycc, chol_cut=1e-5)

    # RHF Trial
    options = {
               'n_eql': 10, # 1 eql step = dt*n_prop_steps (here 160 = 40 au)
               'n_blocks': 50, # tune this for how many samples you want
               'n_walkers': 300,
               'dt':0.005, # time every trot step
               'max_error': 0.0, # set to 0 to run the calculation till n_blocks
               'seed': 17,
               'walker_type': 'rhf',
               'trial': 'ccsd_pt2',
               'free_projection': False,
               'use_gpu': False
               }

    launch_afqmc.run_afqmc(options)
