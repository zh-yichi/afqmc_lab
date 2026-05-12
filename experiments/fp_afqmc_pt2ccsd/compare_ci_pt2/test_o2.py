import os
import jax
jax.config.update("jax_enable_x64", True)
from pyscf import gto, scf, cc

a = 1.20577 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'A' # unit of length
na = 2 # size of a cluster (monomer)
nc_list = [3] # set as integer multiple of monomers
spin = 2 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'O'
basis = 'sto6g'
for nc in nc_list:
    atoms = ""
    for n in range(nc*na):
        shift = ((n - n % na) // na) * (d-a)
        atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

    mol = gto.M(atom=atoms,basis=basis,spin=spin*nc,unit=unit,verbose=4)
    mol.build()

    mf = scf.UHF(mol)
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

    mycc = cc.CCSD(mf,frozen=2*nc)
    mycc.kernel()

    options = {
           'n_eql_blocks': 10,
           'n_prop_steps': 50,
           'n_trj': 300,
           'n_walkers': 300,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'ucisd',
           'dt': 0.01,
           'free_projection': True,
           'use_gpu': True,
           }

    from afqmc import prep, launch_afqmc
    prep.prep_afqmc(mycc,chol_cut=1e-5)
    launch_afqmc.run_afqmc(options)
    os.system(f'mv afqmc.out fp_{nc}.out')


    options = {
           'n_eql_blocks': 10,
           'n_prop_steps': 50,
           'n_trj': 300,
           'n_walkers': 300,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'uccsd_pt2',
           'dt': 0.01,
           'free_projection': True,
           'use_gpu': True,
           }

    # from afqmc import prep, launch_afqmc
    # prep.prep_afqmc(mycc,chol_cut=1e-5)
    launch_afqmc.run_afqmc(options)
    os.system(f'mv afqmc.out fp_pt2_{nc}.out')
