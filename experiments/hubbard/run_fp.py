import numpy as np
import hubbard

NELEC = (1, 0)
NSITE = 2
T     = 1
U     = 4

h1, eri, mol, mf = hubbard.hubbard_mf(
    NSITE, T, U, NELEC, dm0=None, pbc=True, verbose=4)

# mo_a, mo_b = mf.mo_coeff[0][:, :NELEC[0]], mf.mo_coeff[1][:, :NELEC[1]]

mo_a = np.array([[1.0], [0.0]]) # electron pinned to site 0
mo_b = np.zeros((NSITE, 0))

print(mo_a)
print(mo_b)

qmc = hubbard.afqmc_setup(
    h1, U, NELEC, mo_a, mo_b, NSITE)

qmc_result = hubbard.run_free_projection(
                qmc, dt=0.005, n_walkers=500, 
                n_prop_steps=20, n_blocks=80,
                n_traj=200, seed=0, n_exp_terms=10,
                snap_every=10, verbose=4
                )