import numpy as np
from pyscf import gto, scf, cc, lib

import jax
jax.config.update("jax_enable_x64", True)

atomstring = '''
Fe -0.64147387529051 0.51990405379180 0.11450483185168
O 0.33915564253394 2.18819453520299 -0.84159903476570
H 0.65823736209818 2.99799985286513 -0.40575899522591
O -1.29914529040912 -0.20924466129350 -1.80322550106302
H -2.11419692217693 0.03267722517314 -2.27701147100162
O 0.01237770373627 1.24396717111997 2.03533113813669
H 0.83097217272839 1.00756306178418 2.50577204385612
O -1.62284672685044 -1.15048542622041 1.07007991103536
H -1.94592790838151 -1.95766677806096 0.63237073377438
O -2.41866907277644 1.66373032346654 0.25658821238371
H -2.55381601393967 2.56024091790599 -0.09930153080408
O 1.13404022498401 -0.62623224778001 -0.02436092147139
H 1.26370551457750 -1.52400168391753 0.33034524187694
H -3.28110369794222 1.35873477131317 0.59094893393907
H -0.81032015579443 -0.82248643308581 -2.37966496653734
H 1.99795010138314 -0.32692539271158 -0.36001715946551
H 0.57998322602479 2.26642229799559 -1.78148703231078
H -0.47668197075682 1.85673194452426 2.61208351578631
H -1.85653031374812 -1.23356353207299 2.011352050005
'''

mol = gto.M(atom = atomstring,
            basis = {
                'default': 'sto6g',
                'Fe': 'sto6g'
                },
            verbose=4,
            unit='angstrom',
            symmetry=0,
            charge=2,
            spin=4,
            max_memory=10000,
            )

mf = scf.UHF(mol).density_fit()
mf = mf.x2c()
# dm0 = mf.from_chk('../../tz_dz/mf.chk')
mf.chkfile = './mf.chk'
mf.init_guess = 'chk'
#mf.with_df._cderi_to_save = '/projects/bdka/yzhang65/scratch/pyscf/fehydro_hs_tz_cderi.h5'
mf.level_shift = 0.5
mf.max_cycle = 100
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
mycc.conv_tol = 1e-6
mycc.conv_tol_normt = 1e-5
mycc.kernel()

options = {'n_prop_steps': 1,
           'eql_time': 50,
           'n_blocks': 100,
           'n_walkers': 10,
           'mix_precision': True,
           'max_memory': 1000,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_bar',
           }

from afqmc import integral, launch_afqmc
integral.prep_integral(mycc, chol_cut=1e-5)
launch_afqmc.ph_afqmc(options)
