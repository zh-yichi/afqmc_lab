import numpy as np
from pyscf import gto, scf, cc, lib

atomstring = '''
Fe -0.64257176050830 0.51803920561668 0.11515259480577
O 0.11174687498777 2.15027362958097 -0.78003307659188
H 0.70766775417542 2.71590643817051 -0.25346860209865
O -1.31193844940493 -0.30254437211926 -1.59153974321848
H -1.93648366806763 0.13093135988795 -2.20090760771949
O 0.02720918798052 1.33940572113505 1.82122676893196
H 0.65258765622944 0.90686870076571 2.43040311553976
O -1.39648053530566 -1.11444807905756 1.01023916292248
H -1.99331392681300 -1.67940484230818 0.48397490889195
O -2.28682023617329 1.63340828371306 0.41005372492049
H -2.35377021104600 2.47291065207153 -0.08301808819513
O 1.00194372931180 -0.59712335622834 -0.17950165163229
H 1.06807766676477 -1.43758842942053 0.31202545851205
H -3.18922098471738 1.27280247171927 0.48036417156888
H -0.66521991563992 -0.78352006296416 -2.14179468111308
H 1.90454646424149 -0.23678289755293 -0.24845043594753
H 0.43776650602076 2.17113603799071 -1.69794346797289
H -0.61875604533523 1.82123954705219 2.37159866149671
H -1.72126010670064 -1.13595000805262 1.92856878689935
'''

mol = gto.M(atom = atomstring,
            basis = {
                'default': 'sto6g',
                'Fe': 'sto6g'
                },
            verbose = 4,
            unit = 'angstrom',
            symmetry = 0,
            charge = 2,
            spin = 0,
            max_memory = 20000,
            )

mf = scf.UHF(mol).density_fit()
mf = mf.x2c()
mf.chkfile = 'lsmf.chk'
mf.init_guess = 'chk'
# dm0 = mf.from_chk('./hsmf.chk')
mf.max_cycle = 100
mf.level_shift = 0.5
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

# mycc = cc.CCSD(mf)
# mycc.set_frozen()
# mycc.level_shift = 0.5
# mycc.conv_tol = 1e-6
# mycc.conv_tol_normt = 1e-5
# mycc.max_cycle = 100
# mycc.kernel()

# options = {'n_prop_steps': 50,
#            'eql_time': 20,
#            'n_blocks': 100,
#            'n_walkers': 10,
#            'mix_precision': True,
#            'max_memory': 1000,
#            'seed': 17,
#            'walker_type': 'uhf',
#            'trial': 'upt2ccsd_bar',
#            }

# from afqmc import integral, launch_afqmc
# integral.prep_integral(mycc, chol_cut=1e-5)
#launch_afqmc.ph_afqmc(options)
