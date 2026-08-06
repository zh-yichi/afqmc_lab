import jax
jax.config.update("jax_enable_x64", True)

from afqmc import launch_afqmc

options = {'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_red',
           't2_thresh': 1e-1,
           }

script2run = None
launch_afqmc.ph_afqmc(options, script=script2run)

options = {'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_bar',
           }

script2run = None
#launch_afqmc.ph_afqmc(options, script=script2run)

