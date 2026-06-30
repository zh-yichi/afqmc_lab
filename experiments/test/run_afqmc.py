# import jax
# jax.config.update("jax_enable_x64", True)

options = {'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 200,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd',
           }

from afqmc import launch_afqmc
script2run = None
launch_afqmc.ph_afqmc(options, script=script2run)

options = {'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 200,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_bar',
           }

from afqmc import launch_afqmc
script2run = None
launch_afqmc.ph_afqmc(options, script=script2run)

# script2run = "run_afqmc_exp.py"
# launch_afqmc.ph_afqmc(options, script=script2run)
