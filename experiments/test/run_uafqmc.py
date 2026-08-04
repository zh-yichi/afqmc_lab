import jax
jax.config.update("jax_enable_x64", True)

options = {'eql_time': 20,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_ad',
           }

from afqmc import launch_afqmc
script2run = None
#launch_afqmc.ph_afqmc(options, script=script2run)

options = {'eql_time': 20,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_bar',
           }

script2run = None
#launch_afqmc.ph_afqmc(options, script=script2run)


options = {'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'upt2ccsd_wrong',
           }

from afqmc import launch_afqmc
script2run = None
# launch_afqmc.ph_afqmc(options, script=script2run)

options = {'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'uhf_wrong',
           }

script2run = None
# launch_afqmc.ph_afqmc(options, script=script2run)

options = {'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'guide': 'uhf',
           'trial': 'uhf',
           }

script2run = None
launch_afqmc.ph_afqmc(options, script=script2run)

