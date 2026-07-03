options = {'n_prop_steps': 50,
           'eql_time': 20,
           'n_blocks': 100,
           'n_walkers': 10,
           'mix_precision': True,
           'max_memory': 1000,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

