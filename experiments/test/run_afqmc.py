import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

options = {'eql_time': 20,
           'n_blocks': 100,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 8000,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_eff',
           }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

options = {'eql_time': 20,
           'n_blocks': 100,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 8000,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

#from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)
