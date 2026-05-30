import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

options = {'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 3000,
           'mix_precision': True,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

options = {'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 3000,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

options = {'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 3000,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd_ad',
           }

#from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

