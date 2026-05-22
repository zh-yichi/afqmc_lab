import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

options = {'n_eql': 10,
               'n_prop_steps': 50,
               'n_blocks': 100,
               'n_walkers': 300,
               'max_error': 0.0,
               'nchol_chunk': 30,
               'max_memory': 8000,
               'mix_precision': False,
               'seed': 1,
               'walker_type': 'uhf',
               'trial': 'upt2ccsd',
               'dt': 0.005,
               'use_gpu': True,
               'free_projection': False,
               }

from afqmc import launch_afqmc
launch_afqmc.run_afqmc(options)
