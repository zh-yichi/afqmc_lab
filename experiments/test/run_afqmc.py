import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

options = {'eql_time': 80,
           'n_blocks': 1000,
           'n_walkers': 300,
           'max_error': 0.0,
           'nchol_chunk': 30,
           'max_memory': 8000,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

from afqmc import launch_afqmc
script2run = "run_afqmc_pt2ccsd_new.py"
launch_afqmc.ph_afqmc(options, script=script2run)
