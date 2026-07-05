options =  {'n_blocks': 600,
            'n_walkers': 300,
            'nchol_chunk': 30,
            'max_memory': 3000,
            'seed': 17,
            'trial': 'upt2ccsd',
            'mix_precision': False,
            }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

