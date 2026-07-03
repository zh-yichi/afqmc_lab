options =  {'n_blocks': 500,
            'n_walkers': 300,
            'nchol_chunk': 30,
            'max_memory': 3000,
            'seed': 17,
            'trial': 'pt2ccsd_bar',
            'mix_precision': False,
            }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)

