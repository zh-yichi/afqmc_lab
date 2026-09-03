options =  {'n_blocks': 50,
            'n_walkers': 100,
            'max_memory': 3000,
            'seed': 27,
            'trial': 'upt2ccsd_bar',
            'mix_precision': False,
            }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)


options = {'n_blocks': 50,
               'n_walkers': 100,
               'max_memory': 3000,
               'seed': 27,
               'mix_precision': False,  # exact arithmetic, so any difference
                                        # from 'upt2ccsd' is the sampling
               'trial': 'upt2ccsd_sto_chol',
               # --- semistochastic Cholesky sum -------------------------- #
               'chol_cost_ratio': 0.2,  # each walker touches 20% of nchol,
                                        # split 3:1 head:samples
               # 'head_chol_ratio': 0.125,  # override the head half
               # 'n_chol_samples' : 32,     # override the tail half
               # 'n_chol_head'    : 'full', # no sampling; reproduces upt2ccsd
               }

from afqmc import launch_afqmc
launch_afqmc.ph_afqmc(options)
