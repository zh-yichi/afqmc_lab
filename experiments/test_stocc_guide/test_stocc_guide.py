from afqmc import launch_afqmc

options = {'eql_time': 20,
            'n_blocks': 100,
            'n_walkers': 200,
            'max_memory': 3000,
            'seed': 17,
            'walker_type': 'rhf',
            'guide': 'rstoccsd',
            'trial': 'rhf',
            'mix_precision': True,
            't2_thresh': 1e-3,
            'n_slater': 10,
            }

launch_afqmc.ph_afqmc(options, script='run_afqmc_exp.py')


options = {'eql_time': 10,
            'n_blocks': 100,
            'n_walkers': 200,
            'max_memory': 3000,
            'seed': 17,
            'walker_type': 'rhf',
            'guide': 'rcisd',
            'trial': 'rhf',
            'mix_precision': True,
            }

# launch_afqmc.ph_afqmc(options, script='run_afqmc_exp.py')

