from afqmc import integral, launch_afqmc

# options = {
#         'eql_time': 40,
#         'n_blocks': 300,
#         'n_walkers': 300,
#         'max_memory': 3000,
#         'seed': 17,
#         'walker_type': 'uhf',
#         'trial': 'uhf',
#         'mix_precision': True,
#         }

# launch_afqmc.ph_afqmc(options)

options = {
        'eql_time': 40,
        'n_blocks': 300,
        'n_walkers': 300,
        'max_memory': 3000,
        'seed': 17,
        'walker_type': 'rhf',
        'trial': 'cisd',
        'mix_precision': True,
        }

launch_afqmc.ph_afqmc(options)

t2_thresh = 1e-4

options = {
        'eql_time': 40,
        'n_blocks': 300,
        'n_walkers': 300,
        'max_memory': 3000,
        'seed': 17,
        'walker_type': 'rhf',
        'trial': 'stocc_pt2',
        't2_thresh': t2_thresh,
        'n_slater': 100,
        'mix_precision': True,
        }

# launch_afqmc.ph_afqmc(options)
