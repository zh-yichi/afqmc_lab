from afqmc.lno_afqmc import lno_afqmc

options = {'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 300,
           'max_memory': 1000,
           'mix_precision': False,
           'n_batch': 1,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           'dt':0.005,
           }

lno_afqmc.run_lnoafqmc(options)

# options = {'eql_time': 10,
#            'n_blocks': 50,
#            'n_walkers': 300,
#            'max_memory': 1000,
#            'mix_precision': False,
#            'n_batch': 1,
#            'seed': 17,
#            'walker_type': 'uhf',
#            'trial': 'upt2ccsd_ad',
#            'dt':0.005,
#            }

# lno_afqmc.run_lnoafqmc(options)

