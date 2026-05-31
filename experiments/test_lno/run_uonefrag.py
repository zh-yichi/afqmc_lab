from afqmc.lno_afqmc import lno_afqmc
options = {
           'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

lno_afqmc.run_lnoafqmc(options)

options = {
           'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_ad',
           }

lno_afqmc.run_lnoafqmc(options)
