from afqmc.lno_afqmc import lno_afqmc_test

options = {
           'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 5,
           'mix_precision': False,
           'frozen_vir': 30,
           'n_corr_blocks': 50,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

script = 'run_lno_afqmc_pt2ccsd_frozen_vir.py'
lno_afqmc_test.run_lnoafqmc(options, script=script)
