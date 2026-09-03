from afqmc.lno_afqmc import lno_afqmc_test

options = {
           'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 10,
           'mix_precision': True,
           'n_corr_blocks': 50,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

script = None
#lno_afqmc_test.run_lnoafqmc(options, script=script)

options = {
           'eql_time': 10,
           'n_blocks': 200,
           'n_walkers': 50,
           'mix_precision': True,
           'frozen_vir_rate': 2,
           'n_corr_blocks': 50,
           'seed': 343,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

script = 'run_lno_afqmc_pt2ccsd_frozen_vir.py'
lno_afqmc_test.run_lnoafqmc(options, script=script)
