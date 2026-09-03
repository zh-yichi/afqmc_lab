from afqmc.lno_afqmc import lno_afqmc
# options = {
#            'n_prop_steps': 50,
#            'n_blocks': 600,
#            'n_walkers': 300,
#            'max_memory': 2000,
#            'mix_precision': False,
#            'seed': 17,
#            'walker_type': 'uhf',
#            'trial': 'upt2ccsd',
#            }
# lno_afqmc.run_lnoafqmc(options)


options = {
           'n_prop_steps': 50,
           'n_walkers': 100,
           'frozen_vir_rate': 4,
           'n_blocks': 300,
           'n_corr_blocks' : 60,
           'max_memory': 8000,
           'mix_precision': True,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_sto_chol',
           # semistochastic Cholesky sum in the fragment T2*h2 term.
           'chol_cost_ratio': 0.2,
           # e2_0 stays exact; only e2_2_2_1, e2_2_2_2 and e2_2_3 are sampled.
        #    'head_chol_ratio': 0.125, # head = round(0.125 * nchol), summed exactly
        #    'n_chol_samples': 32,      # tail draws per walker per block
           # 'n_chol_head': 'full',  # uncomment to disable sampling entirely and
                                     # reproduce the exact 'upt2ccsd' result
           }

lno_afqmc.run_lnoafqmc(options)
