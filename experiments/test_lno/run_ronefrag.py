from afqmc.lno_afqmc import lno_afqmc
options = {
           'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

lno_afqmc.run_lnoafqmc(options)

options = {
           'eql_time': 30,
           'n_blocks': 300,
           'n_walkers': 300,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd_sto_chol',
           # semistochastic Cholesky sum in the fragment T2*h2 term.
           'chol_cost_ratio': 0.3,
           # e2_0 stays exact; only e2_2_2_1, e2_2_2_2 and e2_2_3 are sampled.
           # 'head_chol_ratio': 0.125, # head = round(0.125 * nchol), summed exactly
           # 'n_chol_samples': ,      # tail draws per walker per block
           # 'n_chol_head': 'full',  # uncomment to disable sampling entirely and
                                     # reproduce the exact 'upt2ccsd' result
           'frozen_vir': 1,
           }


lno_afqmc.run_lnoafqmc(options)
