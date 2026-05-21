from afqmc import launch_afqmc
#    prep.prep_afqmc(mycc, chol_cut=1e-5)

    # RHF Trial
options = {'n_eql': 10, # 1 eql step = dt*n_prop_steps (here 160 = 40 au)
           'n_blocks': 50, # tune this for how many samples you want
           'n_walkers': 300,
           'dt':0.005, # time every trot step
           'max_error': 0.0, # set to 0 to run the calculation till n_blocks
               'seed': 17,
               'walker_type': 'rhf',
               'trial': 'ccsd_pt2',
               'free_projection': False,
               'use_gpu': False
               }

launch_afqmc.run_afqmc(options)
