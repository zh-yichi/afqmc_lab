import numpy as np
from pyscf import gto, scf, lo

lno_num = 2
lno_list = [3e-4,1e-4,3e-5,1e-5]
lno_thresh = lno_list[lno_num-1]

####  test H2 monomers ####
a = 2.2 # bond length in a cluster
d = 100 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 2 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'N'
unit = 'B'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
###########################

mol = gto.M(atom=atoms,
            basis="sto6g",
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=0,
            max_memory=40000,
            )

mf = scf.RHF(mol).density_fit()
mf.kernel()

from afqmc.lno_afqmc import lno_afqmc, tools
lo_coeff, frag_list, frag_name = tools.iao_fragment(mf, frag_type='h2heavy', more_loc='pm')

options = {
           'eql_time': 10,
           'n_prop_steps': 50,
           'n_blocks': 100,
           'n_walkers': 100,
           'mix_precision': False,
           'seed': 17,
           'walker_type': 'rhf',
           'trial': 'pt2ccsd',
           }

lno_afqmc.run_afqmc(
    mf,
    lo_coeff, 
    frag_list,
    frag_name,
    lno_thresh = 1e-6,
    qmc_options = options, 
    chol_cut = 1e-5, 
    target_qmc_err = 1e-3, 
    run_frag = [0,1], 
    nfrozen = None,
    run_mp = True,
    run_cc = True,
    run_qmc = True,
    plot_las = False,
    )
