import numpy as np
from pyscf import gto, scf, lo

lno_num = 2
lno_list = [3e-4,1e-4,3e-5,1e-5]
lno_thresh = lno_list[lno_num-1]

####  test H2 monomers ####
a = 2 # bond length in a cluster
d = 4 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 5 # set as integer multiple of monomers
spin = 0 # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
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
            spin=spin*nc,
            max_memory=40000,
            )

mf = scf.UHF(mol).density_fit()
mf.kernel()

# from pyscf.lno.tools import autofrag_iao
# from pyscf import lo
# import numpy as np
# from pyscf.data import elements

# frozen = elements.chemcore(mol)

# # IAO localization
# orbocc_a = mf.mo_coeff[0][:,frozen:np.count_nonzero(mf.mo_occ[0])]
# orbocc_b = mf.mo_coeff[1][:,frozen:np.count_nonzero(mf.mo_occ[1])]
# lo_coeff_a = lo.iao.iao(mol, orbocc_a)
# lo_coeff_a = lo.orth.vec_lowdin(lo_coeff_a, mf.get_ovlp())
# lo_coeff_b = lo.iao.iao(mol, orbocc_b)
# lo_coeff_b = lo.orth.vec_lowdin(lo_coeff_b, mf.get_ovlp())
# lo_coeff = [lo_coeff_a, lo_coeff_b]
# moliao = lo.iao.reference_mol(mol)
# frag_lolist = autofrag_iao(moliao)
# frag_lolist = [[i,i] for i in frag_lolist]

from afqmc.lno_afqmc import tools, lno_afqmc

lo_coeff, frag_lolist, atm_center = tools.iao_localization(mf)

options = {
           'eql_time': 10,
           'n_blocks': 50,
           'n_walkers': 300,
           'mix_precision': True,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

lno_afqmc.run_afqmc(
              mf,
              lo_coeff = lo_coeff,
              frag_lolist = frag_lolist,
              nfrozen = frozen,
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 5e-4,
              run_frag_list = None,
              atom_group = atm_center,
              plot_las = True,
              )
