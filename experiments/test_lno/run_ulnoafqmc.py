import numpy as np
from pyscf import gto, scf, lo

lno_num = 2
lno_list = [3e-4,1e-4,3e-5,1e-5]
lno_thresh = lno_list[lno_num-1]

####  test H2 monomers ####
a = 2 # bond length in a cluster
d = 3 # distance between each cluster
unit = 'b' # unit of length
na = 2 # size of a cluster (monomer)
nc = 5 # set as integer multiple of monomers
spin = 0*nc # spin per monomer
frozen = 0 # frozen orbital per monomer
elmt = 'H'
unit = 'B'
basis = 'sto6g'
atoms = ""
for n in range(nc*na):
    shift = ((n - n % na) // na) * (d-a)
    atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

###########################
# atoms = '''
# O   -1.485163346097   -0.114724564047    0.000000000000
# H   -1.868415346097    0.762298435953    0.000000000000
# H   -0.533833346097    0.040507435953    0.000000000000
# O    1.416468653903    0.111264435953    0.000000000000
# H    1.746241653903   -0.373945564047   -0.758561000000
# H    1.746241653903   -0.373945564047    0.758561000000
# '''
# unit = "A"
# spin = 0

mol = gto.M(atom=atoms,
            basis="ccpvdz",
            verbose=4,
            unit=unit,
            symmetry=0,
            charge=0,
            spin=spin,
            max_memory=40000,
            )

mf = scf.UHF(mol).density_fit()
mf.kernel()

stable = False
while not stable:
    print(f'mf stability test')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'mf energy: {mf.e_tot}, stability {stable}')
        break

from afqmc.lno_afqmc import tools, lno_afqmc
lo_coeff, frag_lolist, atm_center = tools.iao_localization(mf)
from pyscf.data import elements

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
              nfrozen = elements.chemcore(mol),
              thresh = lno_thresh,
              qmc_options = options,
              chol_cut = 1e-5,
              target_sto_error = 5e-4,
              run_frag_list = None,
              atom_group = atm_center,
              plot_las = False,
              )
