import numpy as np
from pyscf import gto, scf, lo, lib

# ####  test O2 monomers ####
# a = 1.20577 # bond length in a cluster
# d = 100 # distance between each cluster
# unit = 'A' # unit of length
# na = 2 # size of a cluster (monomer)
# nc = 5 # set as integer multiple of monomers
# spin = 2 # spin per monomer
# elmt = 'O'
# basis = 'sto6g'
# atoms = ""
# for n in range(nc*na):
#     shift = ((n - n % na) // na) * (d-a)
#     atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"
# ###########################

atoms = '''
Fe -0.64147387529051 0.51990405379180 0.11450483185168
O 0.33915564253394 2.18819453520299 -0.84159903476570
H 0.65823736209818 2.99799985286513 -0.40575899522591
O -1.29914529040912 -0.20924466129350 -1.80322550106302
H -2.11419692217693 0.03267722517314 -2.27701147100162
O 0.01237770373627 1.24396717111997 2.03533113813669
H 0.83097217272839 1.00756306178418 2.50577204385612
O -1.62284672685044 -1.15048542622041 1.07007991103536
H -1.94592790838151 -1.95766677806096 0.63237073377438
O -2.41866907277644 1.66373032346654 0.25658821238371
H -2.55381601393967 2.56024091790599 -0.09930153080408
O 1.13404022498401 -0.62623224778001 -0.02436092147139
H 1.26370551457750 -1.52400168391753 0.33034524187694
H -3.28110369794222 1.35873477131317 0.59094893393907
H -0.81032015579443 -0.82248643308581 -2.37966496653734
H 1.99795010138314 -0.32692539271158 -0.36001715946551
H 0.57998322602479 2.26642229799559 -1.78148703231078
H -0.47668197075682 1.85673194452426 2.61208351578631
H -1.85653031374812 -1.23356353207299 2.011352050005
'''

# mol = gto.M()
# mol.basis = {'default': 'x2c-svpall', 'Fe': 'x2c-tzvpall'}
# mol.nucmod = 'G'
# mol.unit = 'A'
# mol.spin = 4
# mol.charge = 2
# mol.symmetry = 0
# mol.verbose = 4
# mol.build()

# mf = scf.UHF(mol).density_fit()
# mf.kernel()

# stable = False
# while not stable:
#     print(f'mean-field stability test')
#     if not stable:
#         mo_i, _, stable,_ = mf.stability(return_status=True)
#         dm = mf.make_rdm1(mo_i,mf.mo_occ)
#         mf.kernel(dm0=dm)
#     elif stable:
#         print(f'UHF Energy: {mf.e_tot}, stability {stable}')
#         break

# from afqmc.lno_afqmc import lno_afqmc, tools
# lo_coeff, frag_list, frag_name = tools.iao_fragment(mf, frag_type='h2heavy', more_loc='pm')

path2chk = "../test_iao/hs_tzvp_mf.chk"
mol = lib.chkfile.load_mol(path2chk)
mol.basis = {'default': 'x2c-tzvpall', 'Fe': 'x2c-tzvpall'}
mol.nucmod = 'G'
mol.build()

print(mol.nao)
print(mol.basis)

mf = scf.UHF(mol).density_fit()
mf = mf.x2c()
dm0 = mf.from_chk(path2chk)
mf.chkfile = path2chk
mf.max_cycle = 100
mf.kernel(dm0=dm0)

from afqmc.lno_afqmc import lno_afqmc, tools
lo_coeff, frag_list, frag_name = tools.iao_fragment(
    mf, 
    frag_type='h2heavy', 
    more_loc='pm',
    x2c = True, 
    save2=None, 
    read_from=None
    )

from afqmc.lno_afqmc import lno_afqmc
options = {
           'n_prop_steps': 50,
           'n_blocks': 300,
           'n_walkers': 100,
           'frozen_vir_rate': 4,
           'max_memory': 8000,
           'mix_precision': True,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_sto_chol',
           'chol_cost_ratio': 0.2,
           # semistochastic Cholesky sum in the fragment T2*h2 term.
           # e2_0 stays exact; only e2_2_2_1, e2_2_2_2 and e2_2_3 are sampled.
        #    'head_chol_ratio': 0.125, # head = round(0.125 * nchol), summed exactly
        #    'n_chol_samples': 8,      # tail draws per walker per block
           # 'n_chol_head': 'full',  # uncomment to disable sampling entirely and
                                     # reproduce the exact 'upt2ccsd' result
           }

lno_afqmc.run_afqmc(
    mf,
    lo_coeff, 
    frag_list,
    frag_name,
    lno_thresh = 1e-5,
    qmc_options = options, 
    chol_cut = 1e-5, 
    target_qmc_err = 1e-3, 
    run_frag = [6], 
    nfrozen = None,
    run_mp = True,
    run_cc = True,
    run_qmc = True,
    plot_las = False,
    )
