from pyscf import gto, lib, scf
import basis_set_exchange as bse

def bse_basis(name, elements):
    """Return a PySCF basis dict for the given elements from BSE."""
    out = {}
    for el in set(elements):
        raw = bse.get_basis(name, elements=[el], fmt='nwchem', header=False)
        out[el] = gto.basis.parse(raw)
    return out

path2chk = "./hs_tzvp_mf.chk"
mol = lib.chkfile.load_mol(path2chk)
mol.basis = {'default': 'x2c-tzvpall', 'Fe': 'x2c-tzvpall'}
mol.nucmod = 'G'
mol.build()

print(mol.nao)
print(mol.basis)

mf = scf.UHF(mol).density_fit()
mf = mf.x2c()
dm0 = mf.from_chk(path2chk)
mf.max_cycle = 100
mf.kernel(dm0=dm0)

from afqmc.lno_afqmc import lno_afqmc_test, tools
lo_coeff, frag_list, frag_name = tools.iao_fragment(
    mf, 
    frag_type='h2heavy', 
    more_loc='pm',
    x2c = True, 
    save2=None, 
    read_from=None
    )

options = {'eql_time': 10,
           'n_prop_steps': 50,
           'n_blocks': 10,
        #    'frozen_vir': 10,
        #    'n_corr_blocks': 50,
           'n_walkers': 1,
           'max_memory': 10000,
           'mix_precision': True,
           'n_batch': 1,
           'seed': 18,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd',
           }

# script = 'run_lno_afqmc_pt2ccsd_frozen_vir.py'
script = None

lno_afqmc_test.run_afqmc(
    mf,
    lo_coeff, 
    frag_list,
    frag_name,
    lno_thresh = 1e-5,
    qmc_options = options, 
    chol_cut = 1e-5, 
    target_qmc_err = 1e-3, 
    run_frag = None, 
    nfrozen = None,
    run_mp = True,
    run_cc = True,
    run_qmc = True,
    qmc_script=script,
    plot_las = False,
    )
