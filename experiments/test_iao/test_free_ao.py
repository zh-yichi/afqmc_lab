import numpy as np
from pyscf import gto, lib
from pyscf.scf import atom_hf

path2chk = "./hs_tdz_mf.chk"
mol = lib.chkfile.load_mol(path2chk)

# gto.mole.loads() flattens an empty _ecpbas to shape (0,) instead of (0, 8),
# and build() only reassigns it when an ECP is present, so atom_hf's
# _ecpbas[:,0] indexing raises IndexError. Restore the 2D shape.
# if mol._ecpbas.ndim != 2:
#     mol._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)

print(mol.basis)
print(mol._ecpbas)

# from afqmc.lno_afqmc import tools

# from afqmc.lno_afqmc import lno_afqmc, tools
# lo_coeff, frag_list, frag_name = tools.iao_fragment(mf, frag_type='h2heavy', more_loc='pm') # <-- this has an error

# ref_basis = tools.free_atom_minao(mol) # <-- this has an error
# print(ref_basis)

# atm_scf_result = atom_hf.get_atm_nrhf(mol) # <-- this has an error
# print(atm_scf_result)
