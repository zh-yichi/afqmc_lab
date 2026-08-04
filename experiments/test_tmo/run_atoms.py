import json
import os
from pyscf import gto, scf, cc
import numpy as np

df=json.load(open("./trail.json"))

spins={'ScO':1,'TiO':2,'VO':3,'CrO':4,'MnO':5,'FeO':4,'CuO':1}
atom_spins={'O':2,'Sc':1,'Ti':2,'V':3,'Cr':6,'Mn':5,'Fe':4,'Cu':1}

re={'ScO':1.668,
    'TiO':1.623,
    'VO':1.591,
    'CrO':1.621,
    'MnO':1.648,
    'FeO':1.616,
    'CuO':1.725,
    }

nd={'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(5,0),'Mn':(5,0),'Fe':(5,1),
     'Cu':(5,4) }

atom_list = ["O", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Cu"]

for basis in ['vdz']:
  for el in ["Cr"]:
    for charge in [0]:
      mol=gto.Mole()
      mol.ecp={}
      mol.basis={}
      for e in [el]:
        mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
        mol.basis[e]=gto.basis.parse(df[e][basis])
      mol.charge = 0
      mol.spin = atom_spins[el]
      mol.max_memory = 40000
      atoms = f"{el}  0.0 0.0 0.0"
      mol.build(atom=atoms,verbose=4)

      # These are the orbitals for which we want to read-in an initial DM guess
      TM_3s_orbitals = []
      TM_4s_orbitals = []
      TM_3p_orbitals = []
      TM_3d_orbitals = []
      O_2s_orbitals  = []
      O_2p_orbitals  = []

      aos=mol.ao_labels()
      print('')
      print('AO labels')
      print(aos)
      print('')
      for i,x in enumerate(aos):

        # Find the TM 3s labels
        if (('3s' in x) and (el in x)):
          TM_3s_orbitals.append(i)

        # Find the TM 4s labels
        if (('4s' in x) and (el in x)):
          TM_4s_orbitals.append(i)

        # Find the TM 3p labels
        if (('3p' in x) and (el in x)):
          TM_3p_orbitals.append(i)

        # Find the TM 3d labels
        if (('3d' in x) and (el in x)):
          TM_3d_orbitals.append(i)

        # Find the O 2s labels
        if (('2s' in x) and ('O' in x)):
          O_2s_orbitals.append(i)

        # Find the O 2p labels
        if (('2p' in x) and ('O' in x)):
          O_2p_orbitals.append(i)

      # There should be 5 3d TM orbitals. Let's check this!
      assert len(TM_3d_orbitals)==5

      mf = scf.RHF(mol)
      dm = np.zeros(mf.init_guess_by_minao().shape)

      # The 3s is always doubly-occupied for the TM atom
      for s in TM_3s_orbitals:
        for spin in [0,1]:
          dm[spin,s,s]=1

      # The 4s is always at least singly-occupied for the TM atom
      for s in TM_4s_orbitals:
        dm[0,s,s]=1

      # Control the 4s double-occupancy
      if (el=='Cr'):
        for s in TM_4s_orbitals:
          print('We are singly filling this 4s-orbital: '+str(aos[s]) )
          dm[1,s,s]=0

      # Always doubly-occupy the 3p orbitals for the TM atom
      for p in TM_3p_orbitals:
        for s in [0,1]:
          dm[s,p,p]=1

      # Control the 3d occupancy for CrO...
      if (el=='Cr'):
        for i,d in enumerate(TM_3d_orbitals):

          # These are the 3d orbitals we want to fill to get the correct symmetry
          if ( ('xy' in aos[d]) or ('yz' in aos[d]) or ('z^2' in aos[d]) or ('x2-y2' in aos[d]) ):
            print('We are singly filling this d-orbital: '+str(aos[d]) )
            dm[0,d,d]=1

      mo1 = None
      mf.level_shift = 0.5
      rohf_energy = mf.kernel()
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mo1 = mf.stability()[0]
      mf = mf.newton().run(mo1, mf.mo_occ)
      mf.stability()

      umf = scf.UHF(mol)
      mo_occ = [
              np.array([1 for i in range(mol.nelec[0])] + [0 for i in range(mol.nao-mol.nelec[0])]),
              np.array([1 for i in range(mol.nelec[1])] + [0 for i in range(mol.nao-mol.nelec[1])]),
              ]
      dm0 = umf.make_rdm1([mo1, mo1], mo_occ)
      umf.level_shift = 0.5
      umf.kernel(dm0)
      mo1 = umf.stability()[0]
      umf = umf.newton().run(mo1, umf.mo_occ)
      mo1 = umf.stability()[0]
      umf = umf.newton().run(mo1, umf.mo_occ)

      norb_frozen = 0
      mycc = cc.CCSD(umf)
      mycc.frozen = norb_frozen
      mycc.kernel()

      options = {
           'eql_time': 10,
           'n_blocks': 100,
           'n_walkers': 300,
           'mix_precision': True,
           'max_memory': 3000,
           'seed': 17,
           'walker_type': 'uhf',
           'trial': 'upt2ccsd_cisd',
           }

      from afqmc import integral, launch_afqmc
      integral.prep_integral(mycc, chol_cut=1e-5)
      launch_afqmc.ph_afqmc(options)
      os.system(f"mv afqmc.out afqmc_{el}.out")
