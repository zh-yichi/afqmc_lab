"""N2: exact vs semistochastic-Cholesky pt2CCSD.

Same molecule as test_n2.py (N2 at 3.0 bohr), but with the *restricted* trials,
since pt2ccsd_sto_chol lives in wavefunctions_restricted.py.  So this uses RHF +
restricted walkers where test_n2.py uses UHF + 'uhf' walkers.

Runs the same system three ways and compares:

    rpt2ccsd_bar        exact Cholesky sum in the energy      (reference)
    rpt2ccsd_sto_chol   head exact + sampled tail, M = 32
    rpt2ccsd_sto_chol   head exact + sampled tail, M = 128

Only the T2-contracted part of the energy is sampled: e2_0 -- and hence
e2_2_1 = e2_0 * gt2g -- is always summed exactly, because it is needed to build
the sampling proposal anyway.  The estimator is unbiased, so the three runs must
agree within their error bars; M controls the variance, not the answer.

Knobs for the new trial (both optional):
    n_chol_samples : tail draws per walker per block (default 128)
    n_chol_head    : exactly summed head; 0 = auto = round(nchol/8)

NOTE: this writes FCIDUMP_chol and amplitudes.npz into the working directory,
overwriting whatever is already there -- run it in a scratch directory if you
want to keep the existing ones.
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
jax.config.update("jax_enable_x64", True)

import re
import numpy as np
from pyscf import gto, scf, cc

# ---------------------------------------------------------------- settings
BASIS = "ccpvdz"          # ccpvqz, as in test_n2.py, is where the sampling
                          # actually pays: nchol grows and the head stays at
                          # nchol/8 while the tail budget is fixed.
N_CHOL_SAMPLES = [0,8,32,64]
QMC = {
    "eql_time": 30,
    "n_blocks": 300,
    "n_walkers": 300,
    "max_error": 0.0,
    "max_memory": 30000,
    "seed": 17,
    "walker_type": "rhf",
    "mix_precision": False,   # exact arithmetic, so any difference is the sampling
}

# ---------------------------------------------------------------- molecule
atoms = "N 0.0 0.0 0.0 \n" "N 0.0 0.0 2.0 \n"
mol = gto.M(atom=atoms, unit="b", basis=BASIS, spin=0, verbose=4)
mol.build()

mf = scf.RHF(mol)
mf.kernel()

for i in range(10):
    print(f"mf stability test {i + 1}")
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        print(f"mf energy: {mf.e_tot}, stability {stable}")
        break
    dm = mf.make_rdm1(mo_i, mf.mo_occ)
    mf.kernel(dm0=dm)

mycc = cc.CCSD(mf)
mycc.set_frozen()
mycc.conv_tol = 1e-6
mycc.conv_tol_normt = 3e-5
mycc.kernel()
print(f"\nRHF   = {mf.e_tot:.8f}")
print(f"CCSD  = {mycc.e_tot:.8f}")

# ---------------------------------------------------------------- afqmc runs
from afqmc import integral, launch_afqmc

integral.prep_integral(mycc)

runs = [("exact", {**QMC, "trial": "rpt2ccsd_bar"}, "n2_exact.out")]
for m in N_CHOL_SAMPLES:
    runs.append(
        (f"sto_chol M={m}",
         {**QMC, "trial": "rpt2ccsd_sto_chol", "n_chol_samples": m},
         f"n2_sto_chol_M{m}.out")
    )

for label, options, outfile in runs:
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    launch_afqmc.ph_afqmc(options)
    os.system(f"mv afqmc.out {outfile}")

# ---------------------------------------------------------------- comparison
def parse(outfile):
    """Pull '[Trial] Energy (Ha)  <E> +/- <err>' and the run time out of afqmc.out."""
    energy = error = walltime = None
    with open(outfile) as handle:
        for line in handle:
            hit = re.search(r"\[Trial\] Energy \(Ha\)\s+(\S+)\s+\+/-\s+(\S+)", line)
            if hit:
                energy, error = float(hit.group(1)), float(hit.group(2))
            hit = re.search(r"Run Time\s+(\S.*\S)\s*$", line)
            if hit:
                walltime = hit.group(1)
    return energy, error, walltime


print("\n" + "=" * 74)
print("  N2 pt2CCSD: exact Cholesky sum vs semistochastic sampling")
print("=" * 74)
print(f"  {'run':<18}{'E [Ha]':>14}{'error':>11}{'z vs exact':>12}{'time':>16}")
print("-" * 74)

results = [(label, *parse(outfile)) for label, _, outfile in runs]
e_ref, err_ref, _ = results[0][1], results[0][2], results[0][3]
failures = []
for label, energy, error, walltime in results:
    if energy is None:
        print(f"  {label:<18}{'FAILED -- no energy in output':>50}")
        failures.append(f"{label}: no energy parsed")
        continue
    if label == "exact":
        z_text = "--"
    else:
        combined = np.sqrt(err_ref ** 2 + error ** 2)
        z = abs(energy - e_ref) / combined if combined > 0 else float("inf")
        z_text = f"{z:.2f}"
        if z > 3.0:
            failures.append(f"{label}: {z:.1f} sigma from exact")
    print(f"  {label:<18}{energy:>14.6f}{error:>11.6f}{z_text:>12}{walltime or '?':>16}")

print("-" * 74)
print(f"  reference CCSD    {mycc.e_tot:>14.6f}")
print("=" * 74)
if failures:
    print("\nFAIL")
    for item in failures:
        print(f"  - {item}")
else:
    print("\nPASS  sampled runs agree with the exact Cholesky sum within error bars.")
