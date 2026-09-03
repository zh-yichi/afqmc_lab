"""Sweep the tail sample count M for the semistochastic-Cholesky upt2CCSD trial.

Starts from an existing FCIDUMP_chol / amplitudes.npz in the working directory
(i.e. post-CCSD), and runs the same integrals with:

    upt2ccsd_bar                        exact Cholesky sum          (reference)
    upt2ccsd_sto_chol n_chol_head=full  no sampling, same class     (control)
    upt2ccsd_sto_chol M in M_LIST       head = ratio*nchol + sampled tail

Two references, doing different jobs:

  * 'exact' validates the whole chain.  It uses sampler_pt2, so it walks a
    different trajectory -- comparison is an ordinary z-test.
  * 'head=full' is the *matched* control: it goes through sampler_pt2_sto_chol,
    which splits a key every block regardless of the head setting, so it shares
    an RNG stream with every sampled run.  Its error is therefore the pure
    trajectory error with no Cholesky sampling in it.

That second point is what makes the sweep informative.  The sampling noise is
independent of the trajectory noise, so

    err(M)^2  ~  err_full^2  +  err_sampling(M)^2

and the table reports err_sampling(M) = sqrt(err(M)^2 - err_full^2), which should
fall as 1/sqrt(M).  Pick the smallest M whose err_sampling is comfortably below
your target error -- beyond that you are paying for samples that no longer move
the total error bar.
"""

import os, re
import numpy as np
from afqmc import launch_afqmc

# ---------------------------------------------------------------- settings
M_LIST = (4, 8, 16, 32, 64, 128)   # tail draws per walker per block to compare
HEAD_RATIO = 0.125                 # head = round(HEAD_RATIO * nchol), exact
RUN_EXACT = True                   # include the upt2ccsd_bar reference run
RUN_FULL = True                    # include the head='full' matched control
SKIP_EXISTING = False              # True: reuse .out files already present
                                   # (handy to resume an interrupted sweep --
                                   #  but stale files are reused silently, so
                                   #  clear them if you change QMC settings)

QMC = {
    'n_blocks': 300,
    'n_walkers': 300,
    'max_memory': 8000,
    'seed': 17,
    'walker_type': 'uhf',
    'mix_precision': False,   # exact arithmetic, so differences are the sampling
}

# ---------------------------------------------------------------- run list
runs = []
if RUN_EXACT:
    runs.append(("exact (upt2ccsd_bar)",
                 {**QMC, 'trial': 'upt2ccsd_bar'},
                 "o2_exact.out"))
if RUN_FULL:
    runs.append(("sto_chol head=full",
                 {**QMC, 'trial': 'upt2ccsd_sto_chol', 'n_chol_head': 'full'},
                 "o2_sto_full.out"))
for m in M_LIST:
    runs.append((f"sto_chol M={m}",
                 {**QMC, 'trial': 'upt2ccsd_sto_chol',
                  'n_chol_samples': m, 'head_chol_ratio': HEAD_RATIO},
                 f"o2_sto_M{m}.out"))

print(f"{len(runs)} runs: {', '.join(label for label, _, _ in runs)}")

for label, options, outfile in runs:
    if SKIP_EXISTING and os.path.exists(outfile):
        print(f"\n[skip] {label}: reusing existing {outfile}")
        continue
    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    launch_afqmc.ph_afqmc(options)
    os.system(f"mv afqmc.out {outfile}")


# ---------------------------------------------------------------- parsing
def parse(outfile):
    energy = error = walltime = None
    if not os.path.exists(outfile):
        return None, None, None
    with open(outfile) as handle:
        for line in handle:
            hit = re.search(r"\[Trial\] Energy \(Ha\)\s+(\S+)\s+\+/-\s+(\S+)", line)
            if hit:
                energy, error = float(hit.group(1)), float(hit.group(2))
            hit = re.search(r"Run Time\s+(\S.*\S)\s*$", line)
            if hit:
                walltime = hit.group(1)
    return energy, error, walltime


def seconds(text):
    """'1h 02m 03.00s' / '4m 19.04s' / '9.43s' -> float seconds."""
    if not text:
        return None
    total = 0.0
    hit = re.search(r"(\d+)h", text)
    if hit:
        total += 3600 * int(hit.group(1))
    hit = re.search(r"(\d+)m", text)
    if hit:
        total += 60 * int(hit.group(1))
    hit = re.search(r"([\d.]+)s", text)
    if hit:
        total += float(hit.group(1))
    return total


results = {label: parse(outfile) for label, _, outfile in runs}

e_exact = err_exact = None
if RUN_EXACT:
    e_exact, err_exact, _ = results["exact (upt2ccsd_bar)"]
err_full = results["sto_chol head=full"][1] if RUN_FULL else None
t_exact = seconds(results["exact (upt2ccsd_bar)"][2]) if RUN_EXACT else None

# ---------------------------------------------------------------- table
print("\n" + "=" * 92)
print("  exact vs semistochastic Cholesky: tail sample count sweep")
print("=" * 92)
print(f"  {'run':<24}{'E [Ha]':>14}{'error':>11}{'z vs exact':>11}"
      f"{'samp err':>11}{'time':>14}{'speedup':>9}")
print("-" * 92)

failures = []
sampling_err = {}
for label, _, outfile in runs:
    energy, error, walltime = results[label]
    if energy is None:
        print(f"  {label:<24}{'FAILED -- no energy in output':>50}")
        failures.append(f"{label}: no energy parsed")
        continue

    if e_exact is None or label == "exact (upt2ccsd_bar)":
        z_text = "--"
    else:
        combined = np.sqrt(err_exact ** 2 + error ** 2)
        z = abs(energy - e_exact) / combined if combined > 0 else float("inf")
        z_text = f"{z:.2f}"
        if z > 3.0:
            failures.append(f"{label}: {z:.1f} sigma from exact")

    # sampling-only error, using the matched head='full' control
    if err_full is not None and label.startswith("sto_chol M="):
        excess = error ** 2 - err_full ** 2
        if excess > 0:
            s = np.sqrt(excess)
            sampling_err[int(label.split("=")[1])] = s
            s_text = f"{s:.6f}"
        else:
            s_text = "<noise"     # below the resolution of the error estimate
    else:
        s_text = "--"

    t = seconds(walltime)
    sp_text = f"{t_exact / t:.2f}x" if (t_exact and t) else "--"
    print(f"  {label:<24}{energy:>14.6f}{error:>11.6f}{z_text:>11}"
          f"{s_text:>11}{walltime or '?':>14}{sp_text:>9}")

print("-" * 92)

# ---------------------------------------------------------------- scaling
if len(sampling_err) >= 2:
    print("\n  sampling error should fall as 1/sqrt(M):")
    print(f"    {'M':>6}{'samp err':>12}{'x sqrt(M)':>12}   (this column should be flat)")
    for m in sorted(sampling_err):
        print(f"    {m:>6}{sampling_err[m]:>12.6f}{sampling_err[m] * np.sqrt(m):>12.6f}")
    ms = sorted(sampling_err)
    ratios = [sampling_err[a] / sampling_err[b] for a, b in zip(ms, ms[1:])]
    expected = [np.sqrt(b / a) for a, b in zip(ms, ms[1:])]
    print("\n    consecutive ratios: "
          + ", ".join(f"{r:.2f} (want {e:.2f})" for r, e in zip(ratios, expected)))
elif err_full is not None:
    print("\n  Not enough M values above the error-estimate noise floor to check"
          "\n  1/sqrt(M) scaling -- the sampling error is already well below the"
          "\n  trajectory error, which means M is comfortably large.")

print("=" * 92)
if failures:
    print("\nFAIL")
    for item in failures:
        print(f"  - {item}")
else:
    print("\nPASS  all sampled runs agree with the exact Cholesky sum within error bars.")
