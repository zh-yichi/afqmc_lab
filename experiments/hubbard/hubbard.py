"""
Hubbard-model AFQMC helpers.

Everything the Hubbard notebooks in this directory need, in one place: building the
model as a PySCF mean-field object, factorising the on-site interaction for AFQMC,
the propagator/sampler subclasses that `afqmc` is missing, two drivers (free
projection and constrained path / phaseless), and the walker-analysis tools for the
one-electron / two-site toy problem.

Typical use
-----------
    import hubbard as hub
    hub.setup_jax()

    h1, eri, mol, mf = hub.hubbard_mf(2, t=1.0, u=4.0, nelec=(1, 0),
                                      dm0=hub.polarised_dm0(2))
    sysm = hub.afqmc_setup(h1, 4.0, (1, 0), mf.mo_coeff[0][:, :1],
                           mf.mo_coeff[1][:, :0], 2)

    fp = hub.run_free_projection(sysm, n_blocks=60, n_traj=24,
                                 snap_every=10, verbose=2)
    cp = hub.run_constrained(sysm, n_blocks=120, eql_time=5.0,
                             snap_every=5, verbose=1)
    print(cp.plateau())

Free projection needs independent trajectories -- each one is an unconstrained walk whose
weight decays, so they are averaged by amplitude.  Constrained path does not: it is one
open-ended random walk, equilibrated once and then sampled block after block, following
`afqmc/scripts/run_afqmc.py`.  Its error bar comes from a blocking analysis of that single
series (`sampling.blocking`), not from restarting.

Both drivers return a `QMCResult`, which carries the block energies, the walker
snapshots and the timing.  `snap_every` is a single cadence, in blocks: it says
when to store a walker snapshot *and* when to report progress.  `verbose` says how
much to print (0 silent, 1 running energies, 2 also the reconstructed wavefunction
at every snapshot).
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------------
# jax configuration.  If the caller has already imported jax (the usual case, a
# notebook that ran afqmc.config.setup_jax() first) we only make sure float64 is on
# and print nothing; otherwise we run afqmc's full setup, which also picks the
# backend and must happen before jax is imported.
# --------------------------------------------------------------------------------
_JAX_WAS_IMPORTED = "jax" in sys.modules

from afqmc import config as _afqmc_config  # noqa: E402


def setup_jax() -> None:
    """Configure jax for AFQMC (float64, CPU/GPU backend).  Safe to call twice."""
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


if not _JAX_WAS_IMPORTED:
    _afqmc_config.setup_jax()
else:
    setup_jax()

import jax.numpy as jnp                      # noqa: E402
from jax import jit, random                  # noqa: E402

from afqmc import fp_sampling, linalg_utils, prep, propagation, sampling  # noqa: E402
from afqmc.wavefunctions import wavefunctions_unrestricted as uwf         # noqa: E402

__all__ = [
    "setup_jax",
    "hubbard_h1", "hubbard_eri", "hubbard_mf", "polarised_dm0", "neel_dm0",
    "spin_chol", "charge_chol", "afqmc_setup", "HubbardSystem",
    "propagator_uhf_gen", "propagator_cp", "fp_sampler_gen",
    "Snapshot", "QMCResult", "run_free_projection", "run_constrained",
    "plateau_estimate", "blocking_estimate",
    "theta_of", "walker_angles", "angles_of", "walker_ci_angles",
    "gauge_fix", "ci_vectors", "reconstruct",
    "summarize", "wavefunction",
    "dimer_ci", "exact_mixed_estimator",
]


# ================================================================================
# 1.  The model, and its PySCF mean field
# ================================================================================

def hubbard_h1(n_sites: int, t: float, pbc: bool = False) -> np.ndarray:
    """Nearest-neighbour hopping matrix of an open (or periodic) chain."""
    h1 = np.zeros((n_sites, n_sites))
    for i in range(n_sites - 1):
        h1[i, i + 1] = h1[i + 1, i] = -t
    if pbc and n_sites > 2:
        h1[0, -1] = h1[-1, 0] = -t
    return h1


def hubbard_eri(n_sites: int, u: float) -> np.ndarray:
    """On-site two-electron integrals (ii|ii) = U in chemist notation."""
    eri = np.zeros((n_sites,) * 4)
    for i in range(n_sites):
        eri[i, i, i, i] = u
    return eri


def polarised_dm0(n_sites: int, site: int = 0) -> np.ndarray:
    """SCF guess with a single alpha electron pinned to one site."""
    dm0 = np.zeros((2, n_sites, n_sites))
    dm0[0, site, site] = 1.0
    return dm0


def neel_dm0(n_sites: int) -> np.ndarray:
    """Antiferromagnetic SCF guess, one electron per site with alternating spin."""
    dm0 = np.zeros((2, n_sites, n_sites))
    for i in range(n_sites):
        dm0[i % 2, i, i] = 1.0
    return dm0


def hubbard_mf(n_sites, t, u, nelec, dm0=None, pbc=False, verbose=0, h1=None):
    """
    Build a PySCF UHF object for a Hubbard chain.

    PySCF has no notion of a lattice, but an SCF object only ever asks the molecule
    for `get_hcore`, `get_ovlp` and `_eri`; overriding those three runs the full
    SCF/FCI machinery on any model Hamiltonian.  The lattice sites play the role of
    an orthonormal AO basis, hence `get_ovlp = 1`.

    Returns (h1, eri, mol, mf).  `nelec` is (n_alpha, n_beta).

    `h1` overrides the nearest-neighbour chain with an arbitrary (symmetric)
    hopping matrix; `t` and `pbc` are then ignored.
    """
    from pyscf import ao2mo, gto, scf

    h1 = hubbard_h1(n_sites, t, pbc) if h1 is None else np.asarray(h1, dtype=float)
    eri = hubbard_eri(n_sites, u)

    mol = gto.M()
    mol.nelectron = sum(nelec)
    mol.nao = n_sites
    mol.spin = nelec[0] - nelec[1]
    mol.incore_anyway = True
    mol.verbose = verbose
    mol.build()

    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n_sites)
    mf._eri = ao2mo.restore(8, eri, n_sites)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.verbose = verbose
    mf.kernel(dm0=dm0)
    return h1, eri, mol, mf


# ================================================================================
# 2.  Hubbard-Stratonovich factorisation of the on-site interaction
# ================================================================================

def spin_chol(n_sites: int, u: float):
    r"""
    Spin (Hirsch) decomposition,  L^g_up = -i sqrt(U) |g><g|,  L^g_dn = +i sqrt(U) |g><g|.

    Uses  U n_up n_dn = -U/2 (n_up - n_dn)^2 + U/2 (n_up + n_dn), so the operator
    L is anti-Hermitian and the field propagator exp(i sqrt(dt) x L) is REAL: the
    Hubbard model then has a sign problem but no phase problem.  This is the
    decomposition to use.
    """
    d = np.zeros((n_sites, n_sites, n_sites), dtype=complex)
    for g in range(n_sites):
        d[g, g, g] = 1.0
    return jnp.array(-1j * np.sqrt(u) * d), jnp.array(+1j * np.sqrt(u) * d)


def charge_chol(n_sites: int, u: float):
    r"""
    Charge decomposition,  L^g_up = L^g_dn = sqrt(U) |g><g|.

    Uses  U n_up n_dn = U/2 n^2 - U/2 n.  L is Hermitian, so exp(i sqrt(dt) x L) is
    a complex phase and the walkers leave the real plane -- a phase problem.
    Provided for contrast.
    """
    d = np.zeros((n_sites, n_sites, n_sites), dtype=complex)
    for g in range(n_sites):
        d[g, g, g] = 1.0
    return jnp.array(np.sqrt(u) * d), jnp.array(np.sqrt(u) * d)


@dataclass
class HubbardSystem:
    """Everything the drivers need: Hamiltonian, trial, and their intermediates."""
    ham: Any
    ham_data: dict
    nchol: int
    trial: Any
    wave_data: dict
    h1: np.ndarray
    u: float
    nelec: tuple
    norb: int

    @property
    def trial_vector(self) -> np.ndarray:
        """Occupied alpha trial orbitals as a real numpy array (norb, nocc)."""
        return np.asarray(self.wave_data["mo_coeff"][0]).real

    @property
    def t(self) -> float:
        """Nearest-neighbour hopping, read back off h1."""
        return float(-self.h1[0, 1])

    @property
    def ci_hamiltonian(self) -> np.ndarray:
        """
        Hamiltonian in the determinant basis.

        One electron on N sites: the determinants *are* the sites, so the CI
        Hamiltonian is h1 itself -- U never enters, because U n_up n_dn needs one
        electron of each spin.  Two sites with one electron of each spin: the 4x4
        `dimer_ci` matrix.
        """
        if sum(self.nelec) == 1:
            return np.asarray(self.h1, dtype=float)
        return dimer_ci(self.t, self.u, self.nelec)[0]

    @property
    def ci_trial(self) -> np.ndarray:
        """The trial determinant written as a CI vector in the same basis."""
        if sum(self.nelec) == 1:
            occ = 0 if self.nelec[0] else 1
            return np.asarray(self.wave_data["mo_coeff"][occ]).real.ravel()
        to_ci = dimer_ci(self.t, self.u, self.nelec)[1]
        return to_ci(np.asarray(self.wave_data["mo_coeff"][0]).real,
                     np.asarray(self.wave_data["mo_coeff"][1]).real)


def afqmc_setup(h1, u, nelec, mo_a, mo_b, n_sites,
                decomposition="spin", h0=0.0) -> HubbardSystem:
    """
    Package a Hubbard model plus a UHF trial into the objects `afqmc` expects.

    `mo_a`, `mo_b` are the *occupied* trial orbitals, shape (n_sites, nelec[sigma]).
    """
    chol = spin_chol(n_sites, u) if decomposition == "spin" else charge_chol(n_sites, u)

    ham, ham_data, nchol = prep.get_hamiltonian(h0, (h1, h1), chol, n_sites)

    trial = uwf.uhf(n_sites, tuple(nelec), nchol_chunk=100, n_batch=1)
    wave_data = {"mo_coeff": [jnp.array(mo_a), jnp.array(mo_b)]}
    wave_data["rdm1"] = trial.get_rdm1(wave_data)

    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    return HubbardSystem(ham=ham, ham_data=ham_data, nchol=nchol, trial=trial,
                         wave_data=wave_data, h1=np.asarray(h1), u=float(u),
                         nelec=tuple(nelec), norb=int(n_sites))


# ================================================================================
# 3.  Propagators and samplers that `afqmc` does not provide
# ================================================================================

class propagator_uhf_gen(propagation.propagator_unrestricted):
    """
    Unrestricted phaseless propagator, plus a free-projection step that also works
    when one spin channel is empty.

    `afqmc`'s `propagate_free` divides by 2*nelec[sigma] when re-attaching the
    accumulated phase, which raises ZeroDivisionError for a fully polarised system.
    Multiplying every occupied column of both determinants by c gives a determinant
    factor c^(n_up) * c^(n_dn) = c^(N_e), so c = phase**(1/N_e) works in general.
    (It also fixes `exp(dt*h0_prop + e_estimate)` -> `exp(dt*(h0_prop + e_estimate))`;
    that factor is walker independent and cancels in the estimator, but it can
    overflow.)
    """

    @partial(jit, static_argnums=(0, 1))
    def propagate_free(self, trial, ham_data, prop_data, fields):
        n_tot = trial.nelec[0] + trial.nelec[1]

        shift_term = jnp.einsum("wg,g->w", fields, ham_data["mf_shifts"])
        constants = jnp.exp(-jnp.sqrt(self.dt) * shift_term) * jnp.exp(
            self.dt * (ham_data["h0_prop"] + prop_data["e_estimate"]))

        constants_abs = jnp.abs(constants)
        phase = constants / constants_abs

        prop_data["weights"] = prop_data["weights"] * constants_abs
        prop_data["walkers"] = self._apply_trotprop(ham_data, prop_data["walkers"], fields)

        c = phase ** (1.0 / n_tot)
        prop_data["walkers"] = self._multiply_constant(prop_data["walkers"], jnp.stack((c, c)))
        return prop_data


class propagator_cp(propagator_uhf_gen):
    """
    Constrained-path propagator.

    Identical to the phaseless `propagate` except for the weight update:
        phaseless :  W <- W * |I| * cos(theta)
        CP        :  W <- W * max(0, Re I)
    With the spin decomposition the importance function I is real, so the two are
    the same arithmetic and the two methods coincide exactly.
    """

    @partial(jit, static_argnums=(0, 1))
    def propagate(self, trial, ham_data, prop_data, fields, wave_data):
        force_bias = trial.calc_force_bias(prop_data["walkers"], ham_data, wave_data)
        field_shifts = -jnp.sqrt(self.dt) * (1.0j * force_bias - ham_data["mf_shifts"])
        shifted_fields = fields - field_shifts

        shift_term = jnp.sum(shifted_fields * ham_data["mf_shifts"], axis=1)
        fb_term = jnp.sum(fields * field_shifts - field_shifts * field_shifts / 2.0, axis=1)

        prop_data["walkers"] = self._apply_trotprop(
            ham_data, prop_data["walkers"], shifted_fields)

        overlaps_new = trial.calc_overlap(prop_data["walkers"], wave_data)
        imp_fun = (
            jnp.exp(-jnp.sqrt(self.dt) * shift_term
                    + fb_term
                    + self.dt * (prop_data["pop_control_ene_shift"] + ham_data["h0_prop"]))
            * overlaps_new / prop_data["overlaps"]
        )

        imp_fun_cp = jnp.real(imp_fun)
        imp_fun_cp = jnp.where(jnp.isnan(imp_fun_cp), 0.0, imp_fun_cp)
        imp_fun_cp = jnp.where(imp_fun_cp < 0.0, 0.0, imp_fun_cp)      # kill node crossers
        imp_fun_cp = jnp.where(imp_fun_cp > 100.0, 0.0, imp_fun_cp)

        prop_data["weights"] = imp_fun_cp * prop_data["weights"]
        prop_data["weights"] = jnp.where(prop_data["weights"] > 100, 0.0, prop_data["weights"])
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"] - 0.1 * jnp.array(
            jnp.log(jnp.sum(prop_data["weights"]) / self.n_walkers) / self.dt)
        prop_data["overlaps"] = overlaps_new
        return prop_data


@dataclass
class fp_sampler_gen(fp_sampling.fp_sampler):
    """
    `fp_sampler` with two changes.

    1.  The N_e-column phase redistribution, so a fully polarised system works.

    2.  The per-step renormalisation `W <- n_walkers * W / sum(W)` is *recorded*
        rather than discarded.  That factor is the trajectory's accumulated
        amplitude: it cancels inside one block (the block energy is a ratio, and
        the factor is common to every walker), but it is exactly what weights one
        trajectory against another.  Dropping it biases every average over
        trajectories.  We keep the renormalisation -- without it the weights
        overflow -- and accumulate log(sum(W)/n_walkers) in `prop_data["log_norm"]`,
        so no information is lost.

    `sr_every` sets the stochastic-reconfiguration cadence in steps (1 = every
    step, as in `afqmc`; 0 = never).  SR conserves the total weight, so it does not
    interfere with the bookkeeping -- but reconfiguring every step forces every
    walker weight to the mean, which is strong population control and carries its
    own O(1/n_walkers) bias.
    """

    sr_every: int = 1

    def __hash__(self) -> int:
        # @dataclass regenerates __eq__ and drops __hash__; both must see sr_every
        # or jit will silently reuse a trace made for a different cadence.
        return hash(tuple(self.__dict__.values()))

    @partial(jit, static_argnums=(0, 4, 5))
    def _step_scan(self, prop_data, fields, ham_data, prop, trial, wave_data):
        prop_data = prop.propagate_free(trial, ham_data, prop_data, fields)

        prop_data["walkers"], norms = linalg_utils.qr_vmap_uhf(prop_data["walkers"])
        norms = norms[0] * norms[1]
        norms_abs = jnp.abs(norms)
        phase = norms / norms_abs

        n_walkers = int(prop_data["weights"].shape[0])
        prop_data["weights"] = prop_data["weights"] * norms_abs

        total = jnp.sum(prop_data["weights"])
        prop_data["log_norm"] = prop_data["log_norm"] + jnp.log(total / n_walkers)
        prop_data["weights"] = n_walkers * prop_data["weights"] / total

        n_tot = trial.nelec[0] + trial.nelec[1]
        c = phase ** (1.0 / n_tot)
        prop_data["walkers"] = prop._multiply_constant(prop_data["walkers"], jnp.stack((c, c)))

        prop_data["step"] = prop_data["step"] + 1
        if self.sr_every:
            do_sr = (prop_data["step"] % self.sr_every) == 0
            keep_w = [prop_data["walkers"][0], prop_data["walkers"][1]]
            keep_wt = prop_data["weights"]
            prop_data = prop.stochastic_reconfiguration_local(prop_data)
            prop_data["walkers"] = [jnp.where(do_sr, prop_data["walkers"][k], keep_w[k])
                                    for k in (0, 1)]
            prop_data["weights"] = jnp.where(do_sr, prop_data["weights"], keep_wt)
        return prop_data, fields


_PROPAGATORS = {"cp": propagator_cp, "constrained": propagator_cp,
                "phaseless": propagator_uhf_gen, "ph": propagator_uhf_gen}


# ================================================================================
# 4.  Results
# ================================================================================

@dataclass
class Snapshot:
    """A frozen walker population, taken at the end of a block."""
    run: int                    # trajectory (FP) or run (CP) index
    tau: float
    walkers: np.ndarray         # alpha determinants, (n_walkers, norb, n_up)
    walkers_dn: np.ndarray      # beta  determinants, (n_walkers, norb, n_dn)
    weights: np.ndarray         # (n_walkers,)
    overlaps: np.ndarray        # <Psi_T|phi_w>, (n_walkers,)
    importance: bool            # True if importance sampled (needs 1/overlap de-weighting)
    log_norm: float = 0.0       # log of the renormalisation this trajectory has
                                # accumulated; the walker weights are relative, so
                                # this is what sets the scale of one snapshot
                                # against another (free projection only)


@dataclass
class QMCResult:
    """Block energies plus whatever walker snapshots were requested."""
    method: str
    tau: np.ndarray                       # (n_blocks,)
    energy: np.ndarray                    # (n_blocks,) averaged over runs
    error: np.ndarray                     # (n_blocks,) scatter between runs
    block_energies: np.ndarray            # (n_runs, n_blocks) raw; n_runs = 1 for one walk
    block_weights: Optional[np.ndarray]   # (n_runs, n_blocks) block weights
    snaps: list = field(default_factory=list)
    log_amplitude: Optional[np.ndarray] = None   # (n_traj, n_samples), free projection
    ess: Optional[np.ndarray] = None             # effective no. of trajectories vs tau
    tau_eq: float = 0.0
    n_killed: int = 0
    walltime: float = 0.0
    prop_data: Any = None

    # -- convenience ------------------------------------------------------------
    @property
    def n_runs(self) -> int:
        return self.block_energies.shape[0]

    def snap_taus(self) -> np.ndarray:
        return np.array(sorted({round(s.tau, 8) for s in self.snaps}))

    def snaps_at(self, tau: float, atol: float = 1e-6) -> list:
        return [s for s in self.snaps if abs(s.tau - tau) < atol]

    def plateau(self, tau_eq: Optional[float] = None, n_seg: int = 1,
                verbose: bool = False):
        """
        Converged energy and its error.

        One walk (constrained path / phaseless): the blocks are a single
        autocorrelated series already past equilibration, so the error comes from
        a blocking analysis -- `sampling.blocking`, the same routine
        `afqmc/scripts/run_afqmc.py` uses.  `verbose` lets its table through.

        Several independent runs (free projection): mean and error from their
        scatter.  Block energies inside one run are autocorrelated, so the naive
        per-block standard error is several times too small; averaging whole runs
        (optionally split into `n_seg` contiguous segments) avoids that.
        """
        if self.block_energies.shape[0] == 1 and self.block_weights is not None:
            _, e, err = blocking_estimate(self.block_weights[0],
                                          self.block_energies[0], verbose=verbose)
            return float(e), float(err)
        tau_eq = self.tau_eq if tau_eq is None else tau_eq
        mask = self.tau > tau_eq
        if not mask.any():
            raise ValueError("no blocks with tau > %g" % tau_eq)
        segs = np.array_split(self.block_energies[:, mask], n_seg, axis=1)
        means = np.concatenate([s.mean(axis=1) for s in segs])
        err = means.std(ddof=1) / np.sqrt(means.size) if means.size > 1 else np.nan
        return float(means.mean()), float(err)


def blocking_estimate(weights, energies, min_nblocks: int = 20,
                      zeta: float = 30.0, verbose: bool = False):
    """
    `sampling.blocking` on one walk's block series, outliers filtered first.

    Returns (mean weight, energy, error).  Both library routines print
    unconditionally, so their output is swallowed unless `verbose`.
    """
    weights = np.asarray(weights, dtype=float)
    energies = np.asarray(energies, dtype=float)

    # An exact trial makes every local energy identical, so the block series is
    # constant: its MAD is 0, `filter_outliers` then rejects everything, and the
    # blocking fit has nothing to fit.  The error of a constant series is 0.
    if energies.size and np.ptp(energies) < 1e-14 * max(1.0, abs(energies[0])):
        return float(np.mean(weights)), float(energies[0]), 0.0

    def _binned_error(w, e, n_bins=20):
        """Fallback: plain binned standard error over `n_bins` contiguous bins."""
        n = (len(e) // n_bins) * n_bins
        if n < 2 * n_bins:
            return float(np.std(e, ddof=1) / np.sqrt(len(e))) if len(e) > 1 else np.nan
        wb = w[:n].reshape(n_bins, -1)
        eb = e[:n].reshape(n_bins, -1)
        means = (wb * eb).sum(1) / wb.sum(1)
        return float(means.std(ddof=1) / np.sqrt(n_bins))

    def _run():
        mask = sampling.filter_outliers(energies, zeta=zeta)
        if not mask.any():                       # degenerate spread; keep everything
            mask = np.ones_like(mask)
        w, e = weights[mask], energies[mask]
        try:
            return sampling.blocking(w, e, min_nblocks=min_nblocks, final=True)
        except Exception as exc:                 # the plateau fit can fail outright
            print("blocking fit failed (%s); falling back to a binned error" % exc)
            return float(np.mean(w)), float((w * e).sum() / w.sum()), _binned_error(w, e)

    if verbose:
        return _run()
    with contextlib.redirect_stdout(io.StringIO()):
        return _run()


def plateau_estimate(raw: np.ndarray, discard: float = 0.5, n_seg: int = 1):
    """Same as `QMCResult.plateau`, for a bare (n_runs, n_blocks) array."""
    n_runs, n_blocks = raw.shape
    tail = raw[:, int(discard * n_blocks):]
    segs = np.array_split(tail, n_seg, axis=1)
    means = np.concatenate([s.mean(axis=1) for s in segs])
    err = means.std(ddof=1) / np.sqrt(means.size) if means.size > 1 else np.nan
    return float(means.mean()), float(err)


# ================================================================================
# 5.  Walker analysis
#
# Applies whenever every walker is a single Slater determinant with at most one
# occupied orbital per spin: one electron on N sites (the CI basis is the sites,
# |100...>, |010...>, ...), or one electron of each spin on two sites (the CI basis
# is the four |i_up j_dn>).
# ================================================================================

# How many angles parametrise a normalised real CI vector of this length, and what
# to call them.  A determinant is a ray, so n components need n - 1 angles.
_ANGLE_NAMES = {2: ("theta",), 3: ("theta", "phi")}


def _spin_amplitudes(x):
    """
    Site amplitudes of a walker population, one occupied orbital per spin channel.

    Accepts a Snapshot or a (walkers_up, walkers_dn) pair.  Returns (up, dn) with
    shape (n_walkers, norb) each; `dn` is None when the beta channel is empty.
    """
    if isinstance(x, Snapshot):
        up, dn = x.walkers, x.walkers_dn
    else:
        up, dn = x if isinstance(x, (tuple, list)) else (x, None)

    def flat(w, name):
        if w is None:
            return None
        w = np.asarray(w)
        if w.ndim == 3:
            if w.shape[2] == 0:
                return None
            if w.shape[2] != 1:
                raise ValueError("angle parametrisation needs at most one occupied "
                                 "orbital per spin; %s has %d" % (name, w.shape[2]))
            w = w[:, :, 0]
        if np.iscomplexobj(w):
            if np.abs(w.imag).max() > 1e-10:
                raise ValueError("walkers are not real -- charge decomposition?")
            w = w.real
        return w

    return flat(up, "alpha"), flat(dn, "beta")


def walker_angles(x):
    """
    Parametrise a two-site walker by one angle per spin channel:

        |phi_w> = (cos(alpha), sin(alpha))  x  (cos(beta), sin(beta))

    Returns (alpha, beta); beta is None for a fully polarised system.  Two sites
    only -- with more, use `walker_ci_angles`.
    """
    up, dn = _spin_amplitudes(x)
    if up.shape[1] != 2:
        raise ValueError("walker_angles is the two-site parametrisation; %d sites "
                         "-- use walker_ci_angles" % up.shape[1])
    alpha = np.arctan2(up[:, 1], up[:, 0])
    beta = None if dn is None else np.arctan2(dn[:, 1], dn[:, 0])
    return alpha, beta


def angles_of(c) -> np.ndarray:
    r"""
    Spherical angles of a real one-electron CI vector, in the conventions used by
    the notebooks in this directory:

        2 sites :  c = (cos t, sin t)                    ->  (t,)
        3 sites :  c = (sin t cos p, sin t sin p, cos t) ->  (t, p)

    `c` may be a single vector, shape (n,), or a stack of them, shape (m, n); the
    angles come back in the last axis.  The vector is normalised first, so an
    unnormalised CI vector is fine.
    """
    arr = np.asarray(c, dtype=float)
    single = arr.ndim == 1
    m = np.atleast_2d(arr)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    if m.shape[1] == 2:
        out = np.arctan2(m[:, 1], m[:, 0])[:, None]
    elif m.shape[1] == 3:
        out = np.stack([np.arccos(np.clip(m[:, 2], -1.0, 1.0)),
                        np.arctan2(m[:, 1], m[:, 0])], axis=1)
    else:
        raise ValueError("angles_of parametrises 2 or 3 components, got %d" % m.shape[1])
    return out[0] if single else out


def walker_ci_angles(x) -> np.ndarray:
    """
    Per-walker angles of a one-electron population, shape (n_walkers, n_sites - 1).

    Accepts a Snapshot or a (walkers_up, walkers_dn) pair, like `ci_vectors`.
    """
    return angles_of(ci_vectors(x))


def theta_of(x) -> np.ndarray:
    """Angle of the alpha determinant, |phi> = cos(theta)|1> + sin(theta)|2>."""
    return walker_angles(x)[0]


def gauge_fix(theta: np.ndarray) -> np.ndarray:
    """
    Map an angle to the representative with a positive first component, i.e. to
    (-pi/2, pi/2].

    A determinant is a ray: for one electron (W, phi) and (W, -phi) are the same
    walker, and `orthonormalize_walkers` discards the sign returned by the QR, so
    constrained-path walkers routinely come back globally flipped.  With two spin
    channels only the *simultaneous* flip (up, dn) -> (-up, -dn) is a gauge
    transformation; flipping one channel alone flips the sign of the CI vector.
    """
    return (np.asarray(theta) + np.pi / 2) % np.pi - np.pi / 2


def ci_vectors(x) -> np.ndarray:
    """
    CI vector of every walker in the determinant basis, shape (n_walkers, n_det).

    One electron on two sites      : (cos a, sin a).
    One electron of each spin      : the outer product, in the basis
                                     |i_up j_dn> ordered (0,0), (0,1), (1,0), (1,1),
                                     i.e. [ca*cb, ca*sb, sa*cb, sa*sb].
    """
    up, dn = _spin_amplitudes(x)
    if dn is None:
        return up
    n = up.shape[0]
    return np.einsum("wi,wj->wij", up, dn).reshape(n, -1)


def reconstruct(x, weights=None, overlaps=None) -> np.ndarray:
    """
    Unnormalised CI coefficients of the QMC wavefunction.

    Free projection carries no importance sampling, so the ensemble is the
    wavefunction directly,  |Psi> = sum_w W_w |phi_w>.  Phaseless/CP sample the
    mixed distribution f(phi) = <Psi_T|phi> Psi(phi), so the trial overlap has to
    be divided back out,  |Psi> = sum_w (W_w / <Psi_T|phi_w>) |phi_w>.  Passing a
    `Snapshot` picks the right one automatically.
    """
    if isinstance(x, Snapshot):
        weights = x.weights
        overlaps = x.overlaps if x.importance else None
    c = ci_vectors(x)
    if weights is None:
        weights = np.ones(c.shape[0])
    coef = weights if overlaps is None else weights / overlaps
    return (coef[:, None] * c).sum(0)


def summarize(c: np.ndarray, ham: np.ndarray, v_trial: np.ndarray) -> dict:
    """
    Normalise a CI vector, fix its overall sign, and report both energies.

    E_var = <Psi|H|Psi> / <Psi|Psi>       -- second order in the error of c
    E_mix = <Psi_T|H|Psi> / <Psi_T|Psi>   -- what the sampler reports, first order

    `ham` is the Hamiltonian in the same determinant basis (h1 for one electron,
    `dimer_ci(...)[0]` for one electron of each spin); `v_trial` the trial's CI
    vector.  The sign is fixed by demanding <Psi_T|Psi> > 0.
    """
    cc = np.asarray(c, dtype=float).ravel()
    cc = cc / np.linalg.norm(cc)
    v = np.asarray(v_trial, dtype=float).ravel()
    if v @ cc < 0:
        cc = -cc
    out = dict(c=cc, e_var=float(cc @ ham @ cc), e_mix=float((v @ ham @ cc) / (v @ cc)))
    for name, a in zip(_ANGLE_NAMES.get(cc.size, ()), np.atleast_1d(angles_of(cc))):
        out[name] = float(a)
    return out


def wavefunction(snaps: Sequence[Snapshot], ham, v_trial, group_by_run: bool = True):
    """
    Reconstruct |Psi> from a list of snapshots, all of which must describe the
    same state (one tau for free projection; any set of post-equilibration times
    for constrained path).

    Snapshots are combined by their *amplitude*, not after normalising each one:
    the walker weights inside a snapshot are relative, and `Snapshot.log_norm`
    carries the scale that free projection has divided out along the way.
    Normalising first and averaging afterwards would reintroduce exactly the bias
    that `fp_sampler_gen` avoids.

    The error bar is a jackknife over statistically independent walks (snapshots
    along one run are correlated, so they are pooled first).
    """
    if len(snaps) == 0:
        raise ValueError("no snapshots -- pass snap_every > 0 to the driver")
    ham = np.asarray(ham)
    v = np.asarray(v_trial, dtype=float).ravel()

    cs = np.array([reconstruct(sn) for sn in snaps], dtype=float)     # (n_snaps, n_det)
    logs = np.array([sn.log_norm for sn in snaps], dtype=float)
    cs = np.exp(logs - logs.max())[:, None] * cs                     # common scale

    runs = np.array([sn.run for sn in snaps])
    groups = np.unique(runs)
    g = (np.array([cs[runs == r].sum(axis=0) for r in groups])
         if group_by_run else cs)                                    # (n_groups, n_det)

    def norm_fix(x):
        x = x / np.linalg.norm(x)
        return -x if v @ x < 0 else x

    c_mean = norm_fix(g.sum(axis=0))
    n = g.shape[0]
    if n > 1:
        jk = np.array([norm_fix(g.sum(axis=0) - g[j]) for j in range(n)])
        c_err = np.sqrt((n - 1) / n * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    else:
        c_err = np.full_like(c_mean, np.nan)

    out = dict(c=c_mean, c_err=c_err,
               e_var=float(c_mean @ ham @ c_mean),
               e_mix=float((v @ ham @ c_mean) / (v @ c_mean)),
               n_snaps=len(snaps), n_groups=int(n))
    names = _ANGLE_NAMES.get(c_mean.size, ())
    if names:
        ang = np.atleast_1d(angles_of(c_mean))
        ang_jk = np.atleast_2d(angles_of(jk)) if n > 1 else None
        out["angles"] = ang
        for k, name in enumerate(names):
            out[name] = float(ang[k])
            if n > 1:
                # wrap-safe deviations, so an angle sitting near the arctan2 branch
                # cut does not blow the jackknife up
                d = np.angle(np.exp(1j * (ang_jk[:, k] - ang[k])))
                out[name + "_err"] = float(np.sqrt((n - 1) / n
                                                   * ((d - d.mean()) ** 2).sum()))
            else:
                out[name + "_err"] = np.nan
    for i in range(c_mean.size):
        out["c%d" % (i + 1)] = float(c_mean[i])
        out["c%d_err" % (i + 1)] = float(c_err[i])
    return out


# ================================================================================
# 6.  Exact references for the two-site model
# ================================================================================

def dimer_ci(t: float, u: float, nelec):
    """
    Full CI Hamiltonian of the 2-site Hubbard model in the determinant basis, plus
    a function mapping a Slater determinant (mo_a, mo_b) to its CI vector.
    """
    h = np.array([[0.0, -t], [-t, 0.0]])
    if tuple(nelec) == (1, 0):
        return h, (lambda ca, cb: np.asarray(ca).ravel())
    if tuple(nelec) == (1, 1):
        # basis |i_up j_dn>, i,j = 0,1  ->  (0,0), (0,1), (1,0), (1,1)
        H = (np.kron(h, np.eye(2)) + np.kron(np.eye(2), h)
             + u * np.diag([1.0, 0.0, 0.0, 1.0]))
        return H, (lambda ca, cb: np.kron(np.asarray(ca).ravel(), np.asarray(cb).ravel()))
    raise NotImplementedError("dimer_ci supports nelec = (1,0) or (1,1)")


def exact_mixed_estimator(H, v_trial, v_init, taus) -> np.ndarray:
    r"""E_mix(tau) = <Psi_T|H e^{-tau H}|Phi_0> / <Psi_T|e^{-tau H}|Phi_0>."""
    w, V = np.linalg.eigh(H)
    out = []
    for tau in taus:
        P = V @ np.diag(np.exp(-tau * w)) @ V.T
        out.append((v_trial @ H @ P @ v_init) / (v_trial @ P @ v_init))
    return np.array(out)


# ================================================================================
# 7.  Drivers
# ================================================================================

def _real_if_possible(w):
    w = np.asarray(w)
    if np.iscomplexobj(w) and (w.size == 0 or np.abs(w.imag).max() < 1e-12):
        w = w.real
    return w


def _snapshot(prop_data, run, tau, importance) -> Snapshot:
    return Snapshot(run=int(run), tau=float(tau),
                    walkers=_real_if_possible(prop_data["walkers"][0]).copy(),
                    walkers_dn=_real_if_possible(prop_data["walkers"][1]).copy(),
                    weights=np.asarray(prop_data["weights"]).real.copy(),
                    overlaps=np.asarray(prop_data["overlaps"]).real.copy(),
                    importance=bool(importance),
                    log_norm=float(np.real(prop_data.get("log_norm", 0.0))))


def _can_analyse(system: HubbardSystem) -> bool:
    """
    The CI parametrisation needs at most one occupied orbital per spin: either a
    single electron on any number of sites, or nelec = (1,1) on two sites.
    """
    return (sum(system.nelec) == 1
            or (system.norb == 2 and tuple(system.nelec) == (1, 1)))


def _snap_line(snaps, system: HubbardSystem) -> str:
    """One line describing |Psi> reconstructed from one or more snapshots."""
    if isinstance(snaps, Snapshot):
        snaps = [snaps]
    r = wavefunction(snaps, system.ci_hamiltonian, system.ci_trial)
    c = ", ".join("%+.4f" % x for x in r["c"])
    names = _ANGLE_NAMES.get(r["c"].size, ())
    if names:
        ang = "  ".join(
            "%s = %+.5f%s" % (nm, r[nm], "" if np.isnan(r[nm + "_err"])
                              else " +/- %.5f" % r[nm + "_err"])
            for nm in names)
        return ("        |Psi> :  %s   c = (%s)   E_var = %+.6f"
                % (ang, c, r["e_var"]))
    return "        |Psi> :  c = (%s)   E_var = %+.6f" % (c, r["e_var"])


def run_free_projection(system: HubbardSystem, *,
                        dt=0.0025, n_walkers=500, n_prop_steps=40, n_blocks=30,
                        n_traj=16, seed=0, n_exp_terms=10, sr_every=1,
                        snap_every=0, verbose=0) -> QMCResult:
    """
    Free-projection AFQMC: no force bias, no constraint, exact but with a variance
    that grows with tau.  The estimator is the mixed estimator with the trial
    overlap kept explicitly in the weight, averaged over independent trajectories.

    The returned arrays have n_blocks + 1 entries: entry 0 is the trial at tau = 0,
    entry b + 1 is block b, following afqmc/scripts/run_fpafqmc.py.

    Trajectories are combined by their accumulated amplitude
    A_i = <Psi_T|exp(-tau H)|Phi_0>_i, carried in logs, not by their block weight
    alone -- see `fp_sampler_gen` and `_fp_estimate`.

    sr_every   : stochastic-reconfiguration cadence in propagation steps.  1 (the
                 default, and what `afqmc` does) resets every walker weight to the
                 mean at every step: strong population control, small error bars,
                 an O(1/n_walkers) bias.  Larger values, or 0 for never, give a
                 truer free projection at the cost of a collapsing effective
                 sample size -- watch the ESS column.
    snap_every : store a walker snapshot every this many blocks, in every
                 trajectory (0 = none).  Also sets which tau rows get printed.
                 No snapshot is taken at tau = 0, where every walker is the trial.
    verbose    : 0 silent; 1 the running E(tau) table (throttled to every
                 n_traj // 10 trajectories); 2 also the reconstructed |Psi> at each
                 snapshot (one electron on two sites only).
    """
    prop = propagator_uhf_gen(dt=dt, n_walkers=n_walkers,
                              n_exp_terms=n_exp_terms, n_batch=1)
    hd = system.ham.build_propagation_intermediates(
        dict(system.ham_data), prop, system.trial, system.wave_data)
    smp = fp_sampler_gen(n_prop_steps=n_prop_steps, n_eql_blocks=n_blocks,
                         n_trj=n_traj, n_chol=system.nchol, sr_every=sr_every)

    # Sample 0 is the trial at tau = 0, as in afqmc/scripts/run_fpafqmc.py, so every
    # sample array has n_blocks + 1 rows and row b + 1 holds block b.
    tau = dt * n_prop_steps * np.arange(0, n_blocks + 1)
    row_every = snap_every if snap_every else max(1, n_blocks // 10)
    rows = list(range(0, n_blocks + 1, row_every))
    if rows[-1] != n_blocks:
        rows.append(n_blocks)
    traj_every = max(1, n_traj // 10)          # as in afqmc/scripts/run_fpafqmc.py
    analyse = verbose >= 2 and _can_analyse(system)
    t0 = time.time()

    # log|A| and sign(A) of each trajectory's <Psi_T|exp(-tau H)|Phi_0>, plus the
    # raw block weights and energies.  (n_samples, n_traj) throughout.
    log_amp = np.zeros((n_blocks + 1, n_traj))
    sgn = np.ones((n_blocks + 1, n_traj))
    ws = np.zeros((n_blocks + 1, n_traj))
    es = np.zeros((n_blocks + 1, n_traj))
    snaps, pd = [], None

    for i, key in enumerate(random.split(random.PRNGKey(seed), n_traj)):
        pd = prop.init_prop_data(system.trial, system.wave_data, hd, None)
        pd["key"] = key
        pd["log_norm"] = jnp.array(0.0)        # accumulated renormalisation, in logs
        pd["step"] = jnp.array(0)

        w0 = float(np.real(jnp.sum(pd["weights"] * pd["overlaps"])))
        ws[0, i] = w0
        es[0, i] = float(np.real(pd["e_estimate"]))
        log_amp[0, i] = np.log(abs(w0))
        sgn[0, i] = np.sign(w0)

        # A Python loop over `fp_block` rather than `scan_eql_blocks`: bit-identical
        # results for ~4% more time, and it keeps the intermediate walkers reachable
        # so that snap_every can work.
        for b in range(n_blocks):
            pd, (blk_w, blk_e) = smp.fp_block(pd, hd, prop, system.trial,
                                              system.wave_data)
            w = float(np.real(complex(blk_w)))
            ws[b + 1, i] = w
            es[b + 1, i] = float(np.real(complex(blk_e)))
            log_amp[b + 1, i] = float(pd["log_norm"]) + np.log(abs(w))
            sgn[b + 1, i] = np.sign(w)
            if snap_every and (b + 1) % snap_every == 0:
                snaps.append(_snapshot(pd, i, tau[b + 1], importance=False))

        if verbose and (i == 0 or (i + 1) % traj_every == 0 or i == n_traj - 1):
            amp_i, mean_i, err_i, ess_i = _fp_estimate(
                log_amp[:, :i + 1], es[:, :i + 1], sgn[:, :i + 1])
            print("\nFree projection | trajectories 1-%d of %d | %.1f s"
                  % (i + 1, n_traj, time.time() - t0))
            print("  %6s  %10s  %10s  %9s  %6s"
                  % ("tau", "amplitude", "Energy", "Error", "ESS"))
            for b in rows:
                e_str = "%9s" % "N/A" if np.isnan(err_i[b]) else "%9.5f" % err_i[b]
                print("  %6.2f  %10.3e  %10.5f  %s  %6.1f"
                      % (tau[b], amp_i[b], mean_i[b], e_str, ess_i[b]))
                if analyse and snap_every:
                    # running estimate over every trajectory finished so far, so
                    # this line matches the energy column above it
                    hit = [sn for sn in snaps if abs(sn.tau - tau[b]) < 1e-9]
                    if hit:
                        print(_snap_line(hit, system))

    amp, mean, err, ess = _fp_estimate(log_amp, es, sgn)

    if verbose:
        print("\n" + "=" * 68)
        print("  Free projection | %d trajectories x %d blocks | sr_every = %d | %.1f s"
              % (n_traj, n_blocks, sr_every, time.time() - t0))
        print("  E(tau = %.2f) = %+.5f +/- %.5f" % (tau[-1], mean[-1], err[-1]))
        print("  amplitude relative to tau = 0 : %.3e | effective trajectories: %.1f / %d"
              % (amp[-1], ess[-1], n_traj))
        if ess[-1] < 0.5 * n_traj:
            print("  WARNING: the trajectory amplitudes have spread out; the free")
            print("  projection is losing statistical power at this tau.")
        if analyse and snaps:
            tail = [sn for sn in snaps if sn.tau >= tau[-1] - 1e-9]
            if tail:
                print("  |Psi> at tau = %.2f :" % tail[0].tau)
                print(_snap_line(tail, system))
        print("=" * 68)

    return QMCResult(method="free projection", tau=tau,
                     energy=mean, error=err,
                     block_energies=es.T,               # (n_traj, n_samples)
                     block_weights=ws.T,
                     log_amplitude=log_amp.T, ess=ess,
                     snaps=snaps, tau_eq=0.0, n_killed=0,
                     walltime=time.time() - t0, prop_data=pd)


def _fp_estimate(log_amp, es, sign=None):
    """
    Free-projection mixed estimator, amplitude-weighted over trajectories.

        E(tau) = sum_i A_i(tau) e_i(tau) / sum_i A_i(tau)

    where A_i is trajectory i's unnormalised <Psi_T|exp(-tau H)|Phi_0> and e_i its
    block energy.  A_i spans many orders of magnitude, so it is carried as
    log|A_i| (+ a sign) and the ratio is formed after subtracting the largest
    exponent.

    log_amp, es, sign : (n_samples, n_traj).  Trajectories are the SECOND axis, so
    a running estimate over the first i+1 of them slices as `log_amp[:, :i+1]`.

    Returns (relative amplitude, energy, error, effective no. of trajectories).
    The error is a jackknife over trajectories: for a ratio of sums with unequal
    weights, `std(e)/sqrt(n)` is not the right error.
    """
    log_amp = np.atleast_2d(np.real(log_amp))
    es = np.atleast_2d(np.real(es))
    n_traj = log_amp.shape[1]
    sign = np.ones_like(log_amp) if sign is None else np.atleast_2d(np.real(sign))

    a = sign * np.exp(log_amp - log_amp.max(axis=1, keepdims=True))    # (n_samples, n_traj)
    num, den = (a * es).sum(1), a.sum(1)
    mean = num / den

    # amplitude relative to tau = 0, for display: how fast the projection decays
    amp = np.exp(log_amp - log_amp[0].max()).mean(axis=1)

    if n_traj > 1:
        jk = (num[:, None] - a * es) / (den[:, None] - a)              # leave one out
        err = np.sqrt((n_traj - 1) / n_traj
                      * ((jk - jk.mean(axis=1, keepdims=True)) ** 2).sum(axis=1))
    else:
        err = np.full(mean.shape, np.nan)

    ess = np.abs(a).sum(1) ** 2 / (a ** 2).sum(1)                      # Kish
    return amp, mean, err, ess


def run_constrained(system: HubbardSystem, *,
                    method="cp", dt=0.0025, n_walkers=500, n_prop_steps=40,
                    n_blocks=100, eql_time=5.0, seed=0, n_exp_terms=10,
                    n_bins=10, snap_every=0, verbose=0) -> QMCResult:
    """
    Constrained-path ("cp") or phaseless ("phaseless") AFQMC.

    This is ONE open-ended random walk, laid out like `afqmc/scripts/run_afqmc.py`:
    equilibrate for `eql_time` of imaginary time, throw those blocks away, then keep
    sampling block after block from the same walk.  There is nothing to restart --
    importance sampling gives the walk a steady state, so a longer walk is strictly
    better than several short ones, which would each pay the equilibration cost again.
    (Free projection is the opposite case and does use independent trajectories: its
    weights decay, so there is no steady state to sample from.)

    Consecutive blocks are correlated, so the error bar is a blocking analysis of the
    single series, `sampling.blocking` -- see `QMCResult.plateau`.

    eql_time   : imaginary time discarded before sampling starts.
    n_blocks   : sampling blocks kept, after equilibration.
    n_bins     : snapshots are labelled with the index of the contiguous bin of the
                 walk they came from, so that `wavefunction` jackknifes over bins
                 long compared with the autocorrelation time instead of over
                 individual (correlated) snapshots.
    snap_every : store a walker snapshot every this many sampling blocks (0 = none).
                 Also sets the progress-reporting cadence.
    verbose    : 0 silent; 1 progress lines plus a final summary; 2 also print the
                 reconstructed |Psi> at every snapshot; 3 also the blocking table.
    """
    prop_cls = _PROPAGATORS[method] if isinstance(method, str) else method
    label = "Constrained path" if prop_cls is propagator_cp else "Phaseless"

    prop = prop_cls(dt=dt, n_walkers=n_walkers, n_exp_terms=n_exp_terms, n_batch=1)
    hd = system.ham.build_propagation_intermediates(
        dict(system.ham_data), prop, system.trial, system.wave_data)
    smp = sampling.sampler(n_prop_steps=n_prop_steps, n_blocks=n_blocks,
                           n_chol=system.nchol)

    block_time = dt * n_prop_steps
    n_eql = int(-(-eql_time // block_time))          # ceil
    tau_eql = n_eql * block_time
    tau = tau_eql + block_time * np.arange(1, n_blocks + 1)
    row_every = snap_every if snap_every else max(1, n_blocks // 10)
    analyse = verbose >= 2 and _can_analyse(system)
    t0 = time.time()

    pd = prop.init_prop_data(system.trial, system.wave_data, hd, None)
    pd["key"] = random.PRNGKey(seed)
    pd["n_killed_walkers"] = 0

    # ---- equilibration, discarded -------------------------------------------
    if verbose:
        print("\n%s | %d walkers | equilibrating %d blocks to tau = %.2f"
              % (label, n_walkers, n_eql, tau_eql))
    for b in range(n_eql):
        pd, (wt, en, er) = smp.block_sample(pd, hd, prop, system.trial, system.wave_data)
        if verbose and ((b + 1) % max(1, n_eql // 5) == 0 or b == n_eql - 1):
            print("  eql %4d/%d  tau = %6.2f  E(block) = %+10.5f  %6.1f s"
                  % (b + 1, n_eql, (b + 1) * block_time, float(np.real(en)),
                     time.time() - t0))
    pd["n_killed_walkers"] = 0                       # only count kills while sampling

    # ---- sampling, one continuing walk --------------------------------------
    all_w = np.zeros(n_blocks)
    all_e = np.zeros(n_blocks)
    all_err = np.zeros(n_blocks)
    snaps = []

    if verbose:
        print("\n%s | sampling %d blocks from the same walk" % (label, n_blocks))
        print("  %5s  %6s  %10s  %10s  %9s  %8s"
              % ("block", "tau", "E(block)", "<E>", "err", "walltime"))

    for b in range(n_blocks):
        pd, (wt, en, er) = smp.block_sample(pd, hd, prop, system.trial, system.wave_data)
        all_w[b] = float(np.real(wt))
        all_e[b] = float(np.real(en))
        all_err[b] = float(np.real(er))

        take_snap = snap_every and (b + 1) % snap_every == 0
        if take_snap:
            # label by contiguous bin of the walk, not by "run"
            snaps.append(_snapshot(pd, min(b * n_bins // n_blocks, n_bins - 1),
                                   tau[b], importance=True))

        if verbose and ((b + 1) % row_every == 0 or b == n_blocks - 1):
            m, e = blocking_estimate(all_w[:b + 1], all_e[:b + 1])[1:] \
                if b > 20 else (np.mean(all_e[:b + 1]), np.nan)
            e_str = "%9s" % "--" if np.isnan(e) else "%9.5f" % e
            print("  %5d  %6.2f  %10.5f  %10.5f  %s  %8.1f"
                  % (b + 1, tau[b], all_e[b], m, e_str, time.time() - t0))
            if analyse and take_snap:
                print(_snap_line(snaps[-1], system))

    n_killed = int(pd["n_killed_walkers"])

    result = QMCResult(method=label.lower(), tau=tau,
                       energy=all_e,
                       error=all_err,
                       block_energies=all_e[None, :], block_weights=all_w[None, :],
                       snaps=snaps, tau_eq=float(tau_eql), n_killed=n_killed,
                       walltime=time.time() - t0, prop_data=pd)

    if verbose:
        m, s_err = result.plateau(verbose=verbose >= 3)
        naive = all_e.std(ddof=1) / np.sqrt(n_blocks)
        print("\n" + "=" * 68)
        print("  %s | %d blocks sampled after tau_eq = %.2f | %.1f s"
              % (label, n_blocks, tau_eql, result.walltime))
        print("  E = %+.5f +/- %.5f   (blocking analysis of the single walk)" % (m, s_err))
        print("  the uncorrelated per-block error would be %.5f; consecutive blocks" % naive)
        print("  are autocorrelated, so that one is several times too small.")
        print("  walkers killed by the constraint: %d" % n_killed)
        if analyse and snaps:
            rr = wavefunction(snaps, system.ci_hamiltonian, system.ci_trial)
            print("  |Psi> from %d snapshots in %d bins :" % (rr["n_snaps"], rr["n_groups"]))
            print(_snap_line(snaps, system))
        print("=" * 68)

    return result
