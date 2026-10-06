"""Switching bifurcation ODE data generator.

Four low-dimensional systems are provided, each with a single bifurcation
parameter that can be switched between segments:

    hopf            — Stuart-Landau normal form.
                      μ < 0 : stable fixed point (spiral into origin)
                      μ > 0 : stable limit cycle of radius √μ
                      Clean Hopf bifurcation, 2-D.

    van_der_pol     — Van der Pol oscillator.
                      μ ≈ 0 : near-harmonic oscillation
                      μ >> 0: relaxation oscillation (sharp spikes)
                      Qualitative waveform change, 2-D.

    fitzhugh_nagumo — FitzHugh-Nagumo neural model.
                      I_ext below ~0.34 : quiescent fixed point
                      I_ext above ~0.34 : spiking limit cycle (Hopf)
                      Biologically interpretable, 2-D.

    lorenz63        — Lorenz (1963) system.
                      ρ controls the qualitative regime:
                        ρ < 1    : stable fixed point at origin
                        1 < ρ < ~24.74 : two stable fixed points (C±)
                        ρ > ~24.74 : chaotic strange attractor
                      σ (default 10) and β (default 8/3) fixed via kwargs.
                      Recommended pairs: [28, 35] (both chaotic, different
                      geometry) or [14, 28] (fixed-point → chaos).  3-D.

Usage::

    from data.switching_ode import generate_switching_ode
    ts, switch_times, segment_ids = generate_switching_ode(
        system="hopf",
        param_values=[-0.5, 0.5],
        segment_lengths=[500, 500],
    )
"""

import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# RK4 integrator (autonomous ODEs only — no explicit time dependence)
# ---------------------------------------------------------------------------

def _rk4_step(f, y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# ODE definitions
# ---------------------------------------------------------------------------

def _hopf(mu: float, omega: float = 1.0):
    """Stuart-Landau normal form (Hopf bifurcation).

    dx/dt = μx − ωy − x(x² + y²)
    dy/dt = ωx + μy − y(x² + y²)

    Fixed point at origin for μ < 0; limit cycle of radius √μ for μ > 0.
    """
    def f(y: np.ndarray) -> np.ndarray:
        x, v = y
        r2 = x * x + v * v
        return np.array([mu * x - omega * v - x * r2,
                         omega * x + mu * v - v * r2])
    return f


def _van_der_pol(mu: float):
    """Van der Pol oscillator.

    dx/dt = y
    dy/dt = μ(1 − x²)y − x

    Small μ: near-harmonic (amplitude ≈ 2, sinusoidal).
    Large μ: relaxation oscillation (fast-slow, non-sinusoidal waveform).
    """
    def f(y: np.ndarray) -> np.ndarray:
        x, v = y
        return np.array([v, mu * (1.0 - x * x) * v - x])
    return f


def _fitzhugh_nagumo(I_ext: float, a: float = 0.7, b: float = 0.8,
                     tau: float = 12.5):
    """FitzHugh-Nagumo model.

    dv/dt = v − v³/3 − w + I_ext
    dw/dt = (v + a − bw) / τ

    Below Hopf (I_ext ≲ 0.34): stable fixed point (quiescent).
    Above Hopf (I_ext ≳ 0.34): stable limit cycle (spiking).
    """
    def f(y: np.ndarray) -> np.ndarray:
        v, w = y
        return np.array([v - v ** 3 / 3.0 - w + I_ext,
                         (v + a - b * w) / tau])
    return f


def _lorenz63(rho: float, sigma: float = 10.0, beta: float = 8.0 / 3.0):
    """Lorenz (1963) system.

    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz

    ρ is the primary bifurcation parameter (switching param).
    σ and β are fixed via system_kwargs.
    """
    def f(y: np.ndarray) -> np.ndarray:
        x, v, z = y
        return np.array([
            sigma * (v - x),
            x * (rho - z) - v,
            x * v - beta * z,
        ])
    return f


_SYSTEMS = {
    "hopf":            _hopf,
    "van_der_pol":     _van_der_pol,
    "fitzhugh_nagumo": _fitzhugh_nagumo,
    "lorenz63":        _lorenz63,
}

# Default initial conditions (before warmup) per system.
# Placed near the expected limit-cycle amplitude so that a short warmup is
# sufficient and both segments have visible dynamics.
#   hopf:        limit-cycle radius = sqrt(mu); for mu=0.5 that is ~0.707
#   van_der_pol: limit-cycle amplitude ≈ 2 for all mu > 0
#   fitzhugh_nagumo: near the resting fixed-point
#   lorenz63:    near the classic attractor; warmup will drive it onto it
_DEFAULT_Y0 = {
    "hopf":            np.array([0.707, 0.0]),
    "van_der_pol":     np.array([2.0,   0.0]),
    "fitzhugh_nagumo": np.array([-1.2, -0.6]),
    "lorenz63":        np.array([1.0,   1.0,  1.0]),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_switching_ode(
    system: str,
    param_values: List[float],
    segment_lengths: List[int],
    dt: float = 0.05,
    warmup: int = 1000,
    noise_std: float = 0.0,
    random_seed: int = 42,
    y0: Optional[np.ndarray] = None,
    **system_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a switching bifurcation ODE and return the time series.

    The bifurcation parameter switches abruptly at segment boundaries.  A
    warmup period is used before the first segment to let the trajectory
    settle onto its natural attractor.  Subsequent segments start from the
    last point of the previous one, so transients at switch times are part
    of the data.

    Parameters
    ----------
    system : {'hopf', 'van_der_pol', 'fitzhugh_nagumo', 'lorenz63'}
    param_values : list of float, length n_segments
        Bifurcation parameter per segment.
        - hopf / van_der_pol : μ
        - fitzhugh_nagumo    : I_ext
        - lorenz63           : ρ (rho); σ and β fixed via system_kwargs
    segment_lengths : list of int, length n_segments
    dt : float
        RK4 integration step size.
    warmup : int
        Steps to integrate (with the first segment's parameter) before
        recording begins, to land the trajectory near the initial attractor.
    noise_std : float
        Standard deviation of i.i.d. Gaussian observation noise.
    random_seed : int
    y0 : array-like or None
        Initial condition before the warmup.  Uses system-specific defaults
        when None.
    **system_kwargs
        Extra keyword arguments forwarded to the ODE factory.
        For 'hopf': omega (default 1.0).
        For 'fitzhugh_nagumo': a, b, tau.

    Returns
    -------
    time_series : ndarray of shape (T, N), float32
        Concatenated trajectory across all segments.
    switch_times : ndarray of int, shape (n_segments - 1,)
        Sample indices where the parameter changes.
    segment_ids : ndarray of int, shape (T,)
        Segment index for every sample.
    """
    if system not in _SYSTEMS:
        raise ValueError(
            f"Unknown system {system!r}. Available: {list(_SYSTEMS)}"
        )
    if len(param_values) != len(segment_lengths):
        raise ValueError("param_values and segment_lengths must have equal length.")

    rng = np.random.default_rng(random_seed)
    factory = _SYSTEMS[system]

    y = np.asarray(_DEFAULT_Y0[system] if y0 is None else y0, dtype=float).copy()

    # Warmup: drive the system onto the first attractor
    f0 = factory(param_values[0], **system_kwargs)
    for _ in range(warmup):
        y = _rk4_step(f0, y, dt)

    segments: List[np.ndarray] = []
    seg_id_arrays: List[np.ndarray] = []
    switch_times: List[int] = []
    t_offset = 0

    for seg_idx, (param, seg_len) in enumerate(zip(param_values, segment_lengths)):
        f = factory(param, **system_kwargs)
        traj = np.empty((seg_len, len(y)), dtype=float)
        for t in range(seg_len):
            traj[t] = y
            y = _rk4_step(f, y, dt)

        if noise_std > 0.0:
            traj += rng.normal(0.0, noise_std, traj.shape)

        segments.append(traj)
        seg_id_arrays.append(np.full(seg_len, seg_idx, dtype=int))
        if seg_idx > 0:
            switch_times.append(t_offset)
        t_offset += seg_len

    time_series = np.concatenate(segments, axis=0).astype(np.float32)
    segment_ids = np.concatenate(seg_id_arrays)
    switch_times_arr = np.array(switch_times, dtype=int)

    return time_series, switch_times_arr, segment_ids
