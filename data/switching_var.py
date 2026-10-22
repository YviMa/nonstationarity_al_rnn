"""Switching VAR(1) data generator — structured network scenarios.

Three scientifically-motivated scenarios plus a random-matrix fallback.

Scenarios
---------
"modular_to_hub"
    Segment 1: block-modular connectivity (two clusters with weak cross-coupling).
    Segment 2: hub structure — node 0 drives all others strongly; intra-cluster
    coupling is preserved but cross-cluster coupling is weakened in favour of the
    hub.  Both segments share the same spectral radius so marginal variances and
    autocorrelation time-scales are nearly identical; only the connectivity
    topology changes.  This is the scenario where univariate / sliding-window
    tests see nothing but the network geometry has genuinely changed.

"critical_slowing"
    Same eigenvector structure across segments; eigenvalues scaled from sr_low
    to sr_high.  Mimics "critical slowing down" — dynamics become more persistent
    and autocorrelated without a change in oscillation character.  Noise is
    rescaled per segment (σ_i ∝ √(1 − srᵢ²)) so that stationary variance remains
    approximately constant across regimes.

"sign_flip"
    Segment 2 is identical to segment 1 except that the cross-cluster coupling
    block reverses sign (e.g. inhibitory → excitatory cross-talk).  Spectral
    radius is preserved.  Marginal autocorrelation statistics are nearly
    unchanged; the cross-covariance structure flips.

"random"
    Original random structured-matrix approach.  Segments are interpolated via
    switch_strength; any number of segments is supported.
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _block_diag(blocks: list) -> np.ndarray:
    """Stack a list of 2-D arrays into a block-diagonal matrix."""
    n = sum(b.shape[0] for b in blocks)
    D = np.zeros((n, n))
    start = 0
    for b in blocks:
        s = b.shape[0]
        D[start : start + s, start : start + s] = b
        start += s
    return D


def _spectral_rescale(A: np.ndarray, target_sr: float) -> np.ndarray:
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    return A * (target_sr / sr) if sr > 1e-12 else A


def _structured_matrix(
    rng: np.random.Generator,
    n: int,
    spectral_radius: float,
    n_oscillation_modes: int,
) -> np.ndarray:
    """Random VAR(1) matrix with controlled spectral structure (used by 'random')."""
    n_osc  = min(n_oscillation_modes, n // 2)
    n_real = n - 2 * n_osc
    blocks = []
    if n_osc > 0:
        thetas = rng.uniform(np.pi / 8, np.pi * 7 / 8, size=n_osc)
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            blocks.append(spectral_radius * np.array([[c, -s], [s, c]]))
    for _ in range(n_real):
        ev = rng.uniform(-1.0, 1.0) * spectral_radius
        blocks.append(np.array([[ev]]))
    D = _block_diag(blocks)
    R = rng.standard_normal((n, n))
    A = R @ D @ np.linalg.inv(R)
    return _spectral_rescale(A, spectral_radius)


# ---------------------------------------------------------------------------
# Scenario matrix builders
# ---------------------------------------------------------------------------

def _matrices_modular_to_hub(
    rng: np.random.Generator,
    n_nodes: int,
    spectral_radius: float,
    hub_strength: float,
    cross_coupling: float,
) -> List[np.ndarray]:
    """Return [A_modular, A_hub] — same sr, different topology."""
    if n_nodes < 4 or n_nodes % 2 != 0:
        raise ValueError("modular_to_hub requires even n_nodes >= 4")
    half = n_nodes // 2

    # Intra-cluster matrices (slightly sub-critical so composed system hits target sr)
    A1 = _structured_matrix(rng, half, spectral_radius * 0.85, half // 2)
    A2 = _structured_matrix(rng, half, spectral_radius * 0.85, half // 2)

    # Segment 1 — block-modular + weak random cross-coupling
    A_mod = np.zeros((n_nodes, n_nodes))
    A_mod[:half, :half] = A1
    A_mod[half:, half:] = A2
    A_mod[:half, half:] = rng.standard_normal((half, half)) * cross_coupling
    A_mod[half:, :half] = rng.standard_normal((half, half)) * cross_coupling
    A_mod = _spectral_rescale(A_mod, spectral_radius)

    # Segment 2 — hub (node 0 drives all others)
    A_hub = A_mod.copy()
    # Add strong outgoing connections from node 0 to nodes 1…n-1
    hub_out = rng.uniform(0.5, 1.0, size=n_nodes - 1) * hub_strength
    A_hub[1:, 0] += hub_out
    # Reduce cross-cluster coupling (hub takes over that role)
    A_hub[:half, half:] *= 0.15
    A_hub[half:, :half] *= 0.15
    A_hub = _spectral_rescale(A_hub, spectral_radius)

    return [A_mod, A_hub]


def _matrices_critical_slowing(
    rng: np.random.Generator,
    n_nodes: int,
    sr_low: float,
    sr_high: float,
    n_oscillation_modes: Optional[int],
) -> Tuple[List[np.ndarray], List[float]]:
    """Return ([A_low, A_high], [noise_scale_low, noise_scale_high]).

    Both matrices share the same random eigenvector basis; only the eigenvalue
    magnitudes differ.  Noise scales are chosen so stationary variance is
    approximately equal: σᵢ ∝ √(1 − srᵢ²).
    """
    n_osc  = min(
        n_nodes // 2 if n_oscillation_modes is None else n_oscillation_modes,
        n_nodes // 2,
    )
    n_real = n_nodes - 2 * n_osc

    # Draw angles and real-eigenvalue signs ONCE — shared by both regimes
    thetas     = rng.uniform(np.pi / 6, 5 * np.pi / 6, size=n_osc)
    real_signs = rng.choice([-1, 1], size=n_real)

    # Shared similarity transform
    R     = rng.standard_normal((n_nodes, n_nodes))
    R_inv = np.linalg.inv(R)

    def _build(sr: float) -> np.ndarray:
        blocks = []
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            blocks.append(sr * np.array([[c, -s], [s, c]]))
        for sign in real_signs:
            blocks.append(np.array([[sign * sr * 0.9]]))
        D = _block_diag(blocks)
        return _spectral_rescale(R @ D @ R_inv, sr)

    matrices = [_build(sr_low), _build(sr_high)]

    # Noise scales to equalise stationary variance (scalar approximation):
    # stationary var ≈ σ² / (1 − sr²)  →  σ_matched = σ_base · √(1 − sr²)
    noise_scales = [np.sqrt(1.0 - sr_low ** 2), np.sqrt(1.0 - sr_high ** 2)]

    return matrices, noise_scales


def _matrices_sign_flip(
    rng: np.random.Generator,
    n_nodes: int,
    spectral_radius: float,
    n_oscillation_modes: Optional[int],
) -> List[np.ndarray]:
    """Return [A1, A2] where A2 has sign-flipped cross-cluster coupling."""
    if n_nodes < 4 or n_nodes % 2 != 0:
        raise ValueError("sign_flip requires even n_nodes >= 4")
    half  = n_nodes // 2
    n_osc = n_nodes // 2 if n_oscillation_modes is None else n_oscillation_modes
    A1 = _structured_matrix(rng, n_nodes, spectral_radius, n_osc)
    A2 = A1.copy()
    A2[:half, half:] = -A2[:half, half:]
    A2[half:, :half] = -A2[half:, :half]
    A2 = _spectral_rescale(A2, spectral_radius)
    return [A1, A2]


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

def generate_switching_var(
    scenario: str = "modular_to_hub",
    n_nodes: int = 6,
    segment_lengths: List[int] = (300, 300),
    noise_std: float = 0.1,
    # shared / generic params
    spectral_radius: float = 0.9,
    n_oscillation_modes: Optional[int] = None,
    random_seed: int = 42,
    # modular_to_hub params
    hub_strength: float = 0.4,
    cross_coupling: float = 0.05,
    # critical_slowing params
    sr_low: float = 0.70,
    sr_high: float = 0.97,
    # random scenario params (backward-compat)
    n_switches: int = 1,
    switch_strength: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Generate a multivariate time series from a switching VAR(1) process.

    Parameters
    ----------
    scenario : str
        "modular_to_hub" | "critical_slowing" | "sign_flip" | "random"
    n_nodes : int
        Dimensionality.  "modular_to_hub" and "sign_flip" require even n_nodes >= 4.
    segment_lengths : list of int
        Length of each segment.  Structural scenarios require exactly 2 segments.
    noise_std : float
        Additive Gaussian noise standard deviation.  For "critical_slowing" this
        is the noise for the LOW-sr segment; the HIGH-sr segment's noise is
        automatically rescaled to match stationary variance.
    spectral_radius : float
        Target spectral radius (used by "modular_to_hub", "sign_flip", "random").
    sr_low, sr_high : float
        Spectral radii for the two "critical_slowing" segments.
    hub_strength : float
        Outgoing connection weight from the hub node (segment 2).
    cross_coupling : float
        Cross-cluster coupling strength in segment 1 (modular baseline).
    n_oscillation_modes : int or None
        Complex-conjugate eigenvalue pairs.  None → n_nodes // 2.
    random_seed : int

    Returns
    -------
    time_series : ndarray (T, n_nodes)
    switch_times : ndarray of int (n_switches,)
    var_matrices : list of ndarray
    segment_ids : ndarray of int (T,)
    """
    segment_lengths = list(segment_lengths)
    n_segments      = len(segment_lengths)
    rng             = np.random.default_rng(random_seed)

    # ----------------------------------------------------------------
    # Build connectivity matrices
    # ----------------------------------------------------------------
    noise_stds: List[float] = [noise_std] * n_segments   # default: uniform

    if scenario == "modular_to_hub":
        if n_segments != 2:
            raise ValueError("modular_to_hub requires exactly 2 segments")
        matrices = _matrices_modular_to_hub(
            rng, n_nodes, spectral_radius, hub_strength, cross_coupling
        )

    elif scenario == "critical_slowing":
        if n_segments != 2:
            raise ValueError("critical_slowing requires exactly 2 segments")
        matrices, noise_scales = _matrices_critical_slowing(
            rng, n_nodes, sr_low, sr_high, n_oscillation_modes
        )
        # noise_std is used for the low-sr segment; high-sr scaled to match var
        noise_stds = [noise_std * s / noise_scales[0] for s in noise_scales]

    elif scenario == "sign_flip":
        if n_segments != 2:
            raise ValueError("sign_flip requires exactly 2 segments")
        matrices = _matrices_sign_flip(rng, n_nodes, spectral_radius, n_oscillation_modes)

    elif scenario == "random":
        n_switches_ = n_segments - 1
        n_osc = n_nodes // 2 if n_oscillation_modes is None else n_oscillation_modes

        def _new_mat():
            return _structured_matrix(rng, n_nodes, spectral_radius, n_osc)

        matrices = [_new_mat()]
        for _ in range(n_switches_):
            blended = (1.0 - switch_strength) * matrices[-1] + switch_strength * _new_mat()
            matrices.append(_spectral_rescale(blended, spectral_radius))

    else:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            "Choose: 'modular_to_hub', 'critical_slowing', 'sign_flip', 'random'."
        )

    # ----------------------------------------------------------------
    # Segment metadata
    # ----------------------------------------------------------------
    switch_times = np.cumsum(segment_lengths[:-1], dtype=int)
    T            = sum(segment_lengths)
    segment_ids  = np.zeros(T, dtype=int)
    start        = 0
    for seg_idx, length in enumerate(segment_lengths):
        segment_ids[start : start + length] = seg_idx
        start += length

    # ----------------------------------------------------------------
    # Simulate time series
    # ----------------------------------------------------------------
    X    = np.zeros((T, n_nodes))
    X[0] = rng.standard_normal(n_nodes) * 0.1

    for t in range(1, T):
        seg = segment_ids[t]
        X[t] = (matrices[seg] @ X[t - 1]
                + rng.standard_normal(n_nodes) * noise_stds[seg])

    return X, switch_times, matrices, segment_ids

if __name__ == "__main__":

    for nstd in [0.005, 0.01, 0.05, 0.1, 0.3]:
        X, switch_times, matrices, segment_ids=generate_switching_var(
        scenario="critical_slowing",
        n_nodes = 6,
        segment_lengths = (50000, 50000),
        noise_std = nstd,
        # shared / generic params
        spectral_radius = 0.9,
        n_oscillation_modes= None,
        random_seed= 42,
        # modular_to_hub params
        hub_strength= 0.4,
        cross_coupling= 0.05,
        # critical_slowing params
        sr_low= 0.70,
        sr_high= 0.97,
        # random scenario params (backward-compat)
        n_switches= 1,
        switch_strength= 0.5
        )