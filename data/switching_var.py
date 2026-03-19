"""Switching VAR(1) data generator for regime-detection experiments."""

import numpy as np
from typing import List, Tuple


def _block_diag(blocks):
    """Stack a list of 2-D arrays into a block-diagonal matrix."""
    n = sum(b.shape[0] for b in blocks)
    D = np.zeros((n, n))
    start = 0
    for b in blocks:
        s = b.shape[0]
        D[start : start + s, start : start + s] = b
        start += s
    return D


def _structured_matrix(rng, n, spectral_radius, n_oscillation_modes):
    """Build a VAR(1) matrix with controlled spectral structure.

    Eigenspectrum
    -------------
    n_oscillation_modes complex-conjugate pairs give periodic dynamics.
    Remaining eigenvalues are real, drawn uniformly from
    [-spectral_radius, spectral_radius].
    All eigenvalues are then rescaled so the spectral radius equals the
    requested value exactly.

    A random similarity transform mixes the modes so the matrix is not
    block-diagonal in the standard basis.
    """
    n_osc = min(n_oscillation_modes, n // 2)
    n_real = n - 2 * n_osc

    blocks = []

    # Oscillatory blocks: 2×2 rotation-scaling matrices
    # Phases spread across (π/8, 7π/8) to avoid near-DC and near-Nyquist modes
    if n_osc > 0:
        thetas = rng.uniform(np.pi / 8, np.pi * 7 / 8, size=n_osc)
        for theta in thetas:
            c, s = np.cos(theta), np.sin(theta)
            blocks.append(spectral_radius * np.array([[c, -s], [s, c]]))

    # Real blocks: scalar eigenvalues with random sign
    for _ in range(n_real):
        ev = rng.uniform(-1.0, 1.0) * spectral_radius
        blocks.append(np.array([[ev]]))

    D = _block_diag(blocks)

    # Random similarity transform: A = R D R^{-1}
    R = rng.standard_normal((n, n))
    A = R @ D @ np.linalg.inv(R)

    # Renormalise to exactly the target spectral radius (numerical safety)
    sr = np.max(np.abs(np.linalg.eigvals(A)))
    if sr > 0:
        A = A * (spectral_radius / sr)

    return A


def generate_switching_var(
    n_nodes: int,
    n_switches: int,
    switch_strength: float,
    segment_lengths: List[int],
    noise_std: float = 0.1,
    random_seed: int = 42,
    spectral_radius: float = 0.9,
    n_oscillation_modes: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Generate a multivariate time series from a switching VAR(1) process.

    Parameters
    ----------
    n_nodes : int
        Dimensionality of the time series.
    n_switches : int
        Number of regime switches (produces n_switches + 1 segments).
    switch_strength : float in [0, 1]
        Controls similarity between consecutive VAR matrices.
        0.0 → new matrix identical to previous; 1.0 → fully independent.
    segment_lengths : list of int
        Length of each segment; must have length n_switches + 1.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    random_seed : int
    spectral_radius : float in (0, 1)
        Target spectral radius for every generated matrix.  Values close to 1
        give slow-decaying / oscillatory dynamics; small values give fast
        decay toward a fixed point.  Default 0.9.
    n_oscillation_modes : int or None
        Number of complex-conjugate eigenvalue pairs, each producing a
        periodic mode.  None (default) uses n_nodes // 2 — maximally
        oscillatory.  0 gives only real eigenvalues (monotone dynamics).

    Returns
    -------
    time_series : ndarray of shape (T, n_nodes)
        Simulated multivariate time series.
    switch_times : ndarray of int
        Time indices where each new segment begins (length n_switches).
    var_matrices : list of ndarray
        VAR(1) connectivity matrices, one per segment.
    segment_ids : ndarray of int, shape (T,)
        Integer segment label for every time step.
    """
    assert len(segment_lengths) == n_switches + 1, (
        "segment_lengths must have length n_switches + 1"
    )
    assert 0.0 < spectral_radius < 1.0, "spectral_radius must be in (0, 1)"

    rng = np.random.default_rng(random_seed)

    n_osc = n_nodes // 2 if n_oscillation_modes is None else n_oscillation_modes

    def _new_matrix():
        return _structured_matrix(rng, n_nodes, spectral_radius, n_osc)

    # Build n_switches + 1 VAR matrices via interpolation
    matrices = [_new_matrix()]
    for _ in range(n_switches):
        new_mat = _new_matrix()
        blended = (1.0 - switch_strength) * matrices[-1] + switch_strength * new_mat
        # Rescale blended matrix to the same target spectral radius
        sr = np.max(np.abs(np.linalg.eigvals(blended)))
        if sr > 0:
            blended = blended * (spectral_radius / sr)
        matrices.append(blended)

    # Segment boundary indices
    switch_times = np.cumsum(segment_lengths[:-1], dtype=int)

    # Assign a segment id to every time step
    T = sum(segment_lengths)
    segment_ids = np.zeros(T, dtype=int)
    start = 0
    for seg_idx, length in enumerate(segment_lengths):
        segment_ids[start : start + length] = seg_idx
        start += length

    # Simulate the time series
    X = np.zeros((T, n_nodes))
    X[0] = rng.standard_normal(n_nodes) * 0.1
    for t in range(1, T):
        A = matrices[segment_ids[t]]
        X[t] = A @ X[t - 1] + rng.standard_normal(n_nodes) * noise_std

    return X, switch_times, matrices, segment_ids
