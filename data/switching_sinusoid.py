"""Switching multi-frequency sinusoid data generator.

Each node is a superposition of n_components sinusoids.  At each switch
point the per-node frequencies shift toward a new set of random values,
controlled by switch_strength.  Phases are carried over continuously at
every boundary so the signal has no discontinuities.
"""

import numpy as np
from typing import List, Tuple


def generate_switching_sinusoid(
    n_nodes: int,
    n_switches: int,
    switch_strength: float,
    segment_lengths: List[int],
    n_components: int = 3,
    freq_range: Tuple[float, float] = (0.02, 0.15),
    noise_std: float = 0.01,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Generate a multivariate switching sinusoidal time series.

    Parameters
    ----------
    n_nodes : int
        Number of observed dimensions.
    n_switches : int
        Number of frequency switches (n_switches + 1 segments).
    switch_strength : float in [0, 1]
        How much the frequencies change at each switch.
        0.0 → frequencies unchanged; 1.0 → fully new random frequencies.
    segment_lengths : list of int
        Length of each segment; must have length n_switches + 1.
    n_components : int
        Number of sinusoidal components per node.
    freq_range : (float, float)
        (min, max) frequencies in cycles per sample.  E.g. (0.02, 0.15)
        corresponds to periods of ~7 to 50 samples.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    random_seed : int

    Returns
    -------
    time_series : ndarray of shape (T, n_nodes)
    switch_times : ndarray of int, shape (n_switches,)
        Time index at which each new segment begins.
    freq_params : list of ndarray, each shape (n_nodes, n_components)
        Frequencies (cycles/sample) used in each segment.
    segment_ids : ndarray of int, shape (T,)
        Segment label per time step.
    """
    assert len(segment_lengths) == n_switches + 1
    assert freq_range[0] < freq_range[1]
    rng = np.random.default_rng(random_seed)

    # ------------------------------------------------------------------
    # Build per-segment frequency arrays via interpolation
    # ------------------------------------------------------------------
    def _random_freqs():
        return rng.uniform(freq_range[0], freq_range[1], size=(n_nodes, n_components))

    freq_params = [_random_freqs()]
    for _ in range(n_switches):
        new_f = _random_freqs()
        blended = (1.0 - switch_strength) * freq_params[-1] + switch_strength * new_f
        freq_params.append(np.clip(blended, freq_range[0], freq_range[1]))

    # ------------------------------------------------------------------
    # Fixed per-(node, component) amplitudes, normalised to unit RMS per node
    # ------------------------------------------------------------------
    amplitudes = rng.uniform(0.5, 1.0, size=(n_nodes, n_components))
    # scale so each node has approximately unit variance:
    # Var[node] ≈ sum_k A_k^2 / 2  →  want = 1  →  divide by sqrt(sum A_k^2 / 2)
    for i in range(n_nodes):
        amplitudes[i] /= np.sqrt(np.sum(amplitudes[i] ** 2) / 2)

    # ------------------------------------------------------------------
    # Segment metadata
    # ------------------------------------------------------------------
    switch_times = np.cumsum(segment_lengths[:-1], dtype=int)
    T = sum(segment_lengths)
    segment_ids = np.zeros(T, dtype=int)
    start = 0
    for seg_idx, length in enumerate(segment_lengths):
        segment_ids[start : start + length] = seg_idx
        start += length

    # ------------------------------------------------------------------
    # Simulate with phase continuity at each boundary
    # ------------------------------------------------------------------
    X = np.zeros((T, n_nodes))
    # Initial phases: random per (node, component)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(n_nodes, n_components))

    t_global = 0
    for seg_idx, length in enumerate(segment_lengths):
        f = freq_params[seg_idx]               # (n_nodes, n_components)
        t_local = np.arange(length)            # (length,)
        # phase_t[i, k, t] = 2π f[i,k] * t + phases[i,k]
        phase_t = (
            2.0 * np.pi * f[np.newaxis, :, :] * t_local[:, np.newaxis, np.newaxis]
            + phases[np.newaxis, :, :]
        )  # (length, n_nodes, n_components)
        X[t_global : t_global + length] = (
            (amplitudes[np.newaxis, :, :] * np.sin(phase_t)).sum(axis=2)
        )  # (length, n_nodes)

        # Advance phases for continuity: accumulated phase at end of segment
        phases = 2.0 * np.pi * f * length + phases

        t_global += length

    X += rng.standard_normal(X.shape) * noise_std
    return X, switch_times, freq_params, segment_ids
