"""Bitcode consistency scoring across multiple independent AL-RNN runs.

The core problem: bitcodes are permutation-invariant across runs — regime 0
in run A may be labelled regime 1 in run B.  We therefore compare the
*partitions* of time points, not the labels themselves.

Primary metric
--------------
Adjusted Rand Index (ARI) — counts concordant and discordant pairs of time
points, corrected for chance.  Permutation-invariant, O(T) per pair, range
[-1, 1] (1 = perfect agreement, 0 = chance-level).

Secondary metric (optional, O(T²))
------------------------------------
Co-assignment matrix — C[i, j] = fraction of runs in which time points i and
j share the same bitcode.  Useful for short series where a full T×T view is
feasible; too large for long EMA recordings.
"""

from itertools import combinations
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# ARI-based consistency (recommended — O(K² * T))
# ---------------------------------------------------------------------------

def _rand_index_components(labels_a: np.ndarray, labels_b: np.ndarray):
    """Return (n_agree_same, n_agree_diff, n_total_pairs) for two label vectors."""
    T = len(labels_a)
    # For each pair (i, j) with i < j:
    #   agree_same: labels_a[i]==labels_a[j]  AND  labels_b[i]==labels_b[j]
    #   agree_diff: labels_a[i]!=labels_a[j]  AND  labels_b[i]!=labels_b[j]
    # Computed via contingency table — O(T + n_classes²).
    classes_a = np.unique(labels_a)
    classes_b = np.unique(labels_b)
    contingency = np.zeros((len(classes_a), len(classes_b)), dtype=np.int64)
    a_idx = {c: i for i, c in enumerate(classes_a)}
    b_idx = {c: i for i, c in enumerate(classes_b)}
    for ca, cb in zip(labels_a, labels_b):
        contingency[a_idx[ca], b_idx[cb]] += 1
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    n_pairs = T * (T - 1) // 2
    # n_same_a = Σ C(row_i, 2), n_same_b = Σ C(col_j, 2)
    n_same_a = int(np.sum(row_sums * (row_sums - 1)) // 2)
    n_same_b = int(np.sum(col_sums * (col_sums - 1)) // 2)
    # n_agree_same = Σ C(n_ij, 2)
    n_agree_same = int(np.sum(contingency * (contingency - 1)) // 2)
    return n_agree_same, n_same_a, n_same_b, n_pairs


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted Rand Index between two label vectors.

    Permutation-invariant; corrected for chance agreement.

    Parameters
    ----------
    labels_a, labels_b : ndarray of int, shape (T,)

    Returns
    -------
    ari : float in [-1, 1]
        1  = perfect agreement
        0  = agreement at chance level
       <0  = worse than chance (rare)
    """
    n_ab, n_a, n_b, n_pairs = _rand_index_components(labels_a, labels_b)
    # ARI = (n_ab - expected) / (0.5*(n_a + n_b) - expected)
    # expected = n_a * n_b / n_pairs
    if n_pairs == 0:
        return 1.0
    expected = n_a * n_b / n_pairs
    denominator = 0.5 * (n_a + n_b) - expected
    if denominator == 0:
        return 1.0 if n_ab == expected else 0.0
    return (n_ab - expected) / denominator


def bitcode_consistency(
    bitcodes_runs: List[np.ndarray],
) -> dict:
    """Compute pairwise ARI consistency across multiple independent runs.

    Parameters
    ----------
    bitcodes_runs : list of ndarray of int, shape (T,)
        Bitcode sequence from each run.  All must have the same length T.

    Returns
    -------
    dict with keys:
        'mean_ari'     : float — mean pairwise ARI
        'std_ari'      : float — std of pairwise ARIs
        'pairwise_ari' : list of float — one entry per (run_i, run_j) pair
        'n_runs'       : int
    """
    if len(bitcodes_runs) < 2:
        raise ValueError("Need at least 2 runs to compute consistency.")

    pairwise = [
        adjusted_rand_index(b1, b2)
        for b1, b2 in combinations(bitcodes_runs, 2)
    ]
    return {
        "mean_ari": float(np.mean(pairwise)),
        "std_ari":  float(np.std(pairwise)),
        "pairwise_ari": pairwise,
        "n_runs": len(bitcodes_runs),
    }


# ---------------------------------------------------------------------------
# Co-assignment matrix (optional, O(T²) — use only for short series)
# ---------------------------------------------------------------------------

def coassignment_matrix(
    bitcodes_runs: List[np.ndarray],
    max_T: int = 2000,
) -> Optional[np.ndarray]:
    """Fraction of runs in which each pair of time points shares a bitcode.

    Returns a symmetric T×T matrix where entry (i, j) is the fraction of
    runs where t_i and t_j received the same bitcode.  The diagonal is
    always 1.  Permutation-invariant by construction.

    Parameters
    ----------
    bitcodes_runs : list of ndarray of int, shape (T,)
    max_T : int
        If T > max_T, returns None with a warning (memory guard).

    Returns
    -------
    C : ndarray of float, shape (T, T), or None if T > max_T
    """
    T = len(bitcodes_runs[0])
    if T > max_T:
        import warnings
        warnings.warn(
            f"Co-assignment matrix skipped: T={T} > max_T={max_T}. "
            f"Use ARI-based consistency instead.",
            RuntimeWarning, stacklevel=2,
        )
        return None

    C = np.zeros((T, T), dtype=np.float32)
    K = len(bitcodes_runs)
    for b in bitcodes_runs:
        # Outer equality: same[i,j] = (b[i] == b[j])
        C += (b[:, None] == b[None, :]).astype(np.float32)
    C /= K
    return C
