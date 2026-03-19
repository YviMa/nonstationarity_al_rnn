"""Alignment evaluation between ground-truth switches and inferred regime changes."""

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm


def temporal_alignment_score(
    switch_times: np.ndarray,
    inferred_change_times: np.ndarray,
) -> Dict[str, float]:
    """For each ground-truth switch, find the nearest inferred regime change.

    Parameters
    ----------
    switch_times : array-like of int
        Ground-truth switch time indices (start of each new segment).
    inferred_change_times : array-like of int
        Time indices where the bitcode changes.

    Returns
    -------
    dict with keys:
        'mean_distance' – mean absolute distance (time steps) over all GT switches.
        'max_distance'  – max  absolute distance.
        If inferred_change_times is empty, both values are inf.
    """
    switch_times = np.asarray(switch_times, dtype=float)
    inferred = np.asarray(inferred_change_times, dtype=float)

    if len(inferred) == 0 or len(switch_times) == 0:
        return {"mean_distance": float("inf"), "max_distance": float("inf")}

    distances = np.array(
        [np.min(np.abs(inferred - t)) for t in switch_times]
    )
    return {
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }


def regime_count_ratio(
    bitcodes: np.ndarray,
    n_ground_truth_segments: int,
) -> float:
    """Ratio of inferred distinct regimes to ground-truth number of segments.

    A value close to 1.0 is ideal.

    Parameters
    ----------
    bitcodes : ndarray of int, shape (T,)
    n_ground_truth_segments : int
        Expected number of regimes (n_switches + 1).

    Returns
    -------
    ratio : float
    """
    from .regime_logger import count_unique_regimes

    n_inferred = count_unique_regimes(bitcodes)
    return n_inferred / max(n_ground_truth_segments, 1)


def _shade_background(ax, bitcodes: np.ndarray, code_to_color: dict) -> None:
    """Fill axis background with contiguous bitcode regions."""
    T = len(bitcodes)
    boundaries = np.concatenate(
        [[0], np.where(np.diff(bitcodes) != 0)[0] + 1, [T]]
    )
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        ax.axvspan(s, e, alpha=0.25, color=code_to_color[int(bitcodes[s])],
                   linewidth=0)


def plot_alignment(
    time_series: np.ndarray,
    generated_trajectory: np.ndarray,
    switch_times: np.ndarray,
    bitcodes: np.ndarray,
    n_pwl_units: int,
    pwl_states: np.ndarray = None,
    max_nodes: int = None,
    save_path: str = "alignment.png",
    show: bool = False,
) -> None:
    """Visualise observations, AL-RNN trajectory, and PWL unit states.

    Layout
    ------
    N rows for observations (observed + generated overlay), then P rows for
    the pre-activation trajectory of each PWL unit.  All panels share the
    x-axis and have the same bitcode-shaded background.  A horizontal line
    at y=0 on each PWL panel marks the sign boundary (regime switch point).

    Parameters
    ----------
    time_series : ndarray of shape (T, N)
    generated_trajectory : ndarray of shape (T, N)
    switch_times : array-like of int
    bitcodes : ndarray of int, shape (T,)
    n_pwl_units : int
        P — used for binary bitcode label formatting.
    pwl_states : ndarray of shape (T, P) or None
        Pre-activation values of the P PWL units at every step.  When
        provided, P extra panels are appended below the observation rows.
    max_nodes : int or None
        Maximum number of observation panels to plot.  When None all N nodes
        are shown.  Useful for high-dimensional data (e.g. EMA with 22 items).
    save_path : str
    show : bool
    """
    T, N = time_series.shape
    N = N if max_nodes is None else min(N, max_nodes)
    time_series         = time_series[:, :N]
    generated_trajectory = generated_trajectory[:, :N]
    P = 0 if pwl_states is None else pwl_states.shape[1]
    n_rows = N + P

    unique_codes = np.unique(bitcodes)
    n_codes = len(unique_codes)
    cmap = cm.get_cmap("tab10", max(n_codes, 2))
    code_to_color = {int(c): cmap(i) for i, c in enumerate(unique_codes)}

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, 1.8 * n_rows),
        sharex=True,
        gridspec_kw={"hspace": 0.05},
    )
    if n_rows == 1:
        axes = [axes]

    t_axis = np.arange(T)
    switch_times = np.asarray(switch_times, dtype=int)

    # -- observation panels ------------------------------------------------
    for i in range(N):
        ax = axes[i]
        _shade_background(ax, bitcodes, code_to_color)
        ax.plot(t_axis, time_series[:, i], lw=0.8, color="steelblue",
                label="observed" if i == 0 else None)
        ax.plot(t_axis, generated_trajectory[:, i], lw=0.8, color="coral",
                alpha=0.8, label="AL-RNN" if i == 0 else None)
        for t_gt in switch_times:
            ax.axvline(t_gt, color="black", linestyle="--", lw=1.1, alpha=0.8)
        ax.set_ylabel(f"node {i}", fontsize=8)
        ax.tick_params(labelsize=7)

    # -- PWL unit panels ---------------------------------------------------
    for p in range(P):
        ax = axes[N + p]
        _shade_background(ax, bitcodes, code_to_color)
        ax.plot(t_axis, pwl_states[:, p], lw=0.9, color="mediumpurple")
        ax.axhline(0, color="black", lw=0.8, linestyle=":")
        for t_gt in switch_times:
            ax.axvline(t_gt, color="black", linestyle="--", lw=1.1, alpha=0.8)
        ax.set_ylabel(f"PWL {p}\n(pre-ReLU)", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time step", fontsize=9)

    # -- legend (first panel only) -----------------------------------------
    bit_width = max(n_pwl_units, 1)
    legend_handles = [
        mpatches.Patch(facecolor=code_to_color[int(c)], alpha=0.6,
                       label=f"bitcode {int(c):0{bit_width}b}")
        for c in unique_codes
    ]
    from matplotlib.lines import Line2D
    legend_handles += [
        Line2D([0], [0], color="steelblue",    lw=1.5, label="observed"),
        Line2D([0], [0], color="coral",        lw=1.5, label="AL-RNN"),
        Line2D([0], [0], color="mediumpurple", lw=1.5, label="PWL (pre-ReLU)"),
    ]
    if len(switch_times) > 0:
        legend_handles.append(
            Line2D([0], [0], color="black", lw=1.1, linestyle="--", label="GT switch")
        )
    axes[0].legend(handles=legend_handles, loc="upper right",
                   fontsize=8, framealpha=0.7)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Alignment figure saved to {save_path}")


def plot_bitcode_distributions(
    bitcodes: np.ndarray,
    segment_ids: np.ndarray,
    n_pwl_units: int,
    save_path: str = "bitcode_distributions.png",
    show: bool = False,
) -> None:
    """Bar charts of bitcode relative frequencies, one subplot per segment.

    The x-axis shows all bitcodes that appear anywhere in the sequence (same
    order across every subplot) so distributions are visually comparable.
    Bar colours match those used in ``plot_alignment``.

    Parameters
    ----------
    bitcodes : ndarray of int, shape (T,)
        Active PWL regime (packed integer) at every time step.
    segment_ids : ndarray of int, shape (T,)
        Ground-truth segment label for each time step.
    n_pwl_units : int
        P — used to format binary bitcode labels on the x-axis.
    save_path : str
    """
    unique_codes = np.unique(bitcodes)
    n_codes = len(unique_codes)
    cmap = cm.get_cmap("tab10", max(n_codes, 2))
    code_to_color = {int(c): cmap(i) for i, c in enumerate(unique_codes)}

    segments = np.unique(segment_ids)
    n_segments = len(segments)
    bit_width = max(n_pwl_units, 1)
    x_labels = [f"{int(c):0{bit_width}b}" for c in unique_codes]
    x_pos = np.arange(n_codes)

    fig, axes = plt.subplots(
        1, n_segments,
        figsize=(max(4, 1.5 * n_codes) * n_segments, 3.5),
        sharey=True,
    )
    if n_segments == 1:
        axes = [axes]

    for ax, seg in zip(axes, segments):
        mask = segment_ids == seg
        seg_codes = bitcodes[mask]
        counts = np.array([np.sum(seg_codes == c) for c in unique_codes], dtype=float)
        total = counts.sum()
        rel_freq = counts / total if total > 0 else counts

        bar_colors = [code_to_color[int(c)] for c in unique_codes]
        ax.bar(x_pos, rel_freq, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Segment {int(seg)}", fontsize=10)
        ax.set_xlabel("Bitcode", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Relative frequency", fontsize=9)
    fig.suptitle(
        f"Bitcode distributions per segment  "
        f"(P={n_pwl_units} ReLU unit{'s' if n_pwl_units != 1 else ''})",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Bitcode distribution figure saved to {save_path}")
