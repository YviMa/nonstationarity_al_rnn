"""Experiment: AL-RNN regime detection on switching Lorenz-63.

The Lorenz (1963) system:

    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz

ρ switches between segments; σ and β are fixed.  Interesting pairs:

    [28, 35]   — both chaotic but different attractor geometry (default)
    [14, 28]   — stable C± fixed points  →  classic strange attractor
    [28, 99.96]— chaos → period-3 window (intermittency near the window)

Because the Lorenz attractor is chaotic, the marginal statistics of each
variable differ substantially between ρ regimes (unlike the VAR scenarios),
so even univariate tests will detect the shift.  The challenge for AL-RNN
is to identify the *timing* of the switch and maintain consistent bitcodes
within each regime despite the sensitive dependence on initial conditions.

Edit CONFIG, then run:

    python experiments/run_lorenz.py

from the repository root.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.switching_ode import generate_switching_ode
from experiments.runner import run_experiment


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- Lorenz parameter per segment (ρ / rho) ----------------------------
    # [28, 35]    both chaotic, different attractor shape
    # [14, 28]    fixed-point (C±) → strange attractor
    # [28, 99.96] chaos → period-3 window
    "param_values": [28.0, 99.96],

    # --- Fixed Lorenz parameters (forwarded as system_kwargs) ---------------
    "sigma": 10.0,
    "beta":  8.0 / 3.0,

    # --- Integration ----------------------------------------------------------
    "segment_lengths": [1000, 1000],  # samples per segment
    "dt": 0.02,                        # RK4 step size (0.01 = standard)
    "warmup": 2000,                    # steps to land on attractor before recording
    "noise_std": 0.0,
    "random_seed": 42,

    # --- Temporal external input ----------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",         # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "monotone_gate",          # 'all' | 'pwl_only' | 'monotone_gate'

    # --- Model ----------------------------------------------------------------
    "M": 30,
    "P": 5,
    # P_switch: slow-switching units that receive temporal input + regularization.
    # None = all P units are slow-switching (backward-compatible).
    # Set e.g. P_switch=1 to designate only 1 slow unit and let the rest
    # switch freely for local nonlinear approximation of the attractor.
    "P_switch": 1,

    # --- Training -------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 78,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # Lorenz is chaotic — use a larger n_interleave to prevent error blow-up
    # during free runs.  4–8 is typical for the classic attractor.
    "n_interleave": 10,
    "start_lr": 1e-3,
    "end_lr": 1e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- Regularisation -------------------------------------------------------
    "switch_lambda":    0.001,
    "pwl_timescale":    0.5,
    "timescale_lambda": 1.,
    "incoming_lambda":  1.,
    "pwl_state_lambda": 0.0,   # L2 penalty on |h[-P:]|² — prevents post-switch divergence

    # --- Output ---------------------------------------------------------------
    "figure_path": "lorenz_results/lorenz_alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,

    # --- Consistency ----------------------------------------------------------
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG):
    rho_vals = cfg["param_values"]
    sigma    = cfg.get("sigma", 10.0)
    beta     = cfg.get("beta",  8.0 / 3.0)

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print(f"Generating Lorenz-63  ρ={rho_vals}, σ={sigma}, β={beta:.4f}")
    time_series, switch_times, segment_ids = generate_switching_ode(
        system="lorenz63",
        param_values=rho_vals,
        segment_lengths=cfg["segment_lengths"],
        dt=cfg["dt"],
        warmup=cfg["warmup"],
        noise_std=cfg["noise_std"],
        random_seed=cfg["random_seed"],
        sigma=sigma,
        beta=beta,
    )
    T, N = time_series.shape
    print(f"  T={T}, N={N}, switch_times={switch_times.tolist()}")
    _describe_lorenz(rho_vals, time_series, segment_ids)

    # ------------------------------------------------------------------
    # 2. Hand off to shared runner
    # ------------------------------------------------------------------
    model_path = cfg["figure_path"].replace("_alignment.png", "_model.pt").replace(".png", "_model.pt")

    run_experiment(
        cfg, time_series, switch_times, segment_ids,
        model_path=model_path,
        extra_checkpoint={"rho_vals": rho_vals},
        summary_lines=[
            f"  System                   : Lorenz-63",
            f"  ρ per segment            : {rho_vals}",
        ],
    )


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def _describe_lorenz(rho_vals, time_series, segment_ids):
    """Print per-segment attractor statistics."""
    for i, rho in enumerate(rho_vals):
        seg   = time_series[segment_ids == i]
        means = seg.mean(axis=0)
        stds  = seg.std(axis=0)
        print(f"  Segment {i}  ρ={rho:6.2f}  "
              f"mean=({means[0]:+.2f}, {means[1]:+.2f}, {means[2]:+.2f})  "
              f"std=({stds[0]:.2f}, {stds[1]:.2f}, {stds[2]:.2f})")


if __name__ == "__main__":
    main()
