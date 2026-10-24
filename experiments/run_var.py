"""Experiment: AL-RNN regime detection on switching VAR(1) processes.

Three structured scenarios are available via CONFIG["scenario"]:

    "modular_to_hub"   — connectivity topology changes while marginal statistics
                         stay nearly constant.  Segment 1: two weakly-coupled
                         clusters.  Segment 2: node 0 becomes a hub that drives
                         all others.  Same spectral radius across both segments →
                         similar autocorrelation and variance; only the coupling
                         geometry changes.

    "critical_slowing" — same eigenvector structure but eigenvalues scaled from
                         sr_low to sr_high.  Mimics critical slowing down:
                         dynamics become more persistent without a change in
                         oscillation character.  Noise is auto-rescaled to match
                         stationary variance.

    "sign_flip"        — cross-cluster coupling reverses sign between segments
                         (inhibitory → excitatory).  Marginal autocorrelation is
                         unchanged; cross-covariance structure flips.

    "random"           — original random-matrix approach (any number of segments).

Edit CONFIG, then run:

    python experiments/run_var.py

from the repository root.
"""

import os
import sys
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from data.switching_var import generate_switching_var
from experiments.runner import run_experiment


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- scenario -----------------------------------------------------------
    # "modular_to_hub" | "critical_slowing" | "sign_flip" | "random"
    "scenario": "modular_to_hub",
  #  "scenario": "critical_slowing",
  #  "scenario": "sign_flip",

    # --- data ---------------------------------------------------------------
    # n_nodes must be even >= 4 for structured scenarios
    "n_nodes": 6,
    "segment_lengths": [200, 200],
    "noise_std": 0.05,
    "random_seed": 42,

    # --- scenario-specific params -------------------------------------------
    # shared by modular_to_hub / sign_flip / random:
    "spectral_radius": 0.90,

    # modular_to_hub: strength of hub outgoing connections (segment 2)
    "hub_strength": 0.2,
    # modular_to_hub: cross-cluster coupling magnitude in segment 1
    "cross_coupling": 0.05,

    # critical_slowing: spectral radii for the two segments
    "sr_low":  0.8,
    "sr_high": 0.95,

    # random: interpolation strength between consecutive matrices (0=no change)
    "switch_strength": 0.3,

    # all scenarios: number of complex-conjugate eigenvalue pairs (None = n//2)
    "n_oscillation_modes": None,

    # --- temporal external input --------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",         # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "monotone_gate",          # 'all' | 'pwl_only' | 'monotone_gate'

    # --- model ---------------------------------------------------------------
    "M": 30,
    "P": 1,
    "P_switch": None,   # slow-switching units; None = all P (backward-compatible)

    # --- training ------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 75,
    "alpha": 1.0,
    "end_alpha": 1.0,
    "n_interleave": 4,
    "start_lr": 1e-3,
    "end_lr": 1e-3,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation ------------------------------------------------------
    "switch_lambda": 0.0,
    "pwl_timescale": 0.5,
    "timescale_lambda": 0.001,
    "incoming_lambda": 0.01,
    "pwl_state_lambda": 0.01,   # L2 penalty on |h[-P:]|² — prevents post-switch divergence

    # --- output --------------------------------------------------------------
    # {scenario} is filled in automatically.
    "figure_path": "var_results/{scenario}_alignment.png",
    "plot_every": 100,

    # --- consistency ---------------------------------------------------------
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG) -> None:

    mlflow.set_tracking_uri("sqlite:////gs/home/wernery/nonstationarity_al_rnn/mlflow_test.db")
    mlflow.set_experiment("VAR")

    with mlflow.start_run() as run:

        mlflow.log_params(cfg)

        scenario = cfg["scenario"]

        # ------------------------------------------------------------------
        # 1. Generate dataset
        # ------------------------------------------------------------------
        print(f"Generating switching VAR(1) data  scenario={scenario!r} …")
        time_series, switch_times, matrices, segment_ids = generate_switching_var(
            scenario=scenario,
            n_nodes=cfg["n_nodes"],
            segment_lengths=cfg["segment_lengths"],
            noise_std=cfg["noise_std"],
            spectral_radius=cfg.get("spectral_radius", 0.9),
            sr_low=cfg.get("sr_low", 0.70),
            sr_high=cfg.get("sr_high", 0.97),
            hub_strength=cfg.get("hub_strength", 0.4),
            cross_coupling=cfg.get("cross_coupling", 0.05),
            n_oscillation_modes=cfg.get("n_oscillation_modes"),
            random_seed=cfg["random_seed"],
            switch_strength=cfg.get("switch_strength", 0.5),
        )
        T, N = time_series.shape
        print(f"  T={T}, N={N}, switch_times={switch_times.tolist()}")
        _describe_scenario(scenario, matrices, cfg)

        # ------------------------------------------------------------------
        # 2. Resolve paths and hand off to shared runner
        # ------------------------------------------------------------------
        fig_path   = cfg["figure_path"].format(scenario=scenario)
        cfg        = {**cfg, "figure_path": fig_path}
        model_path = fig_path.replace("_alignment.png", "_model.pt").replace(".png", "_model.pt")

        run_experiment(
            cfg, time_series, switch_times, segment_ids,
            model_path=model_path,
            extra_checkpoint={"scenario": scenario},
            summary_lines=[f"  Scenario                 : {scenario}"],
        )


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def _describe_scenario(scenario: str, matrices: list, cfg: dict) -> None:
    """Print a brief description of what changed between regimes."""
    A0, A1 = matrices[0], matrices[1]
    sr0 = np.max(np.abs(np.linalg.eigvals(A0)))
    sr1 = np.max(np.abs(np.linalg.eigvals(A1)))
    half = cfg["n_nodes"] // 2

    print(f"  Spectral radii           : seg0={sr0:.3f}, seg1={sr1:.3f}")

    if scenario == "modular_to_hub":
        cross0 = np.abs(A0[:half, half:]).mean()
        cross1 = np.abs(A1[:half, half:]).mean()
        hub0   = np.abs(A0[1:, 0]).mean()
        hub1   = np.abs(A1[1:, 0]).mean()
        print(f"  Cross-cluster coupling   : seg0={cross0:.3f}, seg1={cross1:.3f}")
        print(f"  Hub outgoing (node 0)    : seg0={hub0:.3f}, seg1={hub1:.3f}")

    elif scenario == "sign_flip":
        cross0 = A0[:half, half:].mean()
        cross1 = A1[:half, half:].mean()
        print(f"  Cross-cluster mean       : seg0={cross0:+.3f}, seg1={cross1:+.3f}  (sign flip)")

    elif scenario == "critical_slowing":
        print(f"  (sr_low={cfg['sr_low']}, sr_high={cfg['sr_high']}  — same eigenvectors)")


if __name__ == "__main__":
    main()
