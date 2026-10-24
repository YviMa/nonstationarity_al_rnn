"""Experiment: AL-RNN regime detection on switching bifurcation ODEs.

Three low-dimensional systems are available via CONFIG["system"]:

    "hopf"            — Stuart-Landau normal form.
                        param = μ: negative → stable fixed point,
                                   positive → stable limit cycle (radius √μ).
                        Clean Hopf bifurcation; easiest for the model to detect.

    "van_der_pol"     — Van der Pol oscillator.
                        param = μ: must be > 0 for oscillations (μ ≤ 0 is a
                                   stable fixed point, same as Hopf with μ < 0).
                                   small μ → near-harmonic (amplitude ≈ 2),
                                   large μ → relaxation oscillation (spiky).
                        Recommended: [0.5, 3.0]

    "fitzhugh_nagumo" — FitzHugh-Nagumo neural model.
                        param = I_ext: below ~0.34 → quiescent,
                                       above ~0.34 → spiking limit cycle.

Set param_values to one value per segment; segment_lengths controls the
number of samples in each segment (at the chosen dt).

Edit CONFIG, then run:

    python experiments/run_bifurcation.py

from the repository root.
"""

import os
import sys
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.switching_ode import generate_switching_ode
from experiments.runner import run_experiment


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- ODE system --------------------------------------------------------
    "system": "fitzhugh_nagumo",      # "hopf" | "van_der_pol" | "fitzhugh_nagumo"

    # Bifurcation parameter per segment (one entry per segment):
    #
    #   hopf:            μ — negative → stable fixed point (spiral to origin)
    #                        positive → limit cycle of radius √μ
    #                    Recommended: [0.5, -0.5]  (limit cycle → decay)
    #
    #   van_der_pol:     μ — MUST be positive for oscillations (μ < 0 → fixed point!)
    #                        small μ → near-harmonic sine wave (amplitude ≈ 2)
    #                        large μ → relaxation oscillation (sharp spikes)
    #                    Recommended: [0.5, 3.0]
    #
    #   fitzhugh_nagumo: I_ext — below ~0.34 → quiescent, above → spiking
    #                    Recommended: [0.2, 0.7]
  #  "param_values": [0.5, -0.1],
   # "param_values": [0.1, 3.0],
    "param_values": [0.33, .7],

    "segment_lengths": [400, 400],
    "dt": 0.5,            # RK4 integration step size
    "warmup": 0,          # warmup steps before recording (lands on attractor)
    "noise_std": 0.0,
    "random_seed": 42,

    # System-specific kwargs (forwarded to ODE factory):
    "omega": 1.0,          # hopf only: angular frequency of the limit cycle

    # --- temporal external input -------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",         # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "all",          # 'all' | 'pwl_only' | 'monotone_gate'

    # --- model -------------------------------------------------------------
    "M": 20,
    "P": 2,
    "P_switch": None,   # slow-switching units; None = all P (backward-compatible)

    # --- training ----------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 50,
    "alpha": 1.0,
    "end_alpha": 1.0,
    "n_interleave": 2,
    "start_lr": 1e-3,
    "end_lr": 1e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation ----------------------------------------------------
    "switch_lambda": 0.0,
    "pwl_timescale": 0.95,
    "timescale_lambda": 0.0,
    "incoming_lambda": 0.0,
    "pwl_state_lambda": 0.0,   # L2 penalty on |h[-P:]|² — prevents post-switch divergence

    # --- output ------------------------------------------------------------
    "figure_path": "bifurcation_results/alignment.png",
    "plot_every": 100,

    # --- consistency -------------------------------------------------------
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG):

    mlflow.set_tracking_uri("sqlite:////gs/home/wernery/nonstationarity_al_rnn/mlflow_test.db")
    mlflow.set_experiment("Bifurcation")

    with mlflow.start_run() as run:

        mlflow.log_params(cfg)

        system = cfg["system"]

        # ------------------------------------------------------------------
        # 1. Generate data
        # ------------------------------------------------------------------
        system_kwargs = {}
        if system == "hopf" and "omega" in cfg:
            system_kwargs["omega"] = cfg["omega"]

        time_series, switch_times, segment_ids = generate_switching_ode(
            system=system,
            param_values=cfg["param_values"],
            segment_lengths=cfg["segment_lengths"],
            dt=cfg["dt"],
            warmup=cfg["warmup"],
            noise_std=cfg["noise_std"],
            random_seed=cfg["random_seed"],
            **system_kwargs,
        )
        T, N = time_series.shape
        print(f"Data: system={system!r}, T={T}, N={N}, "
            f"params={cfg['param_values']}, switches at {switch_times.tolist()}")

        # ------------------------------------------------------------------
        # 2. Hand off to shared runner
        # ------------------------------------------------------------------
        model_path = cfg["figure_path"].replace(".png", "_model.pt")

        run_experiment(
            cfg, time_series, switch_times, segment_ids,
            model_path=model_path,
            extra_checkpoint={"system": system, "param_values": cfg["param_values"]},
            summary_lines=[
                f"  System                   : {system}",
                f"  Params per segment       : {cfg['param_values']}",
            ],
        )


if __name__ == "__main__":
    main()
