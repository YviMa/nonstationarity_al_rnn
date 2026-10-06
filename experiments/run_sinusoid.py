"""Experiment: AL-RNN regime detection on switching multi-frequency sinusoids.

Each node is a superposition of sinusoidal components; at each regime switch
the per-node frequencies shift by an amount controlled by switch_strength.

Edit CONFIG, then run:

    python experiments/run_sinusoid.py

from the repository root.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.switching_sinusoid import generate_switching_sinusoid
from experiments.runner import run_experiment


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- data ----------------------------------------------------------------
    "n_nodes": 5,
    "n_switches": 2,
    # How much frequencies change at each switch: 0 = identical, 1 = fully new.
    "switch_strength": 0.4,
    "segment_lengths": [300, 200, 300],
    "n_components": 3,           # sinusoidal components per node
    "freq_range": [0.03, 0.3],   # cycles/sample #0.1,0.2
    "noise_std": 0.05,
    "random_seed": 42,

    # --- temporal external input --------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",         # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "monotone_gate",          # 'all' | 'pwl_only' | 'monotone_gate'

    # --- model ---------------------------------------------------------------
    "M": 20,
    "P": 2,
    "P_switch": None,   # slow-switching units; None = all P (backward-compatible)

    # --- training ------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 100,
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
    "pwl_timescale": 0.5,#0.95
    "timescale_lambda": 0.001,
    "incoming_lambda": 0.01,
    "pwl_state_lambda": 0.01,   # L2 penalty on |h[-P:]|² — prevents post-switch divergence

    # --- output --------------------------------------------------------------
    "figure_path": "sinusoid_results/alignment.png",
    "plot_every": 100,

    # --- consistency ---------------------------------------------------------
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG) -> None:
    # ------------------------------------------------------------------
    # 1. Generate dataset
    # ------------------------------------------------------------------
    print("Generating switching sinusoid data …")
    time_series, switch_times, _, segment_ids = generate_switching_sinusoid(
        n_nodes=cfg["n_nodes"],
        n_switches=cfg["n_switches"],
        switch_strength=cfg["switch_strength"],
        segment_lengths=cfg["segment_lengths"],
        n_components=cfg["n_components"],
        freq_range=tuple(cfg["freq_range"]),
        noise_std=cfg["noise_std"],
        random_seed=cfg["random_seed"],
    )
    T, N = time_series.shape
    n_segments = cfg["n_switches"] + 1
    print(f"  T={T}, N={N}, segments={n_segments}, switch_times={switch_times.tolist()}")

    # ------------------------------------------------------------------
    # 2. Build model path (encodes key hypers to avoid overwriting)
    # ------------------------------------------------------------------
    fr = cfg["freq_range"]
    ss = cfg["switch_strength"]
    model_path = cfg["figure_path"].replace(".png", f"_ss{ss}_fr{fr[0]}-{fr[1]}_model.pt")

    run_experiment(
        cfg, time_series, switch_times, segment_ids,
        model_path=model_path,
        extra_checkpoint={"switch_strength": ss, "freq_range": fr},
        summary_lines=[f"  Segments                 : {n_segments}"],
    )


if __name__ == "__main__":
    main()
