"""Experiment: AL-RNN regime detection on switching dynamical systems.

Two data types are supported via CONFIG["data_type"]:

    "sinusoid" – superposition of sinusoids whose frequencies shift at each
                 switch (simple, stable, fully controlled).
    "var"       – switching VAR(1) with structured eigenspectrum.

Edit CONFIG at the top of this file, then run:

    python experiments/run_switching_var.py

from the repository root.
"""

import os
import sys
import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from models.temporal_input import TemporalInput
from alrnn import train_alrnn
from alrnn.training import predict_sequence_using_gtf
from analysis.regime_logger import log_bitcodes, bitcodes_from_pwl_states, count_unique_regimes, regime_change_times
from analysis.alignment import temporal_alignment_score, plot_alignment, plot_bitcode_distributions


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- which generator to use -------------------------------------------
    "data_type": "sinusoid",   # "sinusoid" | "var"

    # --- shared data params ------------------------------------------------
    "n_nodes": 3,
    "n_switches": 1,
    "switch_strength": 0.5,    # 0 = no change, 1 = fully new
    "segment_lengths": [200,200],
    "noise_std": 0.00,
    "random_seed": 42,

    # --- sinusoid-specific (only used when data_type == "sinusoid") --------
    "n_components": 3,         # sinusoidal components per node
    "freq_range": [0.05, 0.2], # cycles/sample → periods 10–20 steps

    # --- VAR-specific (only used when data_type == "var") ------------------
    "spectral_radius": 0.9,
    "n_oscillation_modes": None,   # None = n_nodes // 2

    # --- temporal external input -------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",   # 'none' | 'linear' | 'fourier'

    # --- model -------------------------------------------------------------
    # ITF: M must be >= N + P; the last M-N units evolve freely (true memory).
    # GTF: only M=N is meaningful — obs_to_latent re-initialises all extra units
    #      from the current observation at every step, so M>N adds no recurrence.
    "M": 20,   # for ITF with N=2, P=2: M >= 4; extra units give free latent memory
    "P": 1,

    # --- training ----------------------------------------------------------
    # sequence_length should cover ≥ 2 full cycles of the slowest frequency.
    # With freq_min=0.05 the slowest period is 20 steps → use ≥ 40.
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 100,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # n_interleave: apply teacher forcing every this many steps.
    "n_interleave": 1,
    "start_lr": 1e-3,
    "end_lr": 1e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",    # ITF lets M-N units carry free latent memory
    "device": "cpu",
    "verbose": True,

    # --- regularisation ----------------------------------------------------
    # Differentiable switch-count penalty (scale-invariant event-based proxy).
    # 0.0 = disabled; try 1e-2 … 1.0.
    "switch_lambda": .0,
    # Slow-timescale init for PWL units: A[-P:] is set to this value at init.
    # None = use standard init; ~0.98 gives near-unit-root slow drift.
    "pwl_timescale": 0.95,
    # L2 penalty pulling A[-P:] back toward pwl_timescale during training.
    # 0.0 = disabled; try 1e-2 … 1.0.
    "timescale_lambda": 0,

    # --- output ------------------------------------------------------------
    "figure_path": "alignment.png",
    # Plot the alignment figure every this many epochs during training.
    # None = only plot at the end.
    "plot_every": 100,
}


def _generate_data(cfg: dict):
    """Dispatch to the configured data generator."""
    common = dict(
        n_nodes=cfg["n_nodes"],
        n_switches=cfg["n_switches"],
        switch_strength=cfg["switch_strength"],
        segment_lengths=cfg["segment_lengths"],
        noise_std=cfg["noise_std"],
        random_seed=cfg["random_seed"],
    )
    if cfg["data_type"] == "sinusoid":
        from data.switching_sinusoid import generate_switching_sinusoid
        return generate_switching_sinusoid(
            **common,
            n_components=cfg["n_components"],
            freq_range=tuple(cfg["freq_range"]),
        )
    elif cfg["data_type"] == "var":
        from data.switching_var import generate_switching_var
        return generate_switching_var(
            **common,
            spectral_radius=cfg["spectral_radius"],
            n_oscillation_modes=cfg["n_oscillation_modes"],
        )
    else:
        raise ValueError(f"Unknown data_type: {cfg['data_type']!r}")


def main(cfg: dict = CONFIG) -> None:

    mlflow.set_tracking_uri("sqlite:////gs/home/wernery/nonstationarity_al_rnn/mlflow_test.db")
    mlflow.set_experiment("SwitchingVAR")

    with mlflow.start_run() as run:

        mlflow.log_params(cfg)
        # ------------------------------------------------------------------
        # 1. Generate dataset
        # ------------------------------------------------------------------
        print(f"Generating switching {cfg['data_type']} data …")
        time_series, switch_times, _, segment_ids = _generate_data(cfg)
        T, N = time_series.shape
        n_segments = cfg["n_switches"] + 1
        print(f"  T={T}, N={N}, segments={n_segments}, switch_times={switch_times.tolist()}")

        # ------------------------------------------------------------------
        # 2. Dimension-wise standardisation (zero mean, unit variance per node)
        # ------------------------------------------------------------------
        mean = time_series.mean(axis=0)
        std  = time_series.std(axis=0)
        std[std == 0] = 1.0
        train_series = (time_series - mean) / std

        # ------------------------------------------------------------------
        # 3. Compute temporal external inputs (not part of observations)
        # ------------------------------------------------------------------
        temporal_module = TemporalInput(
            mode=cfg["temporal_input_mode"] if cfg["use_temporal_input"] else "none"
        )
        external_inputs = temporal_module.compute_features(T)  # (T, d_input) or None
        d_input = temporal_module.output_dim
        print(f"  temporal_input_mode={temporal_module.mode!r}, d_input={d_input}")

        # ------------------------------------------------------------------
        # 4b. Helper: run one forward pass and return trajectory + bitcodes
        # ------------------------------------------------------------------
        from alrnn.training import predict_sequence_using_itf

        x_full           = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
        u_full           = (torch.tensor(external_inputs, dtype=torch.float32).unsqueeze(0)
                            if external_inputs is not None else None)
        vis_alpha        = cfg.get("end_alpha") if cfg.get("end_alpha") is not None else cfg["alpha"]
        vis_n_interleave = cfg.get("n_interleave", 1)
        collect_pwl      = cfg["P"] > 0
        base_stem        = cfg["figure_path"].replace(".png", "")

        def _run_trajectory(mdl):
            with torch.no_grad():
                if cfg["forcing_mode"] == "itf":
                    res = predict_sequence_using_itf(
                        mdl, x_full, vis_alpha, u=u_full,
                        n_interleave=vis_n_interleave, return_pwl_states=collect_pwl)
                else:
                    res = predict_sequence_using_gtf(
                        mdl, x_full, vis_alpha, u=u_full,
                        n_interleave=vis_n_interleave, return_pwl_states=collect_pwl)
            if collect_pwl:
                gen_t, pwl_t = res
                pwl_traj = pwl_t[0].cpu().numpy()
            else:
                gen_t, pwl_traj = res, None
            gen_obs = gen_t[0].cpu().numpy() * std + mean
            codes   = (bitcodes_from_pwl_states(pwl_traj) if pwl_traj is not None
                    else log_bitcodes(mdl, train_series, forcing_mode=cfg["forcing_mode"]))
            return gen_obs, pwl_traj, codes

        # ------------------------------------------------------------------
        # 4c. Build mid-training alignment-plot callback
        # ------------------------------------------------------------------
        def _alignment_callback(mdl, epoch):
            is_final  = (epoch >= cfg["num_epochs"])
            tag       = "_final" if is_final else f"_epoch{epoch:04d}"
            save_path = f"{base_stem}{tag}.png"
            gen_obs, pwl_traj, codes = _run_trajectory(mdl)
            n_trans = len(regime_change_times(codes))
            plot_alignment(
                time_series=time_series,
                generated_trajectory=gen_obs,
                switch_times=switch_times,
                bitcodes=codes,
                n_pwl_units=cfg["P"],
                pwl_states=pwl_traj,
                save_path=save_path,
                show=False,   # never block training
            )
            print(f"    [epoch {epoch}] → {save_path}  (transitions: {n_trans})")

        plot_every     = cfg.get("plot_every")
        epoch_callback = _alignment_callback if plot_every is not None else None

        # ------------------------------------------------------------------
        # 4. Train AL-RNN with GTF / ITF
        # ------------------------------------------------------------------
        print(f"\nTraining AL-RNN  M={cfg['M']}, P={cfg['P']}, d_input={d_input}, "
            f"epochs={cfg['num_epochs']} …")
        model, losses = train_alrnn(
            train_data=train_series,
            M=cfg["M"],
            P=cfg["P"],
            num_epochs=cfg["num_epochs"],
            batch_size=cfg["batch_size"],
            sequence_length=cfg["sequence_length"],
            alpha=cfg["alpha"],
            end_alpha=cfg.get("end_alpha"),
            n_interleave=cfg.get("n_interleave", 1),
            start_lr=cfg["start_lr"],
            end_lr=cfg["end_lr"],
            batches_per_epoch=cfg["batches_per_epoch"],
            forcing_mode=cfg["forcing_mode"],
            device=cfg["device"],
            verbose=cfg["verbose"],
            external_inputs=external_inputs,
            switch_lambda=cfg.get("switch_lambda", 0.0),
            pwl_timescale=cfg.get("pwl_timescale"),
            timescale_lambda=cfg.get("timescale_lambda", 0.0),
            epoch_callback=epoch_callback,
            callback_every=plot_every if plot_every is not None else 100,
        )
        print(f"  Final training loss: {losses[-1]:.6f}")

        # ------------------------------------------------------------------
        # 5 + 6. Final trajectory, regime logging, summary
        # ------------------------------------------------------------------
        generated_trajectory, pwl_trajectory, bitcodes = _run_trajectory(model)
        n_unique      = count_unique_regimes(bitcodes)
        change_times  = regime_change_times(bitcodes)
        n_transitions = len(change_times)

        # ------------------------------------------------------------------
        # 7. Alignment evaluation + summary
        # ------------------------------------------------------------------
        alignment = temporal_alignment_score(switch_times, change_times)

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Ground-truth switches    : {n_segments - 1}")
        print(f"  Inferred transitions     : {n_transitions}  (ideal = {n_segments - 1})")
        print(f"  Unique regimes           : {n_unique}")
        print(f"  Alignment mean / max     : {alignment['mean_distance']:.1f} / {alignment['max_distance']:.1f} steps")
        print("=" * 50)

        # ------------------------------------------------------------------
        # 8. Final alignment figure + bitcode-distribution plot
        # ------------------------------------------------------------------
        # If callbacks were used, the final alignment plot was already saved with
        # tag "_final"; otherwise save now with the configured figure_path.
        if epoch_callback is None:
            plot_alignment(
                time_series=time_series,
                generated_trajectory=generated_trajectory,
                switch_times=switch_times,
                bitcodes=bitcodes,
                n_pwl_units=cfg["P"],
                pwl_states=pwl_trajectory,
                save_path=cfg["figure_path"],
                show=True,
            )

        dist_path = cfg["figure_path"].replace(".png", "_bitcode_dist.png")
        plot_bitcode_distributions(
            bitcodes=bitcodes,
            segment_ids=segment_ids,
            n_pwl_units=cfg["P"],
            save_path=dist_path,
            show=True,
        )


if __name__ == "__main__":
    main()
