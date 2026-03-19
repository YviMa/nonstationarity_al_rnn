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

import numpy as np
import torch

from data.switching_sinusoid import generate_switching_sinusoid
from models.temporal_input import TemporalInput
from alrnn import train_alrnn
from alrnn.training import predict_sequence_using_gtf, predict_sequence_using_itf
from analysis.regime_logger import (
    log_bitcodes, bitcodes_from_pwl_states,
    count_unique_regimes, regime_change_times,
)
from analysis.alignment import temporal_alignment_score, plot_alignment, plot_bitcode_distributions
from analysis.consistency import bitcode_consistency


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- data ----------------------------------------------------------------
    "n_nodes": 5,
    "n_switches": 1,
    # How much frequencies change at each switch: 0 = identical, 1 = fully new.
    "switch_strength": 0.3,
    "segment_lengths": [200, 200],
    "n_components": 3,           # sinusoidal components per node
    "freq_range": [0.05, 0.2],   # cycles/sample → periods 5–20 steps
    "noise_std": 0.0,
    "random_seed": 42,

    # --- temporal external input --------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",   # 'none' | 'linear' | 'fourier'

    # --- model ---------------------------------------------------------------
    "M": 20,
    "P": 1,

    # --- training ------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 50,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # Apply teacher forcing every n_interleave steps (1 = every step).
    "n_interleave": 4,
    "start_lr": 1e-3,
    "end_lr": 1e-3,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation ------------------------------------------------------
    "switch_lambda": 0.0,
    "pwl_timescale": 0.95,
    "timescale_lambda": 0.0,
    "incoming_lambda": 0.0,

    # --- output --------------------------------------------------------------
    "figure_path": "sinusoid_results/alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 200,

    # --- consistency ---------------------------------------------------------
    # Number of independent runs for ARI consistency scoring (1 = disabled).
    "n_runs": 3,
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
    # 2. Standardise (dimension-wise)
    # ------------------------------------------------------------------
    mean = time_series.mean(axis=0)
    std  = time_series.std(axis=0)
    std[std == 0] = 1.0
    train_series = (time_series - mean) / std

    # ------------------------------------------------------------------
    # 3. Temporal external inputs
    # ------------------------------------------------------------------
    temporal_module = TemporalInput(
        mode=cfg["temporal_input_mode"] if cfg["use_temporal_input"] else "none"
    )
    external_inputs = temporal_module.compute_features(T)
    d_input = temporal_module.output_dim
    print(f"  temporal_input_mode={temporal_module.mode!r}, d_input={d_input}")

    # ------------------------------------------------------------------
    # 4. Build trajectory helper
    # ------------------------------------------------------------------
    x_full           = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    u_full           = (torch.tensor(external_inputs, dtype=torch.float32).unsqueeze(0)
                        if external_inputs is not None else None)
    vis_alpha        = cfg.get("end_alpha") if cfg.get("end_alpha") is not None else cfg["alpha"]
    vis_n_interleave = cfg.get("n_interleave", 1)
    collect_pwl      = cfg["P"] > 0
    base_stem        = cfg["figure_path"].replace(".png", "")
    os.makedirs(os.path.dirname(cfg["figure_path"]) or ".", exist_ok=True)

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

    def _make_callback(stem):
        def _cb(mdl, epoch):
            is_final  = (epoch >= cfg["num_epochs"])
            tag       = "_final" if is_final else f"_epoch{epoch:04d}"
            save_path = f"{stem}{tag}.png"
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
                show=False,
            )
            print(f"    [epoch {epoch}] → {save_path}  (transitions: {n_trans})")
        return _cb

    plot_every = cfg.get("plot_every")

    # ------------------------------------------------------------------
    # 5. Train (one or more independent runs)
    # ------------------------------------------------------------------
    n_runs       = max(1, cfg.get("n_runs", 1))
    all_bitcodes = []

    for run_idx in range(n_runs):
        run_tag  = f"run {run_idx + 1}/{n_runs} " if n_runs > 1 else ""
        run_stem = base_stem + (f"_run{run_idx + 1:02d}" if n_runs > 1 else "")
        cb       = _make_callback(run_stem) if plot_every is not None else None

        print(f"\nTraining AL-RNN  {run_tag}M={cfg['M']}, P={cfg['P']}, "
              f"d_input={d_input}, epochs={cfg['num_epochs']} …")

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
            incoming_lambda=cfg.get("incoming_lambda", 0.0),
            epoch_callback=cb,
            callback_every=plot_every if plot_every is not None else 100,
        )
        print(f"  Final training loss: {losses[-1]:.6f}")

        _, _, bitcodes = _run_trajectory(model)
        all_bitcodes.append(bitcodes)

    bitcodes      = all_bitcodes[-1]
    n_unique      = count_unique_regimes(bitcodes)
    change_times  = regime_change_times(bitcodes)
    n_transitions = len(change_times)
    generated_trajectory, pwl_trajectory, _ = _run_trajectory(model)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    alignment = temporal_alignment_score(switch_times, change_times)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Ground-truth switches    : {n_segments - 1}")
    print(f"  Inferred transitions     : {n_transitions}  (ideal = {n_segments - 1})")
    print(f"  Unique regimes           : {n_unique}")
    print(f"  Alignment mean / max     : {alignment['mean_distance']:.1f} / "
          f"{alignment['max_distance']:.1f} steps")
    if n_runs > 1:
        consistency = bitcode_consistency(all_bitcodes)
        print(f"  Consistency (ARI)        : {consistency['mean_ari']:.3f} ± "
              f"{consistency['std_ari']:.3f}  (n_runs={n_runs})")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 7. Final plots
    # ------------------------------------------------------------------
    if plot_every is None and n_runs == 1:
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
