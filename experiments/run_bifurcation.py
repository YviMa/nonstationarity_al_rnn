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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.switching_ode import generate_switching_ode
from models.temporal_input import TemporalInput
from alrnn import train_alrnn
from alrnn.training import predict_sequence_using_gtf, predict_sequence_using_itf
from analysis.regime_logger import (
    log_bitcodes, bitcodes_from_pwl_states,
    count_unique_regimes, regime_change_times,
)
from analysis.alignment import (
    temporal_alignment_score, plot_alignment, plot_bitcode_distributions,
)
from analysis.consistency import bitcode_consistency


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
    "warmup": 0,        # warmup steps before recording (lands on attractor)
    "noise_std": 0.0,
    "random_seed": 42,

    # System-specific kwargs (forwarded to ODE factory):
    "omega": 1.0,          # hopf only: angular frequency of the limit cycle

    # --- temporal external input -------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",  # 'none' | 'linear' | 'fourier'

    # --- model -------------------------------------------------------------
    "M": 20,
    "P": 2,

    # --- training ----------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 50,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # Apply teacher forcing every n_interleave steps (1 = every step).
    # Increase for systems with complex autonomous dynamics (e.g. 10–16).
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

    # --- output ------------------------------------------------------------
    "figure_path": "bifurcation_results/alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,

    # --- consistency -------------------------------------------------------
    # Number of independent runs for ARI consistency scoring (1 = disabled).
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG):
    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    system_kwargs = {}
    if cfg["system"] == "hopf" and "omega" in cfg:
        system_kwargs["omega"] = cfg["omega"]

    time_series, switch_times, segment_ids = generate_switching_ode(
        system=cfg["system"],
        param_values=cfg["param_values"],
        segment_lengths=cfg["segment_lengths"],
        dt=cfg["dt"],
        warmup=cfg["warmup"],
        noise_std=cfg["noise_std"],
        random_seed=cfg["random_seed"],
        **system_kwargs,
    )

    n_segments = len(cfg["param_values"])
    T, N = time_series.shape
    print(f"Data: system={cfg['system']!r}, T={T}, N={N}, "
          f"params={cfg['param_values']}, switches at {switch_times.tolist()}")

    # ------------------------------------------------------------------
    # 2. Standardise (dimension-wise mean/std)
    # ------------------------------------------------------------------
    mean = time_series.mean(axis=0)
    std  = time_series.std(axis=0).clip(min=1e-8)
    train_series = (time_series - mean) / std

    # ------------------------------------------------------------------
    # 3. Temporal external input
    # ------------------------------------------------------------------
    temporal_module = TemporalInput(
        mode=cfg["temporal_input_mode"] if cfg.get("use_temporal_input") else "none",
    )
    external_inputs = temporal_module.compute_features(T)  # (T, d) or None
    d_input = 0 if external_inputs is None else external_inputs.shape[1]
    print(f"  temporal_input_mode={temporal_module.mode!r}, d_input={d_input}")

    # ------------------------------------------------------------------
    # 4b. Helper: run one forward pass → trajectory + bitcodes
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

    # ------------------------------------------------------------------
    # 4c. Mid-training alignment-plot callback
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
            show=False,
        )
        print(f"    [epoch {epoch}] → {save_path}  (transitions: {n_trans})")

    plot_every     = cfg.get("plot_every")
    epoch_callback = _alignment_callback if plot_every is not None else None

    # ------------------------------------------------------------------
    # 4. Train AL-RNN (one or more independent runs)
    # ------------------------------------------------------------------
    n_runs       = max(1, cfg.get("n_runs", 1))
    all_bitcodes = []

    for run_idx in range(n_runs):
        run_tag = f"run {run_idx + 1}/{n_runs} " if n_runs > 1 else ""
        print(f"\nTraining AL-RNN  {run_tag}M={cfg['M']}, P={cfg['P']}, "
              f"d_input={d_input}, epochs={cfg['num_epochs']} …")

        if n_runs > 1:
            run_stem = base_stem + f"_run{run_idx + 1:02d}"
            def _run_callback(mdl, epoch, _stem=run_stem):
                is_final  = (epoch >= cfg["num_epochs"])
                tag       = "_final" if is_final else f"_epoch{epoch:04d}"
                save_path = f"{_stem}{tag}.png"
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
            cb = _run_callback if plot_every is not None else None
        else:
            cb = epoch_callback

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
    # 5. Summary
    # ------------------------------------------------------------------
    alignment = temporal_alignment_score(switch_times, change_times)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  System                   : {cfg['system']}")
    print(f"  Params per segment       : {cfg['param_values']}")
    print(f"  Ground-truth switches    : {n_segments - 1}")
    print(f"  Inferred transitions     : {n_transitions}  (ideal = {n_segments - 1})")
    print(f"  Unique regimes           : {n_unique}")
    print(f"  Alignment mean / max     : {alignment['mean_distance']:.1f} / "
          f"{alignment['max_distance']:.1f} steps")
    if n_runs > 1:
        consistency = bitcode_consistency(all_bitcodes)
        print(f"  Consistency (ARI)        : {consistency['mean_ari']:.3f} ± {consistency['std_ari']:.3f}"
              f"  (n_runs={n_runs})")
    print("=" * 50)

    # ------------------------------------------------------------------
    # 6. Final plots
    # ------------------------------------------------------------------
    if epoch_callback is None and n_runs == 1:
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
