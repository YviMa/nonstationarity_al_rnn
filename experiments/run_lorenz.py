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
    # --- Lorenz parameter per segment (ρ / rho) ----------------------------
    # [28, 35]    both chaotic, different attractor shape
    # [14, 28]    fixed-point (C±) → strange attractor
    # [28, 99.96] chaos → period-3 window
    "param_values": [28.0, 55.0],#28,45

    # --- Fixed Lorenz parameters (forwarded as system_kwargs) ---------------
    "sigma": 10.0,
    "beta":  8.0 / 3.0,

    # --- Integration ----------------------------------------------------------
    "segment_lengths": [1000, 1000],  # samples per segment
    "dt": 0.01,                        # RK4 step size (0.01 = standard)
    "warmup": 2000,                    # steps to land on attractor before recording
    "noise_std": 0.0,
    "random_seed": 42,

    # --- Temporal external input ----------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",   # 'none' | 'linear' | 'fourier'

    # --- Model ----------------------------------------------------------------
    "M": 30,
    "P": 4,

    # --- Training -------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 78,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # Lorenz is chaotic — use a larger n_interleave to prevent error blow-up
    # during free runs.  4–8 is typical for the classic attractor.
    "n_interleave": 5,
    "start_lr": 1e-3,
    "end_lr": 1e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- Regularisation -------------------------------------------------------
    "switch_lambda":    0.0,
    "pwl_timescale":    0.98,
    "timescale_lambda": 0.005,
    "incoming_lambda":  0.1,

    # --- Output ---------------------------------------------------------------
    "figure_path": "lorenz_results/lorenz_alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,

    # --- Consistency ----------------------------------------------------------
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
    rho_vals = cfg["param_values"]
    sigma    = cfg.get("sigma", 10.0)
    beta     = cfg.get("beta",  8.0 / 3.0)

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

    n_segments = len(rho_vals)
    T, N = time_series.shape
    print(f"  T={T}, N={N}, switch_times={switch_times.tolist()}")
    _describe_lorenz(rho_vals, time_series, segment_ids)

    # ------------------------------------------------------------------
    # 2. Standardise
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
    external_inputs = temporal_module.compute_features(T)
    d_input = 0 if external_inputs is None else external_inputs.shape[1]
    print(f"  temporal_input_mode={temporal_module.mode!r}, d_input={d_input}")

    # ------------------------------------------------------------------
    # 4. Helpers
    # ------------------------------------------------------------------
    x_full      = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    u_full      = (torch.tensor(external_inputs, dtype=torch.float32).unsqueeze(0)
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

    bitcodes     = all_bitcodes[-1]
    n_unique     = count_unique_regimes(bitcodes)
    change_times = regime_change_times(bitcodes)
    n_transitions = len(change_times)
    generated_trajectory, pwl_trajectory, _ = _run_trajectory(model)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    alignment = temporal_alignment_score(switch_times, change_times)
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  System                   : Lorenz-63")
    print(f"  ρ per segment            : {rho_vals}")
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

    # ------------------------------------------------------------------
    # 8. Save model + run metadata
    # ------------------------------------------------------------------
    model_path = cfg["figure_path"].replace("_alignment.png", "_model.pt").replace(".png", "_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "M": model.M, "P": model.P, "N": model.N, "d_input": model.d_input,
        "config":          cfg,
        "mean":            mean.tolist(),
        "std":             std.tolist(),
        "switch_times":    switch_times.tolist(),
        "segment_ids":     segment_ids.tolist(),
        "train_series":    train_series,
        "time_series":     time_series,
        "external_inputs": external_inputs,
        "rho_vals":        rho_vals,
    }, model_path)
    print(f"\n  Model saved → {model_path}")


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def _describe_lorenz(rho_vals, time_series, segment_ids):
    """Print per-segment attractor statistics."""
    n_segs = len(rho_vals)
    for i, rho in enumerate(rho_vals):
        seg = time_series[segment_ids == i]
        means = seg.mean(axis=0)
        stds  = seg.std(axis=0)
        print(f"  Segment {i}  ρ={rho:6.2f}  "
              f"mean=({means[0]:+.2f}, {means[1]:+.2f}, {means[2]:+.2f})  "
              f"std=({stds[0]:.2f}, {stds[1]:.2f}, {stds[2]:.2f})")


if __name__ == "__main__":
    main()
