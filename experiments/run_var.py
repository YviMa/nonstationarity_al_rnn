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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from data.switching_var import generate_switching_var
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
    # --- scenario -----------------------------------------------------------
    # "modular_to_hub" | "critical_slowing" | "sign_flip" | "random"
    "scenario": "modular_to_hub",
  #  "scenario": "critical_slowing",
  #  "scenario": "sign_flip",

    # --- data ---------------------------------------------------------------
    # n_nodes must be even >= 4 for structured scenarios
    "n_nodes": 6,
    "segment_lengths": [200, 800],
    "noise_std": 0.1,
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
    "sr_high": 0.97,

    # random: interpolation strength between consecutive matrices (0=no change)
    "switch_strength": 0.5,

    # all scenarios: number of complex-conjugate eigenvalue pairs (None = n//2)
    "n_oscillation_modes": None,

    # --- temporal external input --------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",   # 'none' | 'linear' | 'fourier'

    # --- model ---------------------------------------------------------------
    "M": 20,
    "P": 2,

    # --- training ------------------------------------------------------------
    "num_epochs": 300,
    "batch_size": 16,
    "sequence_length": 50,
    "alpha": 1.0,
    "end_alpha": 1.0,
    # Apply teacher forcing every n_interleave steps (1 = every step).
    # Increase for systems with strong autonomous dynamics (e.g. 4–8).
    "n_interleave": 3,
    "start_lr": 1e-3,
    "end_lr": 5e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation ------------------------------------------------------
    "switch_lambda": 0.0,
    "pwl_timescale": 0.97,
    "timescale_lambda": 0.01,
    # L2 penalty on W[-P:, :] (incoming weights to PWL units) toward 0.
    # Keeps PWL units as slow integrators; outgoing W[:, -P:] stays free.
    "incoming_lambda": 0.01,

    # --- output --------------------------------------------------------------
    # {scenario} is filled in automatically.
    "figure_path": "var_results/{scenario}_alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,

    # --- consistency ---------------------------------------------------------
    # Number of independent runs for ARI consistency scoring (1 = disabled).
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG) -> None:
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
    n_segments = len(cfg["segment_lengths"])
    print(f"  T={T}, N={N}, segments={n_segments}, switch_times={switch_times.tolist()}")

    # Brief description of what changed
    _describe_scenario(scenario, matrices, cfg)

    # ------------------------------------------------------------------
    # 2. Standardise (dimension-wise)
    # ------------------------------------------------------------------
    mean = time_series.mean(axis=0)
    std  = time_series.std(axis=0).clip(min=1e-8)
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
    fig_path  = cfg["figure_path"].format(scenario=scenario)
    base_stem = fig_path.replace(".png", "")
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)

    x_full           = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
    u_full           = (torch.tensor(external_inputs, dtype=torch.float32).unsqueeze(0)
                        if external_inputs is not None else None)
    vis_alpha        = cfg.get("end_alpha") if cfg.get("end_alpha") is not None else cfg["alpha"]
    vis_n_interleave = cfg.get("n_interleave", 1)
    collect_pwl      = cfg["P"] > 0

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
    print(f"  Scenario                 : {scenario}")
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
            save_path=fig_path,
            show=True,
        )

    dist_path = fig_path.replace(".png", "_bitcode_dist.png")
    plot_bitcode_distributions(
        bitcodes=bitcodes,
        segment_ids=segment_ids,
        n_pwl_units=cfg["P"],
        save_path=dist_path,
        show=True,
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
