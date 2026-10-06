"""Shared AL-RNN training and evaluation loop.

All experiment scripts (run_var, run_lorenz, run_sinusoid, run_bifurcation)
delegate to ``run_experiment`` after generating their data.  Only EMA is kept
separate due to its multi-subject / missing-data specifics.
"""

import os

import numpy as np
import torch

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


def run_experiment(
    cfg: dict,
    time_series: np.ndarray,
    switch_times: np.ndarray,
    segment_ids: np.ndarray,
    model_path: str,
    extra_checkpoint: dict | None = None,
    summary_lines: list[str] | None = None,
):
    """Train AL-RNN, evaluate, plot, and save a checkpoint.

    Parameters
    ----------
    cfg              : config dict; ``cfg["figure_path"]`` must already be
                       fully resolved (no format placeholders).
    time_series      : (T, N) raw observations.
    switch_times     : 1-D array of ground-truth switch indices.
    segment_ids      : (T,) integer segment labels (0, 1, 2, …).
    model_path       : destination for the ``.pt`` checkpoint.
    extra_checkpoint : extra keys merged into the saved checkpoint dict.
    summary_lines    : lines printed at the top of the SUMMARY block
                       (e.g. system name, scenario, parameter values).

    Returns
    -------
    model, losses, bitcodes
    """
    T, _         = time_series.shape
    n_segments   = len(np.unique(segment_ids))
    fig_path     = cfg["figure_path"]

    # ------------------------------------------------------------------
    # 1. Standardise
    # ------------------------------------------------------------------
    mean         = time_series.mean(axis=0)
    std          = time_series.std(axis=0).clip(min=1e-8)
    train_series = (time_series - mean) / std

    # ------------------------------------------------------------------
    # 2. Temporal external inputs
    # ------------------------------------------------------------------
    temporal_module = TemporalInput(
        mode=cfg["temporal_input_mode"] if cfg.get("use_temporal_input") else "none",
    )
    external_inputs = temporal_module.compute_features(T)
    d_input         = temporal_module.output_dim
    print(f"  temporal_input_mode={temporal_module.mode!r}, d_input={d_input}")

    # ------------------------------------------------------------------
    # 3. Shared helpers
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
    base_stem        = fig_path.replace(".png", "")
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
                max_nodes=cfg.get("plot_max_nodes"),
                save_path=save_path,
                show=False,
            )
            print(f"    [epoch {epoch}] → {save_path}  (transitions: {n_trans})")
        return _cb

    plot_every = cfg.get("plot_every")

    # ------------------------------------------------------------------
    # 4. Train (one or more independent runs)
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
            temporal_input_target=cfg.get("temporal_input_target", "all"),
            pwl_state_lambda=cfg.get("pwl_state_lambda", 0.0),
            P_switch=cfg.get("P_switch"),
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
    # 5. Summary
    # ------------------------------------------------------------------
    alignment = temporal_alignment_score(switch_times, change_times)
    ti_target = cfg.get("temporal_input_target", "all")
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for line in (summary_lines or []):
        print(line)
    print(f"  P / P_switch             : {cfg['P']} / {model.P_switch}")
    print(f"  temporal_input_target    : {ti_target}")
    print(f"  Ground-truth switches    : {n_segments - 1}")
    print(f"  Inferred transitions     : {n_transitions}  (ideal = {n_segments - 1})")
    print(f"  Unique regimes           : {n_unique}")
    print(f"  Alignment mean / max     : {alignment['mean_distance']:.1f} / "
          f"{alignment['max_distance']:.1f} steps")
    if ti_target == "monotone_gate" and hasattr(model, "alpha_gate"):
        αs = [f"{v:.3f}" for v in model.alpha_gate.detach().tolist()]
        τs = [f"{v:.3f}" for v in model.tau_gate.detach().tolist()]
        print(f"  Gate α / τ (final)       : α={αs}  τ={τs}")
    if n_runs > 1:
        consistency = bitcode_consistency(all_bitcodes)
        print(f"  Consistency (ARI)        : {consistency['mean_ari']:.3f} ± "
              f"{consistency['std_ari']:.3f}  (n_runs={n_runs})")
    print("=" * 50)

    _print_temporal_input_norms(model)

    # ------------------------------------------------------------------
    # 6. Final plots
    # ------------------------------------------------------------------
    if plot_every is None and n_runs == 1:
        plot_alignment(
            time_series=time_series,
            generated_trajectory=generated_trajectory,
            switch_times=switch_times,
            bitcodes=bitcodes,
            n_pwl_units=cfg["P"],
            pwl_states=pwl_trajectory,
            max_nodes=cfg.get("plot_max_nodes"),
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

    # ------------------------------------------------------------------
    # 7. Save checkpoint
    # ------------------------------------------------------------------
    checkpoint = {
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
    }
    if extra_checkpoint:
        checkpoint.update(extra_checkpoint)
    torch.save(checkpoint, model_path)
    print(f"\n  Model saved → {model_path}")

    return model, losses, bitcodes


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _print_temporal_input_norms(model):
    """Print the L2 norm of the temporal-input projection weight per latent unit.

    For 'all': every unit has its own row in C.
    For 'pwl_only' / 'monotone_gate': linear units receive zero weight; PWL
    units have rows in C_pwl.

    Comparing the two variants shows whether the unconstrained model
    spontaneously concentrates its C weight on the PWL units (which the
    constrained variants enforce by design).
    """
    norms = model.temporal_input_norms()
    if norms is None:
        return  # temporal input disabled

    M, P = model.M, model.P
    target = model.temporal_input_target
    print(f"\n  Temporal-input weight norms  [target={target!r}]")
    print(f"  {'unit':>5}  {'type':<8}  ‖C[i,:]‖₂")
    print(f"  {'-'*5}  {'-'*8}  {'-'*10}")
    for i, v in enumerate(norms.tolist()):
        kind = "PWL" if i >= M - P else "linear"
        print(f"  {i:>5}  {kind:<8}  {v:.5f}")
