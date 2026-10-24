"""Experiment: AL-RNN regime detection on real EMA data.

Four studies are available — select via CONFIG["study"].  Within each study
every subject is stored as a (T, N) ndarray; select the subject via
CONFIG["subject_id"] (0-indexed, up to n_subjects - 1).

Missing observations (NaN) are linearly interpolated per variable before
training.  The standardisation and AL-RNN training pipeline is otherwise
identical to the bifurcation / sinusoid experiments.

Edit CONFIG, then run:

    python experiments/run_ema.py

from the repository root.
"""

import os
import sys
import mlflow
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from models.temporal_input import TemporalInput
from alrnn import train_alrnn
from alrnn.training import predict_sequence_using_gtf, predict_sequence_using_itf
from analysis.regime_logger import (
    bitcodes_from_pwl_states, log_bitcodes,
    count_unique_regimes, regime_change_times,
)
from analysis.alignment import plot_alignment
from analysis.consistency import bitcode_consistency
from experiments.runner import _print_temporal_input_norms


# ---------------------------------------------------------------------------
# Study registry
# ---------------------------------------------------------------------------
_EMA_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ema_data")

STUDIES = {
    "fisher":   "fisher_data_global_detrended.npy",
    "gschwind": "processed_data_gschwind_full.npy",
    "reeves":   "processed_data_reeves.npy",
    "rowland":  "processed_data_rowland.npy",
}


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CONFIG = {
    # --- data -----------------------------------------------------------------
    # Which study to use.  Keys: "fisher" | "gschwind" | "reeves" | "rowland"
    "study": "fisher",
    # Subject index (0 … n_subjects - 1).  Set to None to print subject count
    # and exit without training.
    "subject_id": 3,

    # --- temporal external input ----------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",         # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "monotone_gate",          # 'all' | 'pwl_only' | 'monotone_gate'

    # --- model ----------------------------------------------------------------
    "M": 30,
    "P": 1,
    "P_switch": None,   # slow-switching units; None = all P (backward-compatible)

    # --- training -------------------------------------------------------------
    "num_epochs": 300,
    # sequence_length=None uses the full trajectory each step.
    # batch_size must stay 1 — any larger value would sample the same start
    # index (0) repeatedly and compute identical gradients.
    "batch_size": 1,
    "sequence_length": None,
    "alpha": 1.0,
    "end_alpha": 1.0,
    "n_interleave": 1,
    "start_lr": 1e-3,
    "end_lr": 1e-3,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation -------------------------------------------------------
    "switch_lambda": 0.0,
    "pwl_timescale": 0.95,
    "timescale_lambda": 0.0,
    "incoming_lambda": 0.01,
    "pwl_state_lambda": 0.1,   # L2 penalty on |h[-P:]|² — prevents post-switch divergence

    # --- output ---------------------------------------------------------------
    # {study} and {subject_id} are filled in automatically.
    "figure_path": "ema_results/ema_{study}_s{subject_id:03d}_alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,
    # Maximum number of observation panels in the alignment plot.
    # EMA studies can have 20+ items; 3 keeps the figure readable.
    "plot_max_nodes": 2,

    # --- consistency ----------------------------------------------------------
    # Number of independent runs for ARI consistency scoring (1 = disabled).
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_subject(study: str, subject_id: int) -> np.ndarray:
    """Return (T, N) float32 array for the chosen subject."""
    path = os.path.join(_EMA_DATA_DIR, STUDIES[study])
    data = np.load(path, allow_pickle=True)
    n_subjects = len(data)
    if subject_id is None:
        print(f"Study '{study}': {n_subjects} subjects (ids 0 … {n_subjects - 1})")
        return None
    if subject_id >= n_subjects:
        raise ValueError(
            f"subject_id={subject_id} out of range for study '{study}' "
            f"({n_subjects} subjects, ids 0 … {n_subjects - 1})."
        )
    return np.asarray(data[subject_id], dtype=np.float32)


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values per variable (column-wise)."""
    x = x.copy()
    T, N = x.shape
    for n in range(N):
        col = x[:, n]
        nans = np.isnan(col)
        if not nans.any():
            continue
        t_idx = np.arange(T)
        col[nans] = np.interp(t_idx[nans], t_idx[~nans], col[~nans])
        x[:, n] = col
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG):
    mlflow.set_tracking_uri("sqlite:////gs/home/wernery/nonstationarity_al_rnn/mlflow_test.db")
    mlflow.set_experiment("EMA")
    
    with mlflow.start_run() as run:

        mlflow.log_params(cfg)

        study      = cfg["study"]
        subject_id = cfg["subject_id"]

        if study not in STUDIES:
            raise ValueError(f"Unknown study {study!r}. Options: {list(STUDIES)}")

        # ------------------------------------------------------------------
        # 1. Load subject data
        # ------------------------------------------------------------------
        raw = _load_subject(study, subject_id)
        if raw is None:
            return   # subject_id=None → just printed the count

        T, N = raw.shape
        nan_frac = np.isnan(raw).mean()
        if nan_frac > 0:
            print(f"  NaN fraction: {nan_frac:.1%} — interpolating linearly")
            raw = _interpolate_nans(raw)

        print(f"Data: study={study!r}, subject={subject_id}, T={T}, N={N}")

        # ------------------------------------------------------------------
        # 2. Standardise (dimension-wise)
        # ------------------------------------------------------------------
        mean = raw.mean(axis=0)
        std  = raw.std(axis=0).clip(min=1e-8)
        train_series = (raw - mean) / std

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
        # 4b. Build trajectory helper and callback
        # ------------------------------------------------------------------
        x_full           = torch.tensor(train_series, dtype=torch.float32).unsqueeze(0)
        u_full           = (torch.tensor(external_inputs, dtype=torch.float32).unsqueeze(0)
                            if external_inputs is not None else None)
        vis_alpha        = cfg.get("end_alpha") if cfg.get("end_alpha") is not None else cfg["alpha"]
        vis_n_interleave = cfg.get("n_interleave", 1)
        collect_pwl      = cfg["P"] > 0

        fig_path  = cfg["figure_path"].format(study=study, subject_id=subject_id)
        base_stem = fig_path.replace(".png", "")
        os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)

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

        def _alignment_callback(mdl, epoch):
            is_final  = (epoch >= cfg["num_epochs"])
            tag       = "_final" if is_final else f"_epoch{epoch:04d}"
            save_path = f"{base_stem}{tag}.png"
            gen_obs, pwl_traj, codes = _run_trajectory(mdl)
            n_trans = len(regime_change_times(codes))
            plot_alignment(
                time_series=raw,
                generated_trajectory=gen_obs,
                switch_times=np.array([], dtype=int),  # no ground truth
                bitcodes=codes,
                n_pwl_units=cfg["P"],
                pwl_states=pwl_traj,
                max_nodes=cfg.get("plot_max_nodes"),
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
                        time_series=raw,
                        generated_trajectory=gen_obs,
                        switch_times=np.array([], dtype=int),
                        bitcodes=codes,
                        n_pwl_units=cfg["P"],
                        pwl_states=pwl_traj,
                        max_nodes=cfg.get("plot_max_nodes"),
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
                temporal_input_target=cfg.get("temporal_input_target", "all"),
                pwl_state_lambda=cfg.get("pwl_state_lambda", 0.0),
                P_switch=cfg.get("P_switch"),
                epoch_callback=cb,
                callback_every=plot_every if plot_every is not None else 100,
            )
            print(f"  Final training loss: {losses[-1]:.6f}")

            mlflow.log_metric("final_training_loss", losses[-1])
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_losses = Path(tmp_dir) / "losses.npy"
                np.save(output_losses, losses)
                mlflow.log_artifact(output_losses)

            _, _, bitcodes = _run_trajectory(model)
            all_bitcodes.append(bitcodes)

        bitcodes      = all_bitcodes[-1]
        n_unique      = count_unique_regimes(bitcodes)
        change_times  = regime_change_times(bitcodes)
        n_transitions = len(change_times)
        generated_trajectory, pwl_trajectory, _ = _run_trajectory(model)

        mlflow.log_metric("n_unique", n_unique)
        mlflow.log_metric("n_transitions", n_transitions)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_bitcodes = Path(tmp_dir) / "bitcodes.npy"
            np.save(output_bitcodes, bitcodes)
            output_changetimes = Path(tmp_dir) / "changetimes.npy"
            np.save(output_changetimes, bitcodes)
            output_gen_traj = Path(tmp_dir) / "generated_trajectory.npy"
            np.save(output_gen_traj, generated_trajectory)
            output_pwl_traj = Path(tmp_dir) / "pwl_trajectory.npy"
            np.save(output_pwl_traj, pwl_trajectory)
            mlflow.log_artifacts(str(Path(tmp_dir)))

        # ------------------------------------------------------------------
        # 5. Summary
        # ------------------------------------------------------------------
        ti_target = cfg.get("temporal_input_target", "all")
        # can't compute temporal alignment score because no ground truth
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"  Study / subject          : {study} / {subject_id}")
        print(f"  T={T}, N={N}")
        print(f"  P / P_switch             : {cfg['P']} / {model.P_switch}")
        print(f"  temporal_input_target    : {ti_target}")
        print(f"  Inferred transitions     : {n_transitions}")
        print(f"  Unique regimes           : {n_unique}")
        if ti_target == "monotone_gate" and hasattr(model, "alpha_gate"):
            αs = [f"{v:.3f}" for v in model.alpha_gate.detach().tolist()]
            τs = [f"{v:.3f}" for v in model.tau_gate.detach().tolist()]
            print(f"  Gate α / τ (final)       : α={αs}  τ={τs}")
        if n_runs > 1:
            consistency = bitcode_consistency(all_bitcodes)
            print(f"  Consistency (ARI)        : {consistency['mean_ari']:.3f} ± {consistency['std_ari']:.3f}"
                f"  (n_runs={n_runs})")
            mlflow.log_metric("consistency", consistency)
        print("=" * 50)

        _print_temporal_input_norms(model)

        # ------------------------------------------------------------------
        # 6. Plots
        # ------------------------------------------------------------------
        if epoch_callback is None and n_runs == 1:
            plot_alignment(
                time_series=raw,
                generated_trajectory=generated_trajectory,
                switch_times=np.array([], dtype=int),
                bitcodes=bitcodes,
                n_pwl_units=cfg["P"],
                pwl_states=pwl_trajectory,
                max_nodes=cfg.get("plot_max_nodes"),
                save_path=fig_path,
                show=True,
            )

        # ------------------------------------------------------------------
        # 7. Save checkpoint
        # ------------------------------------------------------------------
        model_path = fig_path.replace("_alignment.png", "_model.pt").replace(".png", "_model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "M": model.M, "P": model.P, "N": model.N, "d_input": model.d_input,
            "config":          cfg,
            "mean":            mean.tolist(),
            "std":             std.tolist(),
            "train_series":    train_series,
            "time_series":     raw,
            "external_inputs": external_inputs,
            "study":           study,
            "subject_id":      subject_id,
        }, model_path)
        mlflow.log_artifact(model_path)
        print(f"\n  Model saved → {model_path}")

        results_path = Path(fig_path).parent
        mlflow.log_artifacts(results_path)


if __name__ == "__main__":
    main()
