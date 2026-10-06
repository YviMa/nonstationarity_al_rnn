"""Experiment: AL-RNN seizure detection on CHB-MIT EEG (patients chb03 / chb15).

chb03: sessions 1, 2, 3, 4.
chb15: sessions 10, 15, 17, 20, 22, 28.
Each .npz file contains a window of
60 s pre-seizure + ictal period + 60 s post-seizure at 256 Hz across 23
EEG channels that have already been bandpass-filtered, notch-filtered,
common-average re-referenced, clipped, and z-scored per channel.

Ground-truth regime boundaries:
    interictal (0) → ictal (1)    at onset_sample
    ictal (1) → postictal (2)     at offset_sample

Because the raw sampling rate is 256 Hz, individual windows can exceed
10 000 samples.  `subsample_factor` downsamples the time axis before
training so the model focuses on slower regime dynamics rather than
individual EEG cycles.

Edit CONFIG, then run:

    python experiments/run_eeg.py

from the repository root.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from experiments.runner import run_experiment


_EEG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "eeg", "processed")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
_PATIENT_SESSIONS = {
    "chb03": (1, 2, 3, 4),
    "chb15": (10, 15, 17, 20, 22, 28),
}

CONFIG = {
    # --- data -----------------------------------------------------------------
    # Which patient to use ('chb03' | 'chb15').
    "patient": "chb15",
    # Which session to use.  chb03: 1–4  |  chb15: 10, 15, 17, 20, 22, 28.
    "session": 20,
    # Downsample the time axis by this integer factor before training.
    # At 256 Hz a single window can be 10 000+ samples; subsampling reduces
    # the sequence length and focuses the model on slower dynamics.
    # 1 = no subsampling; 8 → effective 32 Hz; 16 → effective 16 Hz.
    "subsample_factor": 16,

    # --- temporal external input ----------------------------------------------
    "use_temporal_input": True,
    "temporal_input_mode": "linear",        # 'none' | 'linear' | 'fourier'
    "temporal_input_target": "monotone_gate",  # 'all' | 'pwl_only' | 'monotone_gate'

    # --- model ----------------------------------------------------------------
    "M": 100,
    "P": 4,
    # P_switch: slow-switching units that receive temporal input + regularization.
    # None = all P units are slow-switching (backward-compatible).
    # E.g. P_switch=1: 1 slow seizure-detection unit + 1 free geometry unit.
    "P_switch": 2,

    # --- training -------------------------------------------------------------
    "num_epochs": 500,
    "batch_size": 16,
    "sequence_length": 50,
    "alpha": 1.0,
    "end_alpha": 1.0,
    "n_interleave": 4,
    "start_lr": 1e-3,
    "end_lr": 5e-4,
    "batches_per_epoch": 10,
    "forcing_mode": "itf",
    "device": "cpu",
    "verbose": True,

    # --- regularisation -------------------------------------------------------
    "switch_lambda":    0.0,
    "pwl_timescale":    0.5,
    "timescale_lambda": 0.01,
    "incoming_lambda":  0.1,
    "pwl_state_lambda": 0.01,

    # --- output ---------------------------------------------------------------
    # {patient} and {session} are filled in automatically.
    "figure_path": "eeg_results/eeg_{patient}_s{session:02d}_alignment.png",
    # Mid-training alignment plots every N epochs (None = only at the end).
    "plot_every": 100,
    # Maximum number of EEG channel panels to show in the alignment plot.
    # With 23 channels the figure would be unreadably tall; 4 keeps it compact.
    "plot_max_nodes": 4,

    # --- consistency ----------------------------------------------------------
    "n_runs": 1,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: dict = CONFIG):
    patient = cfg.get("patient", "chb03")
    if patient not in _PATIENT_SESSIONS:
        raise ValueError(f"patient must be one of {list(_PATIENT_SESSIONS)}, got {patient!r}")
    session = cfg["session"]
    if session not in _PATIENT_SESSIONS[patient]:
        raise ValueError(f"session must be one of {_PATIENT_SESSIONS[patient]} for {patient}, got {session}")

    fname = os.path.join(_EEG_DIR, f"{patient}_{session:02d}_processed.npz")
    d     = np.load(fname, allow_pickle=True)

    # (23, n_samples) → (n_samples, 23); already z-scored per channel
    x_raw         = d["x_window"].T.astype(np.float32)
    onset_sample  = int(d["onset_sample"])
    offset_sample = int(d["offset_sample"])
    sfreq         = float(d["sfreq"])
    ch_names      = d["ch_names"].tolist()

    T_raw, N = x_raw.shape
    print(f"Loaded: {patient} session {session}  |  T={T_raw}, N={N} channels, "
          f"sfreq={sfreq:.0f} Hz")
    print(f"  onset_sample={onset_sample}  offset_sample={offset_sample}  "
          f"seizure duration: {(offset_sample - onset_sample) / sfreq:.1f} s")
    print(f"  total window: {T_raw / sfreq:.1f} s")

    # ------------------------------------------------------------------
    # Subsampling
    # ------------------------------------------------------------------
    sf = max(1, int(cfg.get("subsample_factor", 1)))
    if sf > 1:
        x_ds          = x_raw[::sf]
        onset_ds      = onset_sample  // sf
        offset_ds     = offset_sample // sf
        T             = len(x_ds)
        eff_sfreq     = sfreq / sf
        print(f"  Subsampled ×{sf}: T={T}, effective sfreq={eff_sfreq:.1f} Hz")
    else:
        x_ds      = x_raw
        onset_ds  = onset_sample
        offset_ds = offset_sample
        T         = T_raw
        eff_sfreq = sfreq

    # ------------------------------------------------------------------
    # Ground-truth segmentation
    # 3 regimes: interictal (0) → ictal (1) → postictal (2)
    # ------------------------------------------------------------------
    switch_times = np.array([onset_ds, offset_ds], dtype=int)
    segment_ids  = np.zeros(T, dtype=int)
    segment_ids[onset_ds:offset_ds] = 1
    segment_ids[offset_ds:]         = 2

    # ------------------------------------------------------------------
    # Resolve paths (fill in {patient} and {session} placeholders)
    # ------------------------------------------------------------------
    fig_path   = cfg["figure_path"].format(patient=patient, session=session)
    model_path = fig_path.replace(".png", "_model.pt")
    cfg        = dict(cfg)
    cfg["figure_path"] = fig_path

    # ------------------------------------------------------------------
    # Hand off to shared runner
    # ------------------------------------------------------------------
    run_experiment(
        cfg, x_ds, switch_times, segment_ids,
        model_path=model_path,
        extra_checkpoint={
            "session":          session,
            "onset_sample":     onset_ds,
            "offset_sample":    offset_ds,
            "subsample_factor": sf,
            "sfreq_original":   sfreq,
            "sfreq_effective":  eff_sfreq,
            "ch_names":         ch_names,
        },
        summary_lines=[
            f"  System                   : CHB-MIT EEG  patient {patient}",
            f"  Session                  : {session}",
            f"  Channels (N)             : {N}",
            f"  Subsample factor         : {sf}  (effective {eff_sfreq:.1f} Hz)",
            f"  Seizure onset / offset   : t={onset_ds} / {offset_ds}  "
            f"({onset_ds / eff_sfreq:.1f} s / {offset_ds / eff_sfreq:.1f} s)",
        ],
    )


if __name__ == "__main__":
    main()
