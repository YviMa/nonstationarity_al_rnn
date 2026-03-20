"""Classical stationarity and structural-change tests on switching time series.

Purpose
-------
Establish a baseline for what univariate / sliding-window statistics can and
cannot detect, against which AL-RNN regime detection is compared.

Tests applied
-------------
Per-node (univariate, across the full series):
  ADF              — Augmented Dickey-Fuller.  H0: unit root.  p<α → stationary.
  KPSS             — H0: stationary.  p<α → non-stationary.
  CUSUM-of-squares — Inclan-Tiao (1994) statistic for variance stability.
                     Designed for time series; does not assume iid.
                     Computes the sup of the centered cumulative sum of squared
                     residuals; the empirical p-value is obtained via bootstrap.
  Δmean / Δvar     — Standardised mean shift and relative variance change between
                     the two known segments.

Multivariate (one value per dataset):
  ΔCorr            — Frobenius norm of the correlation-matrix difference.
  Andrews sup-F    — Andrews (1993) sup-Wald test for a structural break at an
                     UNKNOWN date.  Scans the Chow F-statistic over candidate
                     break dates in the middle 70% of the series; takes the
                     supremum.  Bootstrap p-value (residual bootstrap, 199 reps).
                     Also reports the most likely break date (argmax of F-scan).
  E-divisive stat  — Energy-based change point statistic (Chen & Zhang 2015).
                     Non-parametric, multivariate, detects any distributional
                     shift.  Empirical p-value from permutation test.

Sliding-window trajectories (saved in plots):
  Rolling std, lag-1 autocorrelation, and cross-correlation distance per window.
  Windows are NON-overlapping to reduce dependency between test statistics.

Outputs
-------
  stationarity_results/per_node.csv   — one row per (dataset × node)
  stationarity_results/summary.csv    — one row per dataset
  stationarity_results/<nn>_<label>_sliding.png  — trajectory plots

Edit the DATASETS list below to control what is tested.
Run from the repository root:  python experiments/stationarity_tests.py
"""

import csv
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not found — ADF/KPSS tests will be skipped.")
    print("          Install with:  pip install statsmodels\n")

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except ImportError:
    HAS_RUPTURES = False
    print("[WARNING] ruptures not found — change-point detection will be skipped.")
    print("          Install with:  pip install ruptures\n")


# ---------------------------------------------------------------------------
# Dataset definitions — edit freely
# ---------------------------------------------------------------------------

DATASETS = [
    # ---------------------------------------------------------------- sinusoid
    {
        "label": "sinusoid_weak_switch",
        "description": "Sinusoid — switch_strength=0.3 (small frequency change)",
        "generator": "sinusoid",
        "n_nodes": 5,
        "n_switches": 1,
        "switch_strength": 0.3,
        "segment_lengths": [200, 200],
        "n_components": 3,
        "freq_range": (0.05, 0.2),
        "noise_std": 0.0,
        "random_seed": 42,
    },
    {
        "label": "sinusoid_strong_switch",
        "description": "Sinusoid — switch_strength=1.0 (maximal frequency shift)",
        "generator": "sinusoid",
        "n_nodes": 5,
        "n_switches": 1,
        "switch_strength": 1.0,
        "segment_lengths": [200, 200],
        "n_components": 3,
        "freq_range": (0.05, 0.2),
        "noise_std": 0.0,
        "random_seed": 42,
    },
    # ---------------------------------------------------------- VAR scenarios
    {
        "label": "var_modular_to_hub",
        "description": "VAR — modular_to_hub (marginals unchanged, topology changes)",
        "generator": "var",
        "scenario": "modular_to_hub",
        "n_nodes": 6,
        "segment_lengths": [200, 200],
        "noise_std": 0.1,
        "spectral_radius": 0.90,
        "hub_strength": 0.2,
        "cross_coupling": 0.05,
        "random_seed": 42,
    },
    {
        "label": "var_critical_slowing",
        "description": "VAR — critical_slowing (sr: 0.80→0.97, variance matched)",
        "generator": "var",
        "scenario": "critical_slowing",
        "n_nodes": 6,
        "segment_lengths": [200, 200],
        "noise_std": 0.1,
        "sr_low": 0.80,
        "sr_high": 0.97,
        "random_seed": 42,
    },
    {
        "label": "var_sign_flip",
        "description": "VAR — sign_flip (cross-cluster coupling reverses sign)",
        "generator": "var",
        "scenario": "sign_flip",
        "n_nodes": 6,
        "segment_lengths": [200, 200],
        "noise_std": 0.1,
        "spectral_radius": 0.90,
        "random_seed": 42,
    },
]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

OUTPUT_DIR         = "stationarity_results"
WINDOW_SIZE        = 50    # non-overlapping window length for sliding tests
ALPHA_ADF          = 0.05
ALPHA_KPSS         = 0.05
ALPHA_BREAK        = 0.05
N_BOOTSTRAP        = 499   # permutations for CUSUM-sq and E-divisive p-values
RUPTURES_PEN       = "bic" # penalty for ruptures: "bic" | "aic" | float


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def _generate(cfg: dict):
    if cfg["generator"] == "sinusoid":
        from data.switching_sinusoid import generate_switching_sinusoid
        ts, switch_times, _, seg_ids = generate_switching_sinusoid(
            n_nodes=cfg["n_nodes"],
            n_switches=cfg["n_switches"],
            switch_strength=cfg["switch_strength"],
            segment_lengths=cfg["segment_lengths"],
            n_components=cfg.get("n_components", 3),
            freq_range=cfg.get("freq_range", (0.05, 0.2)),
            noise_std=cfg.get("noise_std", 0.0),
            random_seed=cfg.get("random_seed", 42),
        )
    elif cfg["generator"] == "var":
        from data.switching_var import generate_switching_var
        ts, switch_times, _, seg_ids = generate_switching_var(
            scenario=cfg["scenario"],
            n_nodes=cfg["n_nodes"],
            segment_lengths=cfg["segment_lengths"],
            noise_std=cfg.get("noise_std", 0.1),
            spectral_radius=cfg.get("spectral_radius", 0.9),
            sr_low=cfg.get("sr_low", 0.70),
            sr_high=cfg.get("sr_high", 0.97),
            hub_strength=cfg.get("hub_strength", 0.4),
            cross_coupling=cfg.get("cross_coupling", 0.05),
            random_seed=cfg.get("random_seed", 42),
        )
    else:
        raise ValueError(f"Unknown generator: {cfg['generator']!r}")
    return ts, switch_times, seg_ids


# ---------------------------------------------------------------------------
# Univariate tests
# ---------------------------------------------------------------------------

def _adf(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, _, _, _, _ = adfuller(x, autolag="AIC")
    return stat, pval


def _kpss_test(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pval, _, _ = kpss(x, regression="c", nlags="auto")
    return stat, pval


def _cusum_sq_stat(x: np.ndarray) -> float:
    """Inclan-Tiao (1994) CUSUM-of-squares statistic.

    Measures the maximum deviation of the cumulative squared residuals from
    the expected linear trend.  Large values → variance is not constant.
    Does not require iid — valid for weakly dependent series.
    """
    e2   = (x - x.mean()) ** 2
    cumE = np.cumsum(e2)
    k    = np.arange(1, len(x) + 1)
    Ck   = cumE / cumE[-1] - k / len(x)   # centred cumulative fraction
    return float(np.max(np.abs(Ck)))


def _cusum_sq_pvalue(x: np.ndarray, n_boot: int = N_BOOTSTRAP,
                     rng: np.random.Generator = None) -> tuple:
    """Bootstrap p-value for the CUSUM-of-squares statistic.

    H0: constant variance.  Null distribution obtained by block-bootstrap
    (block length ≈ √T) to preserve short-range autocorrelation structure.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    obs_stat = _cusum_sq_stat(x)
    T        = len(x)
    block    = max(1, int(np.sqrt(T)))
    n_blocks = int(np.ceil(T / block))

    null_stats = np.empty(n_boot)
    for b in range(n_boot):
        starts  = rng.integers(0, T, size=n_blocks)
        boot    = np.concatenate([x[s : s + block] for s in starts])[:T]
        null_stats[b] = _cusum_sq_stat(boot)

    pval = float((null_stats >= obs_stat).mean())
    return obs_stat, pval


# ---------------------------------------------------------------------------
# Multivariate tests
# ---------------------------------------------------------------------------

def _andrews_supF(ts: np.ndarray, trim: float = 0.15,
                  n_boot: int = 199,
                  rng: np.random.Generator = None) -> tuple:
    """Andrews (1993) sup-F test for a VAR(1) structural break at an unknown date.

    Scans the Chow F-statistic over all candidate break dates in the middle
    (1 - 2*trim) fraction of the series (default: middle 70%).  The supremum
    over all candidates is the test statistic; its location is the MLE of the
    break date.

    P-value is obtained via a residual bootstrap: the full-sample VAR(1) fit is
    used to generate bootstrap samples under H0 (no break), and the null
    distribution of sup-F is estimated from those samples.

    Returns
    -------
    supF     : float  — supremum F-statistic
    break_t  : int    — estimated break date in [1, T-1] (original time index)
    pvalue   : float  — bootstrap p-value (H0: no structural break)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X       = ts[:-1]          # (T-1, N)  regressors z_{t-1}
    Y       = ts[1:]           # (T-1, N)  targets    z_t
    T_eff, N = X.shape

    t_lo = max(N + 1, int(np.ceil(trim * T_eff)))
    t_hi = min(T_eff - N - 1, int(np.floor((1.0 - trim) * T_eff)))
    candidates = np.arange(t_lo, t_hi + 1)

    if len(candidates) == 0:
        return float("nan"), -1, float("nan")

    # Degrees of freedom (joint test across all N equations)
    # df1 = N * N  (N regressors × N equations)
    # df2 = N * (T_eff - 2N)
    df1 = N * N
    df2 = N * (T_eff - 2 * N)
    if df2 <= 0:
        return float("nan"), -1, float("nan")

    def _scan(Y_use):
        beta_p, _, _, _ = np.linalg.lstsq(X, Y_use, rcond=None)
        rss_pool = float(((Y_use - X @ beta_p) ** 2).sum())

        best_F = -np.inf
        best_t = int(candidates[0])
        for t in candidates:
            b1, _, _, _ = np.linalg.lstsq(X[:t],  Y_use[:t],  rcond=None)
            b2, _, _, _ = np.linalg.lstsq(X[t:],  Y_use[t:],  rcond=None)
            rss1 = float(((Y_use[:t]  - X[:t]  @ b1) ** 2).sum())
            rss2 = float(((Y_use[t:]  - X[t:]  @ b2) ** 2).sum())
            denom = rss1 + rss2
            if denom <= 0:
                continue
            F = ((rss_pool - denom) / df1) / (denom / df2)
            if F > best_F:
                best_F = F
                best_t = int(t)
        return best_F, best_t

    supF, best_t = _scan(Y)

    # Residual bootstrap under H0
    beta_full, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resids = Y - X @ beta_full

    null_supFs = np.empty(n_boot)
    for b in range(n_boot):
        idx       = rng.integers(0, T_eff, size=T_eff)
        Y_boot    = X @ beta_full + resids[idx]
        null_supFs[b], _ = _scan(Y_boot)

    pval = float((null_supFs >= supF).mean())
    # best_t is in [t_lo, t_hi] ⊂ [0, T-2] relative to X/Y;
    # convert back to original ts index (break falls between ts[best_t] and ts[best_t+1])
    return float(supF), best_t + 1, pval


def _energy_stat(X: np.ndarray, Y: np.ndarray) -> float:
    """Energy distance between two multivariate samples (Chen & Zhang 2015).

    E(X, Y) = 2·mean(||x-y||) − mean(||x-x'||) − mean(||y-y'||)
    A large value means the two sets are drawn from different distributions.
    """
    def _mean_dist(A, B):
        diff = A[:, None, :] - B[None, :, :]          # (n,m,d)
        return float(np.sqrt((diff ** 2).sum(axis=2)).mean())

    return 2 * _mean_dist(X, Y) - _mean_dist(X, X) - _mean_dist(Y, Y)


def _edivisive_pvalue(ts: np.ndarray, seg_ids: np.ndarray,
                      n_perm: int = N_BOOTSTRAP,
                      rng: np.random.Generator = None) -> tuple:
    """Permutation p-value for the E-divisive change-point statistic.

    Tests H0: no distributional change between the two known segments.
    The observed energy statistic is compared to a null distribution obtained
    by randomly permuting rows of ts (exchangeability under H0).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X0  = ts[seg_ids == 0]
    X1  = ts[seg_ids == 1]
    obs = _energy_stat(X0, X1)

    n0, n1 = len(X0), len(X1)
    pool   = np.vstack([X0, X1])
    null_stats = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(n0 + n1)
        null_stats[i] = _energy_stat(pool[idx[:n0]], pool[idx[n0:]])

    pval = float((null_stats >= obs).mean())
    return float(obs), pval


# ---------------------------------------------------------------------------
# Sliding-window trajectories (non-overlapping)
# ---------------------------------------------------------------------------

def _sliding_windows(ts: np.ndarray, window: int):
    """Non-overlapping window centres and stats: std, lag-1 ACF, cross-corr dist."""
    T, N   = ts.shape
    starts = np.arange(0, T - window + 1, window)
    centres = starts + window // 2

    stds    = np.full((len(starts), N), np.nan)
    acf1    = np.full((len(starts), N), np.nan)
    xcorr_d = np.full(len(starts), np.nan)   # Frobenius dist from first window

    C_ref = None
    for i, s in enumerate(starts):
        seg = ts[s : s + window]
        stds[i]  = seg.std(axis=0)
        # lag-1 autocorrelation per node
        for n in range(N):
            c = np.corrcoef(seg[:-1, n], seg[1:, n])[0, 1]
            acf1[i, n] = c if np.isfinite(c) else 0.0
        # cross-correlation distance from first window
        C = np.corrcoef(seg.T)
        if C_ref is None:
            C_ref = C
        xcorr_d[i] = np.linalg.norm(C - C_ref, ord="fro")

    return centres, stds, acf1, xcorr_d


# ---------------------------------------------------------------------------
# Ruptures change-point detection
# ---------------------------------------------------------------------------

def _ruptures_breakpoints(ts: np.ndarray, pen=RUPTURES_PEN):
    """Detect change points with ruptures (PELT, RBF kernel — covariance-sensitive).

    Returns list of detected break indices (exclusive end of each segment).
    """
    if not HAS_RUPTURES:
        return []
    import ruptures as rpt
    model  = rpt.KernelCPD(kernel="rbf", min_size=20).fit(ts)
    if isinstance(pen, str):
        # BIC/AIC penalty: pen = log(T)*k for BIC
        T, N = ts.shape
        pen_val = np.log(T) * N if pen == "bic" else 2 * N
    else:
        pen_val = float(pen)
    breaks = model.predict(pen=pen_val)
    return breaks[:-1]   # exclude the sentinel T at the end


# ---------------------------------------------------------------------------
# Full test runner
# ---------------------------------------------------------------------------

def _run_tests(ts: np.ndarray, switch_times: np.ndarray, seg_ids: np.ndarray,
               dataset_label: str, dataset_description: str, cfg: dict):
    """Return (per_node_rows, summary_row)."""
    T, N    = ts.shape
    n_segs  = len(np.unique(seg_ids))
    rng     = np.random.default_rng(cfg.get("random_seed", 0))

    per_node_rows = []
    adf_pvals, kpss_pvals, cusum_pvals = [], [], []
    delta_means, delta_vars = [], []

    for n in range(N):
        col = ts[:, n]
        row = {
            "dataset_label": dataset_label,
            "description":   dataset_description,
            "generator":     cfg["generator"],
            "scenario":      cfg.get("scenario", cfg["generator"]),
            "T": T, "N": N,
            "switch_times":  ";".join(str(s) for s in switch_times.tolist()),
            "node": n,
        }

        if HAS_STATSMODELS:
            adf_stat, adf_p = _adf(col)
            row["adf_statistic"] = round(adf_stat, 5)
            row["adf_pvalue"]    = round(adf_p, 5)
            row["adf_call"]      = "stationary" if adf_p < ALPHA_ADF else "non-stationary"
            adf_pvals.append(adf_p)

            ks_stat, ks_p = _kpss_test(col)
            row["kpss_statistic"] = round(ks_stat, 5)
            row["kpss_pvalue"]    = round(ks_p, 5)
            row["kpss_call"]      = "non-stationary" if ks_p < ALPHA_KPSS else "stationary"
            kpss_pvals.append(ks_p)
        else:
            for k in ("adf_statistic", "adf_pvalue", "adf_call",
                      "kpss_statistic", "kpss_pvalue", "kpss_call"):
                row[k] = ""

        # CUSUM-of-squares (variance stability, bootstrap p-value)
        cusum_stat, cusum_p = _cusum_sq_pvalue(col, rng=rng)
        row["cusum_sq_stat"]  = round(cusum_stat, 5)
        row["cusum_sq_pvalue"] = round(cusum_p, 4)
        row["cusum_sq_call"]  = "unstable variance" if cusum_p < ALPHA_BREAK else "stable variance"
        cusum_pvals.append(cusum_p)

        if n_segs >= 2:
            s0, s1     = col[seg_ids == 0], col[seg_ids == 1]
            pooled_std = col.std()
            dm = abs(s1.mean() - s0.mean()) / (pooled_std + 1e-12)
            dv = (s1.std() - s0.std()) / (s0.std() + 1e-12)
            row["delta_mean"]    = round(dm, 5)
            row["delta_var_pct"] = round(dv * 100, 3)
            delta_means.append(dm)
            delta_vars.append(dv)
        else:
            row["delta_mean"] = row["delta_var_pct"] = ""

        per_node_rows.append(row)

    # ── Multivariate tests ────────────────────────────────────────────────
    if n_segs >= 2:
        s0, s1   = ts[seg_ids == 0], ts[seg_ids == 1]
        C0, C1   = np.corrcoef(s0.T), np.corrcoef(s1.T)
        delta_corr = float(np.linalg.norm(C1 - C0, ord="fro"))

        andrews_supF, andrews_break_t, andrews_p = _andrews_supF(ts, rng=rng)
        ediv_stat, ediv_p = _edivisive_pvalue(ts, seg_ids, rng=rng)
    else:
        delta_corr = andrews_supF = andrews_break_t = andrews_p = float("nan")
        ediv_stat = ediv_p = float("nan")

    # Ruptures detected breaks
    rpt_breaks = _ruptures_breakpoints(ts)

    # ── Overall calls ──────────────────────────────────────────────────────
    def _overall(pvals, flag_fn, n_total, pos_label, neg_label):
        n_flag = int(sum(flag_fn(p) for p in pvals))
        if n_flag == n_total: return pos_label
        if n_flag == 0:       return neg_label
        return f"mixed ({n_flag}/{n_total} {pos_label})"

    adf_overall  = (_overall(adf_pvals,   lambda p: p < ALPHA_ADF,  N, "stationary",       "non-stationary")
                    if adf_pvals  else "")
    kpss_overall = (_overall(kpss_pvals,  lambda p: p >= ALPHA_KPSS, N, "stationary",       "non-stationary")
                    if kpss_pvals else "")
    cusum_overall = _overall(cusum_pvals, lambda p: p >= ALPHA_BREAK, N, "stable variance", "unstable variance")

    summary_row = {
        "dataset_label": dataset_label,
        "description":   dataset_description,
        "generator":     cfg["generator"],
        "scenario":      cfg.get("scenario", cfg["generator"]),
        "T": T, "N": N,
        "switch_times":  ";".join(str(s) for s in switch_times.tolist()),
        # ADF
        "adf_pvalue_mean":      round(float(np.mean(adf_pvals)),  5) if adf_pvals  else "",
        "adf_pvalue_max":       round(float(np.max(adf_pvals)),   5) if adf_pvals  else "",
        "adf_overall_call":     adf_overall,
        # KPSS
        "kpss_pvalue_mean":     round(float(np.mean(kpss_pvals)), 5) if kpss_pvals else "",
        "kpss_pvalue_min":      round(float(np.min(kpss_pvals)),  5) if kpss_pvals else "",
        "kpss_overall_call":    kpss_overall,
        # CUSUM-of-squares
        "cusum_sq_stat_mean":   round(float(np.mean(cusum_pvals)), 4),
        "cusum_sq_overall_call": cusum_overall,
        # Segment comparison
        "delta_mean_mean":      round(float(np.mean(delta_means)), 5) if delta_means else "",
        "delta_mean_max":       round(float(np.max(delta_means)),  5) if delta_means else "",
        "delta_var_pct_mean":   round(float(np.mean(delta_vars)) * 100, 3) if delta_vars else "",
        "delta_var_pct_max":    round(float(np.max(np.abs(delta_vars))) * 100, 3) if delta_vars else "",
        # Multivariate
        "delta_corr":             round(delta_corr, 5) if np.isfinite(delta_corr) else "",
        "andrews_supF":           round(andrews_supF, 4) if np.isfinite(andrews_supF) else "",
        "andrews_break_t":        int(andrews_break_t) if np.isfinite(andrews_supF) else "",
        "andrews_pvalue":         round(andrews_p, 4) if np.isfinite(andrews_p) else "",
        "andrews_call":           ("break detected" if np.isfinite(andrews_p) and andrews_p < ALPHA_BREAK
                                   else "no break" if np.isfinite(andrews_p) else ""),
        "edivisive_stat":         round(ediv_stat, 5) if np.isfinite(ediv_stat) else "",
        "edivisive_pvalue":       round(ediv_p, 4) if np.isfinite(ediv_p) else "",
        "edivisive_call":         ("change detected" if np.isfinite(ediv_p) and ediv_p < ALPHA_BREAK
                                   else "no change" if np.isfinite(ediv_p) else ""),
        # Ruptures
        "ruptures_breaks":      ";".join(str(b) for b in rpt_breaks) if rpt_breaks else "none",
        "ruptures_n_breaks":    len(rpt_breaks),
    }

    return per_node_rows, summary_row


# ---------------------------------------------------------------------------
# CSV field lists
# ---------------------------------------------------------------------------

PER_NODE_FIELDS = [
    "dataset_label", "description", "generator", "scenario",
    "T", "N", "switch_times", "node",
    "adf_statistic", "adf_pvalue", "adf_call",
    "kpss_statistic", "kpss_pvalue", "kpss_call",
    "cusum_sq_stat", "cusum_sq_pvalue", "cusum_sq_call",
    "delta_mean", "delta_var_pct",
]

SUMMARY_FIELDS = [
    "dataset_label", "description", "generator", "scenario",
    "T", "N", "switch_times",
    "adf_pvalue_mean", "adf_pvalue_max", "adf_overall_call",
    "kpss_pvalue_mean", "kpss_pvalue_min", "kpss_overall_call",
    "cusum_sq_stat_mean", "cusum_sq_overall_call",
    "delta_mean_mean", "delta_mean_max",
    "delta_var_pct_mean", "delta_var_pct_max",
    "delta_corr",
    "andrews_supF", "andrews_break_t", "andrews_pvalue", "andrews_call",
    "edivisive_stat", "edivisive_pvalue", "edivisive_call",
    "ruptures_breaks", "ruptures_n_breaks",
]


def _write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Sliding-window plot
# ---------------------------------------------------------------------------

def _save_sliding_plot(ts, switch_times, label, save_path):
    import matplotlib.pyplot as plt
    T, N = ts.shape
    centres, stds, acf1, xcorr_d = _sliding_windows(ts, WINDOW_SIZE)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"hspace": 0.08, "height_ratios": [2, 1, 1, 1]})

    # Raw time series (first node only for readability, others faded)
    for n in range(N):
        axes[0].plot(np.arange(T), ts[:, n], lw=0.5, alpha=0.4 if n > 0 else 0.9)
    for st in switch_times:
        for ax in axes:
            ax.axvline(st, color="black", linestyle="--", lw=1.0)
    axes[0].set_ylabel("time series", fontsize=8)

    # Rolling std
    for n in range(N):
        axes[1].plot(centres, stds[:, n], lw=1.0, label=f"n{n}")
    axes[1].set_ylabel(f"std (w={WINDOW_SIZE})", fontsize=8)
    axes[1].legend(fontsize=6, ncol=min(N, 6), loc="upper right")

    # Lag-1 autocorrelation (mean across nodes)
    axes[2].plot(centres, acf1.mean(axis=1), lw=1.2, color="darkorange")
    axes[2].fill_between(centres, acf1.min(axis=1), acf1.max(axis=1),
                          alpha=0.2, color="darkorange")
    axes[2].set_ylabel("lag-1 ACF\n(mean±range)", fontsize=8)
    axes[2].axhline(0, color="grey", lw=0.5)

    # Cross-correlation distance from first window
    axes[3].plot(centres, xcorr_d, lw=1.2, color="mediumpurple")
    axes[3].set_ylabel("ΔCorr from\nfirst window", fontsize=8)
    axes[3].set_xlabel("Time step (window centre)", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=7)

    fig.suptitle(label, fontsize=9, y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(r):
    print(f"  ADF          : {r['adf_overall_call']}  (p̄={r['adf_pvalue_mean']})")
    print(f"  KPSS         : {r['kpss_overall_call']}  (p̄={r['kpss_pvalue_mean']})")
    print(f"  CUSUM-sq     : {r['cusum_sq_overall_call']}")
    print(f"  Andrews sup-F: {r['andrews_call']}  (supF={r['andrews_supF']}, p={r['andrews_pvalue']}, break_t={r['andrews_break_t']})")
    print(f"  E-divisive   : {r['edivisive_call']}  (stat={r['edivisive_stat']}, p={r['edivisive_pvalue']})")
    print(f"  ΔCorr        : {r['delta_corr']}")
    print(f"  Ruptures     : {r['ruptures_n_breaks']} break(s) at {r['ruptures_breaks']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_per_node  = []
    all_summaries = []

    for idx, cfg in enumerate(DATASETS):
        label = cfg["label"]
        desc  = cfg.get("description", label)
        print(f"\n[{idx + 1}/{len(DATASETS)}] {desc}")

        ts, switch_times, seg_ids = _generate(cfg)
        print(f"  T={len(ts)}, N={ts.shape[1]}, switch_times={switch_times.tolist()}")

        per_node_rows, summary_row = _run_tests(
            ts, switch_times, seg_ids, label, desc, cfg
        )
        all_per_node.extend(per_node_rows)
        all_summaries.append(summary_row)
        _print_summary(summary_row)

        safe = label[:50]
        _save_sliding_plot(ts, switch_times, desc,
                           os.path.join(OUTPUT_DIR, f"{idx:02d}_{safe}_sliding.png"))

    print(f"\nWriting results to {OUTPUT_DIR}/")
    _write_csv(os.path.join(OUTPUT_DIR, "per_node.csv"),  PER_NODE_FIELDS,  all_per_node)
    _write_csv(os.path.join(OUTPUT_DIR, "summary.csv"),   SUMMARY_FIELDS,   all_summaries)
    print("Done.")


if __name__ == "__main__":
    main()
