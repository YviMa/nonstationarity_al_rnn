"""PWL bitcode-forced loss scan for trained AL-RNN models.

Background
----------
An AL-RNN with P piecewise-linear (PWL) units defines 2**P distinct linear
dynamical systems (DSs), one per bitcode pattern.  The bitcode at time t is

    bc[t] = (h_t[-P:] > 0).int()

where h_t[-P:] are the last-P components of the free latent state that get
passed into model.forward as the pre-activation input to ReLU.  Inside the
forward pass::

    relu_out[-P:] = relu(h_t[-P:])   →  0         if bc[i] = 0
                                         h_t[-P+i]  if bc[i] = 1

The EFFECTIVE CONNECTIVITY of the network therefore differs between bitcodes:
when bc[i] = 0 the i-th column of W is gated out (relu_out = 0 so W[:, i]
contributes nothing); when bc[i] = 1 the full column is active.

The key diagnostic question
---------------------------
For a 2-regime time series with a true switch at time t_true, the model
should learn two dominant bitcodes bc_pre (regime 1) and bc_post (regime 2).
We can test: at what candidate switch time t* does forcing the transition
bc_pre → bc_post minimise the reconstruction MSE?  If the model has correctly
learned regime-linked DSs, the loss scan should exhibit a minimum near
t* = t_true, providing an explicit loss-landscape explanation for *why* the
model can sense nonstationarities.

Forcing is done by clamping the sign of h[-P:] — magnitude is preserved, only
the sign is controlled — which selects the appropriate DS at each step while
keeping the hidden-state dynamics as intact as possible.

API
---
::

    from analysis.pwl_switch_scan import bitcode_forced_scan, plot_scan_result

    scan = bitcode_forced_scan(model, x_full, u_full,
                               alpha=cfg["end_alpha"],
                               n_interleave=cfg["n_interleave"])
    plot_scan_result(scan, switch_times_gt=switch_times,
                     save_path="var_results/pwl_scan.png")
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# Natural run — bitcodes and per-step MSE
# ---------------------------------------------------------------------------

def natural_run(
    model,
    x: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    n_interleave: int = 1,
) -> dict:
    """Run ITF normally; return predictions, bitcode trajectory, and per-step MSE.

    Parameters
    ----------
    model : AL_RNN with P > 0
    x : (1, T, N) — standardised full sequence
    u : (1, T, d_input) or None
    alpha, n_interleave : forcing parameters

    Returns
    -------
    dict with keys:
        'predictions'    : (T-1, N) float ndarray — Z[t] ≈ x[t+1]
        'pwl_preact'     : (T, P)   float ndarray — h[-P:] at each step
        'bitcodes'       : (T, P)   int   ndarray — sign pattern (0/1)
        'per_step_mse'   : (T-1,)   float ndarray — MSE(Z[t], x[t+1])
        'total_mse'      : float
    """
    N, P = model.N, model.P
    x_ = x.permute(1, 0, 2)          # (T, 1, N)
    u_ = u.permute(1, 0, 2) if u is not None else None
    T = x_.shape[0]

    pred_list: list = []
    preact_list: list = []

    with torch.no_grad():
        h = model.obs_to_latent(x_[0])[:, N:].detach().clone()
        preact_list.append(h[:, -P:].cpu().numpy())

        z_out = model(torch.cat([x_[0], h], dim=1),
                      u=u_[0] if u_ is not None else None)
        pred_list.append(z_out[:, :N].cpu().numpy())
        h = z_out[:, N:]

        for t in range(1, T):
            preact_list.append(h[:, -P:].cpu().numpy())
            if t % n_interleave == 0:
                obs = alpha * x_[t] + (1.0 - alpha) * z_out[:, :N]
            else:
                obs = z_out[:, :N]
            z_out = model(torch.cat([obs, h], dim=1),
                          u=u_[t] if u_ is not None else None)
            pred_list.append(z_out[:, :N].cpu().numpy())
            h = z_out[:, N:]

    preds    = np.concatenate(pred_list[:-1], axis=0)    # (T-1, N)
    preact   = np.concatenate(preact_list,    axis=0)    # (T, P)
    target   = x_[1:, 0, :].cpu().numpy()               # (T-1, N)

    per_step = ((preds - target) ** 2).mean(axis=1)     # (T-1,)
    bitcodes = (preact > 0).astype(int)                  # (T, P)

    return {
        "predictions":   preds,
        "pwl_preact":    preact,
        "bitcodes":      bitcodes,
        "per_step_mse":  per_step,
        "total_mse":     float(per_step.mean()),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def auto_detect_bitcodes(bitcodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer bc_pre and bc_post by majority vote in each half of the series.

    Parameters
    ----------
    bitcodes : (T, P) int array — per-step bitcode patterns

    Returns
    -------
    bc_pre  : (P,) int array — dominant bitcode in first half
    bc_post : (P,) int array — dominant bitcode in second half
    """
    T = len(bitcodes)
    mid = T // 2
    bc_pre  = (bitcodes[:mid].mean(axis=0) >= 0.5).astype(int)
    bc_post = (bitcodes[mid:].mean(axis=0) >= 0.5).astype(int)
    return bc_pre, bc_post


def _forced_run(model, x_, u_, alpha, n_interleave, bc_seq: np.ndarray,
                units: list) -> float:
    """ITF run with per-step forced bitcode; return total MSE.

    Parameters
    ----------
    bc_seq : (T, len(units)) int array — desired bitcode at each step
    units  : which of the P PWL unit indices to force (others evolve freely)
    """
    N, P = model.N, model.P
    T = x_.shape[0]
    target = x_[1:, 0, :].cpu()   # (T-1, N)

    h = model.obs_to_latent(x_[0])[:, N:].detach().clone()

    # Force bitcode at t=0 (determines DS at first forward call)
    h = _force_sign(h, bc_seq[0], P, units)

    z_out = model(torch.cat([x_[0], h], dim=1),
                  u=u_[0] if u_ is not None else None)
    pred_list = [z_out[:, :N]]
    h = z_out[:, N:]

    for t in range(1, T):
        # Force the sign of selected h[-P:] units to bc_seq[t]
        h = _force_sign(h, bc_seq[t], P, units)

        if t % n_interleave == 0:
            obs = alpha * x_[t] + (1.0 - alpha) * z_out[:, :N]
        else:
            obs = z_out[:, :N]

        z_out = model(torch.cat([obs, h], dim=1),
                      u=u_[t] if u_ is not None else None)
        pred_list.append(z_out[:, :N])
        h = z_out[:, N:]

    preds = torch.cat(pred_list[:-1], dim=0)   # (T-1, N)
    return float(((preds - target) ** 2).mean().item())


def _force_sign(
    h: torch.Tensor,
    bc: np.ndarray,
    P: int,
    units: Optional[list] = None,
) -> torch.Tensor:
    """Clamp sign of selected PWL units in h[:, -P:], preserving magnitude.

    Parameters
    ----------
    bc    : int array of length len(units) — desired sign (0=negative, 1=positive)
    units : which of the P PWL unit indices (0-based within [-P:]) to force;
            None means force all P units.  Unselected units are left untouched.
    """
    if P == 0:
        return h
    if units is None:
        units = list(range(P))
    if not units:
        return h
    h = h.clone()
    for idx, i in enumerate(units):
        mag = h[:, -P + i].abs()
        if bc[idx] == 0:
            h[:, -P + i] = -mag     # force negative  →  ReLU will output 0
        else:
            h[:, -P + i] = mag      # force positive  →  ReLU will output mag
    return h


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def bitcode_forced_scan(
    model,
    x: torch.Tensor,
    u: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    n_interleave: int = 1,
    trim: float = 0.05,
    step: int = 5,
    pwl_units: Optional[list] = None,
    bc_pre: Optional[np.ndarray] = None,
    bc_post: Optional[np.ndarray] = None,
) -> dict:
    """Sweep over candidate switch times with forced bitcode transitions.

    For each candidate t*, runs ITF with:
        h[-P:] sign forced to bc_pre  for t <  t*  (DS of regime 1)
        h[-P:] sign forced to bc_post for t >= t*  (DS of regime 2)

    Only the PWL units listed in ``pwl_units`` are forced; the remaining units
    evolve freely.  This allows isolating the one or two slow "switching" units
    when P is large (e.g. P=4) without having to enumerate all 2^P bitcodes.

    If bc_pre / bc_post are None they are auto-detected from the natural run
    by majority vote in the first / second half of the series, restricted to
    the selected units.

    Parameters
    ----------
    model : AL_RNN with P > 0
    x : Tensor (1, T, N)
    u : Tensor (1, T, d_input) or None
    alpha, n_interleave : match those used at evaluation time
    trim : skip first/last ``trim`` fraction of T as candidates
    step : evaluate every ``step``-th candidate time (default 5)
    pwl_units : list of int or None
        Indices into the P PWL units (0-based, so 0 = h[M-P], P-1 = h[M-1]).
        None means use all P units.
    bc_pre  : int array of length len(pwl_units) or None — bitcode for t < t*
    bc_post : int array of length len(pwl_units) or None — bitcode for t >= t*

    Returns
    -------
    dict with keys:
        'switch_times'     : (n_candidates,) int array
        'scan_losses'      : (n_candidates,) float array
        'loss_natural'     : float — unforced ITF MSE
        'loss_always_pre'  : float — bc_pre forced throughout
        'loss_always_post' : float — bc_post forced throughout
        'bc_pre'           : int array used for pre-switch forcing
        'bc_post'          : int array used for post-switch forcing
        'pwl_units'        : list of int — units that were forced
        'natural_result'   : full output of natural_run()
    """
    assert model.P > 0, "bitcode_forced_scan requires model.P > 0."

    P = model.P
    units: list = list(range(P)) if pwl_units is None else list(pwl_units)
    assert all(0 <= i < P for i in units), \
        f"pwl_units entries must be in [0, P-1]=[0, {P-1}]"

    x_ = x.permute(1, 0, 2)          # (T, 1, N)
    u_ = u.permute(1, 0, 2) if u is not None else None
    T = x_.shape[0]

    # --- natural run to get bitcodes for auto-detection ----------------------
    nat = natural_run(model, x, u=u, alpha=alpha, n_interleave=n_interleave)
    # Restrict bitcodes to the selected units for majority-vote detection
    bitcodes_sel = nat["bitcodes"][:, units]   # (T, len(units))
    if bc_pre is None or bc_post is None:
        _bc_pre, _bc_post = auto_detect_bitcodes(bitcodes_sel)
        if bc_pre  is None: bc_pre  = _bc_pre
        if bc_post is None: bc_post = _bc_post

    bc_pre  = np.asarray(bc_pre,  dtype=int)
    bc_post = np.asarray(bc_post, dtype=int)

    print(f"  pwl_units={units}  bc_pre={bc_pre.tolist()}  bc_post={bc_post.tolist()}")
    if np.array_equal(bc_pre, bc_post):
        print("  WARNING: bc_pre == bc_post — model may not have learned "
              "regime-linked bitcodes for the selected units; scan will be flat.")

    # --- candidate switch times -----------------------------------------------
    t_lo = max(1, int(np.ceil(trim * T)))
    t_hi = min(T - 1, int(np.floor((1 - trim) * T)))
    candidates = np.arange(t_lo, t_hi + 1, step, dtype=int)

    # --- reference baselines --------------------------------------------------
    bc_all_pre  = np.tile(bc_pre,  (T, 1))   # (T, len(units))
    bc_all_post = np.tile(bc_post, (T, 1))   # (T, len(units))

    with torch.no_grad():
        loss_always_pre  = _forced_run(model, x_, u_, alpha, n_interleave, bc_all_pre,  units)
        loss_always_post = _forced_run(model, x_, u_, alpha, n_interleave, bc_all_post, units)

        scan_losses = np.empty(len(candidates))
        for i, t_star in enumerate(candidates):
            bc_seq = np.concatenate([
                np.tile(bc_pre,  (int(t_star), 1)),
                np.tile(bc_post, (T - int(t_star), 1)),
            ], axis=0)   # (T, len(units))
            scan_losses[i] = _forced_run(model, x_, u_, alpha, n_interleave, bc_seq, units)

    return {
        "switch_times":     candidates,
        "scan_losses":      scan_losses,
        "loss_natural":     nat["total_mse"],
        "loss_always_pre":  loss_always_pre,
        "loss_always_post": loss_always_post,
        "bc_pre":           bc_pre,
        "bc_post":          bc_post,
        "pwl_units":        units,
        "natural_result":   nat,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_scan_result(
    scan: dict,
    switch_times_gt: Optional[np.ndarray] = None,
    title: str = "PWL bitcode-forced loss scan",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot the loss scan together with reference baselines.

    Parameters
    ----------
    scan : output of :func:`bitcode_forced_scan`
    switch_times_gt : ground-truth regime-change indices (vertical lines)
    title, save_path, show : display options
    """
    t_star  = scan["switch_times"]
    losses  = scan["scan_losses"]
    l_nat   = scan["loss_natural"]
    l_pre   = scan["loss_always_pre"]
    l_post  = scan["loss_always_post"]

    best_idx  = int(np.argmin(losses))
    best_t    = t_star[best_idx]
    best_loss = losses[best_idx]

    fig, ax = plt.subplots(figsize=(10, 4))

    units_str = str(scan.get("pwl_units", "all"))
    ax.plot(t_star, losses, color="steelblue", lw=1.5,
            label=f"forced-transition loss  L(t*)  [units {units_str}]")
    ax.axhline(l_nat,  color="gray",     lw=1.2, ls="-.",
               label=f"natural (no forcing)  L={l_nat:.5f}")
    ax.axhline(l_pre,  color="seagreen", lw=1.2, ls="--",
               label=f"always bc_pre={scan['bc_pre'].tolist()}  L={l_pre:.5f}")
    ax.axhline(l_post, color="tomato",   lw=1.2, ls="--",
               label=f"always bc_post={scan['bc_post'].tolist()}  L={l_post:.5f}")
    ax.axvline(best_t, color="steelblue", lw=1.0, ls=":", alpha=0.8,
               label=f"minimum at t*={best_t}  (L={best_loss:.5f})")
    ax.scatter([best_t], [best_loss], color="steelblue", zorder=5, s=50)

    if switch_times_gt is not None:
        for i, gt in enumerate(np.atleast_1d(switch_times_gt)):
            ax.axvline(gt, color="darkorange", lw=1.5, ls="-", alpha=0.85,
                       label="ground-truth switch" if i == 0 else None)

    ax.set_xlabel("Forced switch time  t*")
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Effective-connectivity helper (for notebook analysis)
# ---------------------------------------------------------------------------

def effective_w_matrices(model) -> dict:
    """Extract the two effective connectivity matrices for each DS region.

    For bitcode bit i = 0: column W[:, M-P+i] is zeroed (ReLU output = 0).
    For bitcode bit i = 1: column W[:, M-P+i] is active.

    Returns full W (all bits = 1) and W_inactive (all bits = 0) as numpy arrays,
    plus the incremental 'switch matrix' W_delta = W_full - W_inactive.
    """
    with torch.no_grad():
        W = model.W.cpu().numpy().copy()   # (M, M)
    M, P = model.M, model.P

    W_inactive = W.copy()
    W_inactive[:, M - P:] = 0.0   # zero columns for all PWL units off

    W_full = W.copy()              # all PWL units on (unchanged)

    return {
        "W_full":     W_full,
        "W_inactive": W_inactive,
        "W_delta":    W_full - W_inactive,   # non-zero only in last P columns
    }
