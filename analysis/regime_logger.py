"""Bitcode / regime logger for trained AL-RNN models."""

import numpy as np
import torch


def _pack_bits(arr):
    """Pack a boolean/int array into a single integer (MSB first)."""
    val = 0
    for b in arr:
        val = (val << 1) | int(b)
    return val


def log_bitcodes(
    model,
    observations: np.ndarray,
    forcing_mode: str = 'gtf',
    alpha: float = 1.0,
) -> np.ndarray:
    """Record the active PWL regime (bitcode) at each time step.

    For GTF the bitcode at step t is the sign pattern of
    obs_to_latent(x_t)[-P:] — the observation projected into latent space.

    For ITF the free latent units (last M-N of z) are never determined by
    the observation alone; they carry autonomous state.  Here the full ITF
    forward pass is run with the given alpha so the free units evolve
    correctly, and the sign pattern of h[-P:] at each step gives the bitcode.

    Each P-bit binary vector is packed into a single integer (MSB first).
    When P == 0 the model is linear and all bitcodes are 0.

    Parameters
    ----------
    model : AL_RNN
        Trained model in eval mode.
    observations : ndarray of shape (T, N)
        Raw observation sequence.
    forcing_mode : str
        'gtf' or 'itf' — must match the mode used during training.
    alpha : float
        Teacher-forcing strength for the ITF forward pass (default 1.0).

    Returns
    -------
    bitcodes : ndarray of int64, shape (T,)
    """
    T = len(observations)
    bitcodes = np.zeros(T, dtype=np.int64)

    if model.P == 0:
        return bitcodes

    model.eval()
    with torch.no_grad():
        if forcing_mode == 'itf':
            from alrnn.training import predict_sequence_using_itf
            x_batch = torch.tensor(observations, dtype=torch.float32).unsqueeze(0)
            _, pwl_states = predict_sequence_using_itf(
                model, x_batch, alpha=alpha, return_pwl_states=True,
            )  # pwl_states: (1, T, P)
            pwl_np = pwl_states[0].cpu().numpy()  # (T, P)
            for t in range(T):
                bitcodes[t] = _pack_bits((pwl_np[t] >= 0).astype(int))
        else:
            x = torch.tensor(observations, dtype=torch.float32)
            for t in range(T):
                z = model.obs_to_latent(x[t])
                bitcodes[t] = _pack_bits((z[-model.P:] >= 0).int().cpu().numpy())

    return bitcodes


def bitcodes_from_pwl_states(pwl_states: np.ndarray) -> np.ndarray:
    """Pack the sign of pre-activation PWL states into integer bitcodes.

    Parameters
    ----------
    pwl_states : ndarray of shape (T, P)
        Pre-activation values of the P ReLU units at every step.

    Returns
    -------
    bitcodes : ndarray of int64, shape (T,)
        Bit i of code t is 1 iff pwl_states[t, i] >= 0.
    """
    T, P = pwl_states.shape
    bitcodes = np.zeros(T, dtype=np.int64)
    for t in range(T):
        bitcodes[t] = _pack_bits((pwl_states[t] >= 0).astype(int))
    return bitcodes


def count_unique_regimes(bitcodes: np.ndarray) -> int:
    """Return the number of distinct bitcodes in the sequence.

    Parameters
    ----------
    bitcodes : ndarray of int, shape (T,)

    Returns
    -------
    n_unique : int
    """
    return int(np.unique(bitcodes).size)


def regime_change_times(bitcodes: np.ndarray) -> np.ndarray:
    """Return time indices where the bitcode changes.

    A change at index t means bitcodes[t] != bitcodes[t - 1].

    Parameters
    ----------
    bitcodes : ndarray of int, shape (T,)

    Returns
    -------
    change_times : ndarray of int
    """
    if len(bitcodes) < 2:
        return np.array([], dtype=int)
    return (np.where(np.diff(bitcodes) != 0)[0] + 1).astype(int)
