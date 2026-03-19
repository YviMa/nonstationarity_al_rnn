"""Optional temporal feature module for AL-RNN experiments.

Temporal features are used as *external inputs* to the latent dynamics via
the factor loading matrix C, i.e.

    z_{t+1} = A z_t + phi(z_t) W^T + h  +  C u_t

They are never part of the observation vector and are never predicted.
"""

import math
import numpy as np
import torch
import torch.nn as nn


class TemporalInput(nn.Module):
    """Produce a temporal feature vector u_t for use as an external input.

    Parameters
    ----------
    mode : str
        'none'    – disabled; output_dim == 0, compute_features returns None.
        'linear'  – single scalar t / (T-1) (normalized time).
        'fourier' – sin/cos pairs at n_fourier_pairs frequencies.
    n_fourier_pairs : int
        Number of frequency pairs used when mode='fourier'.
    """

    def __init__(self, mode: str = "none", n_fourier_pairs: int = 4):
        super().__init__()
        if mode not in ("none", "linear", "fourier"):
            raise ValueError(f"Unknown temporal input mode: {mode!r}")
        self.mode = mode
        self.n_fourier_pairs = n_fourier_pairs

    @property
    def output_dim(self) -> int:
        """Dimensionality of u_t (d_input for the AL-RNN)."""
        if self.mode == "none":
            return 0
        if self.mode == "linear":
            return 1
        return 2 * self.n_fourier_pairs

    def forward(self, t: int, T: int) -> torch.Tensor:
        """Compute the external-input feature vector for time step t.

        Parameters
        ----------
        t : int
            Current time index (0-based).
        T : int
            Total number of time steps in the sequence.

        Returns
        -------
        u_t : Tensor of shape (output_dim,)
        """
        if self.mode == "none":
            return torch.zeros(0, dtype=torch.float32)
        if self.mode == "linear":
            return torch.tensor([t / max(T - 1, 1)], dtype=torch.float32)
        # fourier
        freqs = torch.arange(1, self.n_fourier_pairs + 1, dtype=torch.float32)
        phase = 2.0 * math.pi * freqs * t / max(T - 1, 1)
        return torch.cat([torch.sin(phase), torch.cos(phase)])

    def compute_features(self, T: int) -> np.ndarray | None:
        """Pre-compute the full external-input matrix for a sequence of length T.

        Parameters
        ----------
        T : int
            Number of time steps.

        Returns
        -------
        u : ndarray of shape (T, output_dim), or None when mode='none'.
        """
        if self.output_dim == 0:
            return None
        return np.stack([self.forward(t, T).numpy() for t in range(T)])
