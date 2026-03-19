import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AL_RNN(nn.Module):
    """Almost-Linear Recurrent Neural Network.

    Latent dynamics:
        z_{t+1} = A * z_t + phi(z_t) @ W^T + h  [ + C @ u_t ]

    where phi applies ReLU to the last P components of z.
    The optional term C @ u_t integrates an external input u_t of dimension
    d_input via a learned factor loading matrix C (shape M × d_input).
    u_t is never reconstructed or included in the prediction loss.

    When P=0 and M=N, this reduces to a linear VAR(1) model.
    When P>0, the last P units introduce piecewise-linear nonlinearity,
    and a readout matrix B maps between observation space (N) and
    latent space (M).

    Parameters
    ----------
    M : int
        Total number of latent units.
    P : int
        Number of piecewise-linear (ReLU) units (last P of M).
    N : int
        Number of observation/readout units (first N of M).
    d_input : int
        Dimensionality of optional external input u_t (0 = disabled).
    """

    def __init__(self, M, P, N, d_input=0, pwl_timescale=None):
        """
        Parameters
        ----------
        pwl_timescale : float or None
            If set, the last P entries of A are initialised to this value
            (e.g. 0.99) so the PWL units start with a slow intrinsic time
            scale, biasing them toward gradual drift rather than rapid sign
            changes.  Has no effect when P == 0.
        """
        super().__init__()
        self.M = M
        self.P = P
        self.N = N
        self.d_input = d_input

        self.A, self.W, self.h = self._initialize_AWh(pwl_timescale)
        self.B = self._init_uniform((N, M))
        self.C = (nn.Parameter(torch.randn(M, d_input) * 0.01)
                  if d_input > 0 else None)

    def forward(self, z, u=None):
        """One-step latent dynamics.

        Parameters
        ----------
        z : Tensor of shape (batch, M) or (M,)
        u : Tensor of shape (batch, d_input) or (d_input,), optional
            External input for this time step.

        Returns
        -------
        z_next : Tensor of same shape as z
        """
        z_unactivated = torch.clone(z)
        if self.P > 0:
            z[..., -self.P:] = F.relu(z[..., -self.P:])
        out = self.A * z_unactivated + z @ self.W.t() + self.h
        if u is not None and self.C is not None:
            out = out + u @ self.C.t()
        return out

    def obs_to_latent(self, x):
        """Map observations to latent space.

        M == N: identity mapping (latent IS observation).
        M >= N: x @ B to project, then force first N components to x.
        M <  N: x @ B to project down (B acts as encoder).

        Parameters
        ----------
        x : Tensor of shape (batch, N) or (N,)

        Returns
        -------
        z : Tensor of shape (batch, M) or (M,)
        """
        if self.M == self.N:
            return x.clone()

        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        z = x @ self.B  # (batch, N) @ (N, M) -> (batch, M)

        if self.M > self.N:
            # First N latent components = observations
            z = z.clone()
            z[:, :self.N] = x

        if squeeze:
            z = z.squeeze(0)
        return z

    def latent_to_obs(self, z):
        """Map latent state to observations.

        M >= N: first N components of z.
        M <  N: z @ B^T to project up via readout matrix.
        """
        if self.M >= self.N:
            return z[..., :self.N]
        # M < N: B is (N, M), B^T is (M, N), z @ B^T -> (..., N)
        return z @ self.B.t()

    def get_coefficient_matrix(self):
        """Extract effective N x N coefficient matrix for the linear case.

        M == N, P == 0:
            Effective matrix is (diag(A) + W)^T.
        M != N, P == 0:
            Full pipeline obs -> latent -> dynamics -> obs gives
            effective matrix B @ (diag(A) + W) @ B^T  (in column-vector
            convention for prediction = matrix @ state).
        P > 0:
            Returns None (nonlinear).

        Returns
        -------
        coef_matrix : ndarray of shape (N, N), or None if nonlinear.
        """
        if self.P > 0:
            return None

        with torch.no_grad():
            Aw = torch.diag(self.A) + self.W  # (M, M)

            if self.M == self.N:
                return Aw.cpu().numpy().T

            # M != N: obs -> latent via B, dynamics via Aw, latent -> obs via B^T
            # In row-vector terms: x_next = (x @ B) @ Aw^T @ B^T
            # For column-vector prediction = matrix @ state:
            # Transpose for col-vec: x_next = B @ Aw @ B^T @ x
            B = self.B  # (N, M)
            effective = B @ Aw @ B.t()  # (N, M) @ (M, M) @ (M, N) = (N, N)
            return effective.cpu().numpy()

    # -- Initialization (matches AL-RNN tutorial notebook) --

    def _initialize_AWh(self, pwl_timescale=None):
        A = nn.Parameter(torch.diagonal(
            self._normalized_positive_definite(self.M), 0))
        if pwl_timescale is not None and self.P > 0:
            with torch.no_grad():
                A[-self.P:] = pwl_timescale
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def _normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = R.T @ R / M + np.eye(M, dtype=np.float32)
        eigenvalues = np.linalg.eigvals(K)
        lambda_max = np.max(np.abs(eigenvalues))
        return torch.tensor(K / lambda_max).float()

    def _init_uniform(self, shape):
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)
