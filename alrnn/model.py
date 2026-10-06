import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AL_RNN(nn.Module):
    """Almost-Linear Recurrent Neural Network.

    Latent dynamics:
        z_{t+1} = A * z_t + phi(z_t) @ W^T + h  [ + temporal_input(u_t) ]

    where phi applies ReLU to the last P components of z.
    The optional temporal input term integrates an external input u_t of
    dimension d_input via a learned projection matrix; the projection target
    is controlled by ``temporal_input_target``:

        'all'           — current behaviour: C (M × d_input) maps u_t to all
                          latent units.  Equivalent to  +  C @ u_t.
        'pwl_only'      — C_pwl (P × d_input) maps u_t only onto the P
                          piecewise-linear unit pre-activations; linear units
                          receive no temporal signal.
        'monotone_gate' — same as 'pwl_only' but the temporal projection is
                          scaled by a learned sigmoid gate:
                              gate_t = σ(α · (u_t[0] − τ))
                          where α and τ are scalar nn.Parameters initialised
                          at α=5, τ=0.5.  Biases the model toward a single
                          clean transition at a learned midpoint.
                          Requires mode='linear' temporal features so that
                          u_t[0] = t/(T−1) ∈ [0,1].

    When P=0 and M=N, this reduces to a linear VAR(1) model.
    When P>0, the last P units introduce piecewise-linear nonlinearity,
    and a readout matrix B maps between observation space (N) and
    latent space (M).

    Parameters
    ----------
    M : int
        Total number of latent units.
    P : int
        Total number of piecewise-linear (ReLU) units (last P of M).
    N : int
        Number of observation/readout units (first N of M).
    d_input : int
        Dimensionality of optional external input u_t (0 = disabled).
    P_switch : int or None
        Number of *slow-switching* PWL units among the P units — specifically
        the last P_switch of the P units (positions M-P_switch … M-1).
        These receive temporal input and all regularization (pwl_timescale
        init, timescale_lambda, incoming_lambda, pwl_state_lambda,
        switch_lambda).
        The remaining P - P_switch units are *free* PWL units that can switch
        rapidly to support local nonlinear approximation (e.g. Lorenz-63
        attractor geometry) and receive none of the above.
        None (default) → P_switch = P, i.e. all units are slow-switching,
        which is fully backward-compatible with the original behaviour.
    temporal_input_target : str
        One of 'all', 'pwl_only', 'monotone_gate' (see class docstring).
    """

    _VALID_TARGETS = ('all', 'pwl_only', 'monotone_gate')

    def __init__(self, M, P, N, d_input=0, pwl_timescale=None,
                 temporal_input_target='all', P_switch=None):
        """
        Parameters
        ----------
        pwl_timescale : float or None
            If set, the last P_switch entries of A are initialised to this
            value (e.g. 0.99) so the slow-switching units start with a slow
            intrinsic time scale.  Has no effect when P_switch == 0.
        temporal_input_target : str
            'all' (default), 'pwl_only', or 'monotone_gate'.
        P_switch : int or None
            Number of slow-switching PWL units (last P_switch of the P units).
            None → P_switch = P (all slow-switching; backward-compatible).
        """
        super().__init__()
        if temporal_input_target not in self._VALID_TARGETS:
            raise ValueError(
                f"temporal_input_target must be one of {self._VALID_TARGETS}, "
                f"got {temporal_input_target!r}"
            )
        P_switch = P if P_switch is None else int(P_switch)
        if P_switch > P:
            raise ValueError(f"P_switch ({P_switch}) must be <= P ({P})")
        self.M        = M
        self.P        = P
        self.P_switch = P_switch
        self.N        = N
        self.d_input  = d_input
        self.temporal_input_target = temporal_input_target

        self.A, self.W, self.h = self._initialize_AWh(pwl_timescale)
        self.B = self._init_uniform((N, M))

        # Temporal input projection(s) — only onto the P_switch slow units
        if d_input > 0:
            if temporal_input_target == 'all':
                # Full projection to all M units (original behaviour)
                self.C     = nn.Parameter(torch.randn(M, d_input) * 0.01)
                self.C_pwl = None
            else:
                # Projection restricted to the P_switch slow-switching units
                self.C     = None
                self.C_pwl = (nn.Parameter(torch.randn(P_switch, d_input) * 0.01)
                              if P_switch > 0 else None)
                if temporal_input_target == 'monotone_gate' and P_switch > 0:
                    # Per-unit sigmoid gates: σ(alpha_gate[i] * (t_norm − tau_gate[i]))
                    # alpha_gate shape (P_switch,), tau_gate shape (P_switch,).
                    # tau_gate initialised evenly across (0, 1) to break symmetry.
                    self.alpha_gate = nn.Parameter(torch.full((P_switch,), 5.0))
                    self.tau_gate   = nn.Parameter(
                        torch.linspace(1 / (P_switch + 1), P_switch / (P_switch + 1),
                                       P_switch)
                    )
        else:
            self.C     = None
            self.C_pwl = None

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

        if u is not None:
            if self.C is not None:
                # 'all': add to every latent unit
                out = out + u @ self.C.t()
            elif self.C_pwl is not None and self.P_switch > 0:
                if self.temporal_input_target == 'monotone_gate':
                    # Per-unit gate: σ(α[i] · (t_norm − τ[i]))
                    # u[..., 0:1] shape (..., 1); alpha/tau_gate shape (P_switch,)
                    # → gate shape (..., P_switch) via broadcasting
                    gate = torch.sigmoid(
                        self.alpha_gate * (u[..., 0:1] - self.tau_gate)
                    )  # (..., P_switch)
                    pwl_contrib = gate * (u @ self.C_pwl.t())  # (..., P_switch)
                else:
                    # 'pwl_only': direct projection onto slow-unit pre-activations
                    pwl_contrib = u @ self.C_pwl.t()  # (..., P_switch)
                out = out.clone()
                out[..., -self.P_switch:] = out[..., -self.P_switch:] + pwl_contrib

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

    def temporal_input_norms(self):
        """Return per-unit temporal-input weight L2 norms as a (M,) tensor.

        Units with no temporal projection (linear units under 'pwl_only' /
        'monotone_gate') are assigned norm 0.  Returns None when d_input == 0.
        """
        if self.d_input == 0:
            return None
        norms = torch.zeros(self.M)
        if self.C is not None:
            norms = self.C.detach().cpu().norm(dim=1)       # (M,)
        elif self.C_pwl is not None and self.P_switch > 0:
            norms[-self.P_switch:] = self.C_pwl.detach().cpu().norm(dim=1)  # (P_switch,)
        return norms

    # -- Initialization (matches AL-RNN tutorial notebook) --

    def _initialize_AWh(self, pwl_timescale=None):
        A = nn.Parameter(torch.diagonal(
            self._normalized_positive_definite(self.M), 0))
        if pwl_timescale is not None and self.P_switch > 0:
            with torch.no_grad():
                A[-self.P_switch:] = pwl_timescale
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
