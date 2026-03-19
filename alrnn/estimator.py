import numpy as np
import pandas as pd
import torch

from .model import AL_RNN
from .training import train_alrnn


class ALRNNEstimator:
    """AL-RNN estimator matching the NetworkEstimator interface.

    Wraps AL-RNN training and prediction so it can be used in the same
    rolling-window framework as VAR/Lasso/AR models.

    For the linear case (P=0, M=N), estimate_network() returns a coefficient
    matrix compatible with ``state = matrix @ state``. For the nonlinear case,
    it returns None but the trained model is stored for use via predict().

    Parameters
    ----------
    M : int or None
        Total latent units. If None, defaults to N (number of variables).
    P : int
        Number of ReLU units. 0 = fully linear (equivalent to VAR).
    num_epochs, batch_size, sequence_length, alpha,
    start_lr, end_lr, batches_per_epoch : training hyperparameters
        Forwarded to train_alrnn().
    finetune_epochs : int or None
        Epochs used for warm-start continuation. Defaults to num_epochs.
    alpha : float
        GTF forcing strength (only used when forcing_mode='gtf').
    forcing_mode : str
        'gtf' — Generalized Teacher Forcing (blended, uses alpha).
        'itf' — Identity Teacher Forcing (direct injection into first N units,
                free latent units N..M-1 evolve freely). Requires M >= N + P.
    device : str
        'cpu' or 'cuda'.
    verbose : bool
    """

    def __init__(self, M=None, P=0, num_epochs=500, finetune_epochs=None,
                 batch_size=8, sequence_length=50, alpha=1.0,
                 start_lr=1e-3, end_lr=1e-5, batches_per_epoch=30,
                 forcing_mode='gtf', device='cpu', verbose=False):
        self.M = M
        self.P = P
        self.num_epochs = num_epochs
        self.finetune_epochs = finetune_epochs if finetune_epochs is not None else num_epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.batches_per_epoch = batches_per_epoch
        self.forcing_mode = forcing_mode
        self.device = device
        self.verbose = verbose
        self._model = None

    def reset(self):
        """Discard the stored model (call between subjects/sessions)."""
        self._model = None

    def estimate_network(self, data, handle_missing='interpolate', warm_start=False):
        """Train AL-RNN and return coefficient matrix (linear case) or None.

        Parameters
        ----------
        data : ndarray (T, N)
        handle_missing : str
            'interpolate', 'drop', or anything else for no handling.
        warm_start : bool
            If True and a model already exists, continue training it on the
            new (expanded) data instead of initialising from scratch.
            Set to False at the start of each new subject/session.

        Returns
        -------
        coef_matrix : ndarray (N, N) or None
            Coefficient matrix for the linear case (P=0, M=N).
            None for the nonlinear case (model is still trained and stored).
        """
        if handle_missing == 'interpolate':
            df = pd.DataFrame(data)
            data_clean = df.interpolate(
                method='linear', limit_direction='both').values.astype(np.float32)
        elif handle_missing == 'drop':
            valid = ~np.isnan(data).any(axis=1)
            data_clean = data[valid].astype(np.float32)
        else:
            data_clean = data.astype(np.float32)

        if len(data_clean) < 10:
            return None

        N = data_clean.shape[1]
        M = self.M if self.M is not None else N
        seq_len = min(self.sequence_length, len(data_clean) - 1)

        existing_model = self._model if warm_start else None
        epochs = self.finetune_epochs if warm_start else self.num_epochs

        try:
            self._model, _ = train_alrnn(
                data_clean, M=M, P=self.P,
                num_epochs=epochs, batch_size=self.batch_size,
                sequence_length=seq_len, alpha=self.alpha,
                start_lr=self.start_lr, end_lr=self.end_lr,
                batches_per_epoch=self.batches_per_epoch,
                device=self.device, verbose=self.verbose,
                model=existing_model,
                forcing_mode=self.forcing_mode,
            )
        except Exception as e:
            if self.verbose:
                print(f"    AL-RNN training failed: {e}")
            self._model = None
            return None

        return self._model.get_coefficient_matrix()

    def predict(self, state, horizon=1):
        """Multi-step prediction using the last trained model.

        For GTF mode: maps observation through encoder B, runs latent dynamics,
        reads out via decoder B.
        For ITF mode: places observation in first N latent units, free latent
        units start at zero, runs dynamics, reads out first N units directly.

        Parameters
        ----------
        state : ndarray of shape (N,)
            Current observation (in scaled space).
        horizon : int
            Number of steps to predict ahead.

        Returns
        -------
        prediction : ndarray of shape (N,)
        """
        if self._model is None:
            raise RuntimeError("No trained model. Call estimate_network() first.")

        model = self._model
        N = model.N
        M = model.M
        dev = next(model.parameters()).device

        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(dev)

            if self.forcing_mode == 'itf':
                h = torch.zeros(1, M - N, dtype=torch.float32, device=dev)
                z = torch.cat([x, h], dim=1)
                for _ in range(horizon):
                    z = model(z)
                return z[0, :N].cpu().numpy()
            else:
                z = model.obs_to_latent(x)
                for _ in range(horizon):
                    z = model(z)
                return model.latent_to_obs(z).squeeze(0).cpu().numpy()
