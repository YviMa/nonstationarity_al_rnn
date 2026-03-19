import copy
import numpy as np
import torch
import torch.nn as nn

from .model import AL_RNN


class ALRNNDataset:
    """Time series dataset that produces batches of subsequences for AL-RNN training.

    Parameters
    ----------
    data : ndarray of shape (T, N)
        Single-subject time series (already preprocessed/scaled).
    external_inputs : ndarray of shape (T, d_input) or None
        Optional external input sequence (e.g. temporal features).
        Sampled in lock-step with data; never used as a prediction target.
    sequence_length : int
        Length of subsequences to sample.
    batch_size : int
        Number of subsequences per batch.
    device : str
        Torch device ('cpu' or 'cuda').
    """

    def __init__(self, data, external_inputs=None, sequence_length=50,
                 batch_size=8, device='cpu'):
        self.X = torch.tensor(data, dtype=torch.float32).to(device)
        self.T, self.N = data.shape
        self.U = (torch.tensor(external_inputs, dtype=torch.float32).to(device)
                  if external_inputs is not None else None)
        self.sequence_length = min(sequence_length, self.T - 1)
        self.batch_size = batch_size
        self.device = device

    def sample_batch(self):
        """Sample a batch of (input, target, external_input) subsequences.

        Returns
        -------
        x : Tensor of shape (batch, seq_len, N) — input subsequences
        y : Tensor of shape (batch, seq_len, N) — targets (shifted by 1)
        u : Tensor of shape (batch, seq_len, d_input), or None
        """
        max_start = self.T - self.sequence_length - 1
        if max_start < 1:
            max_start = 1
        starts = torch.randint(0, max_start, (self.batch_size,))

        x = torch.stack([self.X[s:s + self.sequence_length] for s in starts])
        y = torch.stack([self.X[s + 1:s + self.sequence_length + 1] for s in starts])
        u = (torch.stack([self.U[s:s + self.sequence_length] for s in starts])
             if self.U is not None else None)
        return x, y, u


def generalized_teacher_forcing(z, z_hat, alpha):
    """Blend model prediction with observed value at every step.

    Parameters
    ----------
    z : Tensor — model prediction (observation space)
    z_hat : Tensor — true observation
    alpha : float — forcing strength (1.0 = use observation fully,
                    0.0 = use model prediction fully)
    """
    return alpha * z_hat + (1 - alpha) * z


def predict_sequence_using_gtf(model, x, alpha, u=None, n_interleave=1,
                               return_pwl_states=False):
    """Forward pass with Generalized Teacher Forcing.

    Teacher forcing is applied every ``n_interleave`` steps (default 1 =
    every step).  Between forcing events the model runs freely (same as
    alpha=0 for those steps).  This matches the notebook training loop where
    ``n_interleave=16`` is used for chaotic systems like Lorenz-63.

    Parameters
    ----------
    model : AL_RNN
    x : Tensor of shape (batch, seq_len, N)
    alpha : float — forcing strength at each forcing event
    u : Tensor of shape (batch, seq_len, d_input) or None
    n_interleave : int
        Apply teacher forcing every this many steps.  1 = every step
        (original behaviour).  Must be >= 1.
    return_pwl_states : bool

    Returns
    -------
    Z : Tensor of shape (batch, seq_len, N)
    pwl_states : Tensor of shape (batch, seq_len, P)  [only if requested]
    """
    P = model.P
    x_ = x.permute(1, 0, 2)  # (seq_len, batch, N)
    u_ = u.permute(1, 0, 2) if u is not None else None
    T = x_.shape[0]

    Z = torch.empty_like(x_)
    collect = return_pwl_states and P > 0
    pwl_list = [] if collect else None

    # Initial step: map first observation through model
    lat = model(model.obs_to_latent(x_[0]), u=u_[0] if u_ is not None else None)
    z = model.latent_to_obs(lat)
    Z[0] = z
    if collect:
        pwl_list.append(lat[:, -P:])

    for t in range(1, T):
        if t % n_interleave == 0:
            z_forced = generalized_teacher_forcing(z, x_[t], alpha)
        else:
            z_forced = z  # free run between forcing events
        lat = model(model.obs_to_latent(z_forced), u=u_[t] if u_ is not None else None)
        z = model.latent_to_obs(lat)
        Z[t] = z
        if collect:
            pwl_list.append(lat[:, -P:])

    Z_out = Z.permute(1, 0, 2)  # (batch, seq_len, N)
    if collect:
        return Z_out, torch.stack(pwl_list, dim=1)  # (batch, seq_len, P)
    return Z_out


def predict_sequence_using_itf(model, x, alpha, u=None, n_interleave=1,
                               return_pwl_states=False):
    """Forward pass with Identity Teacher Forcing (ITF).

    Neuron groups
    -------------
    z[0 : N]      — readout / observed units.  At each step their value is an
                    alpha-blend of the model's own prediction and the true
                    observation.  They carry temporal history via the recurrent
                    dynamics AND receive grounding from the data.
    z[N : M]      — free latent units.  Fully unobserved; evolve according to
                    the autonomous dynamics across the entire sequence.
    z[M-P : M]    — the P piecewise-linear (ReLU) units, a subset of the free
                    latent units (guaranteed by M >= N + P).  These act as
                    nonlinear regime switches.

    Because the observation map is the identity (readout = first N units), no
    encoder/decoder (B matrix) is needed or used here.

    alpha=1.0 : true observation fully replaces model prediction each step
    alpha=0.0 : free running — model drives its own first-N prediction too

    With n_interleave > 1 forcing only happens at multiples of n_interleave;
    between those steps the readout units evolve freely (equivalent to
    alpha=0 for those steps).  Matches the notebook's n_interleave=16 for
    Lorenz-63 training.

    Requires M >= N + P so the readout units (first N) do not overlap with the
    ReLU units (last P).

    Parameters
    ----------
    model : AL_RNN
    x : Tensor of shape (batch, seq_len, N)
    alpha : float — forcing strength at each forcing event
    u : Tensor of shape (batch, seq_len, d_input) or None
        External input sequence injected via factor loading matrix C.
    n_interleave : int
        Apply teacher forcing every this many steps.  1 = every step.
    return_pwl_states : bool
        If True and model.P > 0, also return the pre-activation values of the
        P ReLU units at every step as a (batch, seq_len, P) tensor.
        These are the raw z_out[-P:] values before they are ReLU'd on the
        next forward call — their sign defines the active bitcode.

    Returns
    -------
    Z : Tensor of shape (batch, seq_len, N) — predictions in observation space
    pwl_states : Tensor of shape (batch, seq_len, P), only when return_pwl_states=True and P>0
    """
    N = model.N
    M = model.M
    P = model.P
    x_ = x.permute(1, 0, 2)                      # (seq_len, batch, N)
    u_ = u.permute(1, 0, 2) if u is not None else None
    T, batch_size, _ = x_.shape

    Z = torch.empty_like(x_)
    collect = return_pwl_states and P > 0
    pwl_list = [] if collect else None

    # Seed free latent units from the B projection of the first observation,
    # matching the notebook: z = x[0] @ B; z[:N] = x[0].
    with torch.no_grad():
        h = model.obs_to_latent(x_[0])[:, N:].detach().clone()

    z_in  = torch.cat([x_[0], h], dim=1)
    z_out = model(z_in, u=u_[0] if u_ is not None else None)
    Z[0]  = z_out[:, :N]
    h     = z_out[:, N:]
    if collect:
        pwl_list.append(h[:, -P:])

    for t in range(1, T):
        if t % n_interleave == 0:
            obs_forced = alpha * x_[t] + (1 - alpha) * z_out[:, :N]
        else:
            obs_forced = z_out[:, :N]  # free run between forcing events
        z_in  = torch.cat([obs_forced, h], dim=1)
        z_out = model(z_in, u=u_[t] if u_ is not None else None)
        Z[t]  = z_out[:, :N]
        h     = z_out[:, N:]
        if collect:
            pwl_list.append(h[:, -P:])

    Z_out = Z.permute(1, 0, 2)  # (batch, seq_len, N)
    if collect:
        return Z_out, torch.stack(pwl_list, dim=1)  # (batch, seq_len, P)
    return Z_out


def train_alrnn(train_data, M, P, num_epochs=500, batch_size=8,
                sequence_length=50, alpha=1.0, end_alpha=None,
                n_interleave=1,
                start_lr=1e-3, end_lr=1e-5, batches_per_epoch=30,
                device='cpu', verbose=False, model=None,
                forcing_mode='gtf', external_inputs=None,
                weight_decay=1e-5, switch_lambda=0.0, pwl_timescale=None,
                timescale_lambda=0.0,
                epoch_callback=None, callback_every=100):
    """Train an AL-RNN model on a single subject's time series.

    Parameters
    ----------
    train_data : ndarray of shape (T, N)
        Training time series (already demeaned/scaled).
    M : int
        Total latent units.
    P : int
        Number of ReLU units (0 = fully linear).
    num_epochs : int
        Training epochs.
    batch_size : int
    sequence_length : int
        Subsequence length for training batches.
    alpha : float
        Initial teacher-forcing strength (start of schedule).
    end_alpha : float or None
        Final teacher-forcing strength.  If None, alpha is held constant.
    n_interleave : int
        Apply teacher forcing every this many steps (default 1 = every step).
        The notebook uses 16 for Lorenz-63 so the model must sustain free
        runs of 16 steps, forcing it to learn autonomous dynamics rather than
        pure one-step-ahead prediction.
        Final teacher-forcing strength.  If None, alpha is held constant.
        If set, alpha decays linearly from ``alpha`` to ``end_alpha`` over
        ``num_epochs`` epochs.
    start_lr, end_lr : float
        Learning rate schedule endpoints (exponential decay).
    batches_per_epoch : int
    device : str
    verbose : bool
    model : AL_RNN or None
        If provided, continue training this model instead of creating a new one.
        Useful for warm-start rolling window training.
    forcing_mode : str
        'gtf' — Generalized Teacher Forcing (blended, uses alpha).
        'itf' — Identity Teacher Forcing (direct injection into first N units).
                Requires M >= N + P so the forced/readout units (first N) do
                not overlap with the ReLU units (last P).
    external_inputs : ndarray of shape (T, d_input) or None
        Optional external input sequence injected into the latent dynamics
        via factor loading matrix C.  Never used as a prediction target.
        When provided a new model is initialised with d_input set accordingly.
    switch_lambda : float
        Weight for the differentiable switch-count penalty (default 0 = off).
        Uses a scale-invariant event-based proxy: each PWL unit's value is
        normalised to ≈ ±1 via s/(|s|.detach()+ε), then
        (norm_t − norm_{t+1})²/4 ≈ 1 at a sign crossing and ≈ 0 otherwise.
        Summed over the sequence and averaged over batch and P units this
        approximates the hard switch count.  Gradient ≈ 1/(|s|+ε) — largest
        near zero crossings.  Has no effect when P == 0.
    pwl_timescale : float or None
        If provided, the last P diagonal entries of A are initialised to this
        value (e.g. 0.98) so the PWL units start with a slow intrinsic time
        scale.  Ignored when a pre-built model is passed in.
    timescale_lambda : float
        Weight for the timescale regularisation (default 0 = off).  Penalises
        the L2 deviation of A[-P:] from pwl_timescale, keeping the slow time
        scale during training.  Requires pwl_timescale to be set.  Has no
        effect when P == 0 or pwl_timescale is None.
    epoch_callback : callable or None
        If provided, called as ``epoch_callback(model, epoch)`` after every
        ``callback_every`` epochs (and at the final epoch).  The model is
        temporarily put in eval mode for the call, then restored to train mode.
        Intended for mid-training visualisation — e.g. alignment plots.
    callback_every : int
        How often to fire ``epoch_callback`` (default 100 epochs).

    Returns
    -------
    model : AL_RNN
        Trained model in eval mode (best by training loss).
    losses : list of float
        Average loss per epoch.
    """
    N = train_data.shape[1]
    d_input = external_inputs.shape[1] if external_inputs is not None else 0

    if forcing_mode == 'itf' and M < N + P:
        raise ValueError(
            f"Identity Teacher Forcing requires M >= N + P "
            f"(got M={M}, N={N}, P={P} → need M >= {N + P}). "
            f"The first N latent units are forced readout states and must "
            f"not overlap with the last P ReLU units."
        )

    if forcing_mode == 'gtf' and M > N:
        import warnings
        warnings.warn(
            f"GTF with M ({M}) > N ({N}): obs_to_latent re-initialises the last "
            f"{M - N} latent units from the current (blended) observation at every "
            f"step, so those units carry no autonomous temporal history. Effective "
            f"recurrent memory = N = {N} dimensions regardless of M.  "
            f"GTF is designed for M == N (identity latent space, no encoder needed).  "
            f"To benefit from M > N with free latent memory, use forcing_mode='itf': "
            f"first N units are blended with observations (alpha), last M-N evolve "
            f"freely, and the last P of those are the nonlinear regime switches.",
            UserWarning, stacklevel=2,
        )

    seq_len = min(sequence_length, len(train_data) - 1)
    dataset = ALRNNDataset(train_data, external_inputs=external_inputs,
                           sequence_length=seq_len,
                           batch_size=batch_size, device=device)

    if model is None:
        model = AL_RNN(M=M, P=P, N=N, d_input=d_input,
                       pwl_timescale=pwl_timescale).to(device)
    else:
        model = model.to(device)

    # In ITF the observation map is the identity (readout = first N units);
    # B is never called, so freeze it to keep the optimizer graph clean.
    if forcing_mode == 'itf':
        model.B.requires_grad_(False)

    loss_fn = nn.MSELoss()
    # Only optimize parameters that require gradients (excludes frozen B in ITF)
    optimizer = torch.optim.RAdam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=start_lr, weight_decay=weight_decay,
    )
    gamma = np.exp(np.log(end_lr / start_lr) / max(num_epochs, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    _end_alpha = alpha if end_alpha is None else end_alpha
    best_model_state = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    losses = []

    model.train()
    for epoch in range(num_epochs):
        # Linear alpha schedule: start strong, relax toward autonomous dynamics
        t = epoch / max(num_epochs - 1, 1)
        alpha_t = alpha + (_end_alpha - alpha) * t

        collect_pwl = switch_lambda > 0.0 and model.P > 0
        epoch_losses = []
        epoch_switch_counts = []
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
            x, y, u = dataset.sample_batch()
            if forcing_mode == 'itf':
                result = predict_sequence_using_itf(model, x, alpha_t, u=u,
                                                    n_interleave=n_interleave,
                                                    return_pwl_states=collect_pwl)
            else:
                result = predict_sequence_using_gtf(model, x, alpha_t, u=u,
                                                    n_interleave=n_interleave,
                                                    return_pwl_states=collect_pwl)
            if collect_pwl:
                z_hat, pwl_states = result
                # Scale-invariant event-based switch penalty.
                # Normalise each value to ≈ ±1: s / (|s|.detach() + ε).
                # Gradient ≈ 1/(|s|+ε) — largest near zero crossings.
                # (norm_t - norm_{t+1})²/4 ≈ 1 at a sign crossing, ≈ 0 otherwise.
                # Summed over time, averaged over batch and P → ≈ hard switch count.
                norm_sign = pwl_states / (pwl_states.detach().abs() + 1e-6)
                switch_count = (
                    (norm_sign[:, :-1, :] - norm_sign[:, 1:, :]).pow(2) / 4.0
                ).sum(dim=1).mean()
                loss = loss_fn(z_hat, y) + switch_lambda * switch_count
                epoch_switch_counts.append(switch_count.item())
            else:
                z_hat = result
                loss = loss_fn(z_hat, y)
            if timescale_lambda > 0.0 and pwl_timescale is not None and model.P > 0:
                loss = loss + timescale_lambda * (model.A[-model.P:] - pwl_timescale).pow(2).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % 100 == 0 or epoch == num_epochs - 1):
            msg = f"    Epoch {epoch}/{num_epochs}: loss={avg_loss:.6f}  alpha={alpha_t:.3f}"
            if epoch_switch_counts:
                msg += f"  switch_count={np.mean(epoch_switch_counts):.2f}"
            print(msg)

        if epoch_callback is not None and (
            (epoch + 1) % callback_every == 0 or epoch == num_epochs - 1
        ):
            model.eval()
            epoch_callback(model, epoch + 1)
            model.train()

    model.load_state_dict(best_model_state)
    model.eval()
    return model, losses
