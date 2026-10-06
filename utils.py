import torch
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

from datetime import datetime
import json
import seaborn as sns


def predict_sequence_using_gtf(model, x, s, alpha):
    """
    Performs an entire forward pass of the model given training sequence `x`
    and possible external inputs `s` using generalized teacher forcing with
    forcing strength `alpha`.
    """
    T = x.shape[1]
    x_ = x.permute(1, 0, 2)
    if s is not None:
        s_ = s.permute(1, 0, 2)
    else:
        s_ = [None] * T

    Z = torch.empty_like(x_)
    # initial prediction based on first data point
    z = model(x_[0], s_[0])
    Z[0] = z
    # forward pass using GTF
    for t in range(1, T):
        z_hat = x_[t]
        z = model(generalized_teacher_forcing(z, z_hat, alpha), s_[t])
        Z[t] = z
    return Z.permute(1, 0, 2)

def train_with_early_stopping(model, train_ds, val_ds, optimizer, loss_fn,
                              num_epochs, alpha, patience=100):
    """
    Train model with early stopping on validation loss.
    Stops if no improvement for 'patience' epochs.
    """
    
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    losses = []
    best_state = None
    
    model.train()
    with trange(num_epochs, unit="batches") as pbar:
        for epoch in pbar:
            optimizer.zero_grad()
            x, y, s = train_ds.sample_batch()
            y_hat = predict_sequence_using_gtf(model, x, s, alpha)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                val_loss = evaluate(model, val_ds, loss_fn, alpha)
                pbar.set_postfix(train_loss=loss.item(), val_loss=val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    best_state = model.state_dict()
                else:
                    epochs_since_improvement += 10

                if epochs_since_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return losses, best_val_loss


def generalized_teacher_forcing(z, z_hat, alpha):
    return alpha * z_hat + (1 - alpha) * z

def train(model, dataset, optimizer, loss_fn, num_epochs, alpha):
    """
    Simple training method. Train `model` for `num_epochs` where one epoch is defined
    by processing a single training batch.
    """
    model.train()
    losses = []
    with trange(num_epochs, unit="batches") as pbar:
      pbar.set_description(f"Training Progress")
      for epoch in pbar:
          optimizer.zero_grad()
          x, y, s = dataset.sample_batch()
          y_hat = predict_sequence_using_gtf(model, x, s, alpha)
          loss = loss_fn(y_hat, y)
          loss.backward()
          optimizer.step()
          if epoch % 10 == 0:
              pbar.set_postfix(loss=loss.item())
          losses.append(loss.item())
    return losses


def compute_and_smooth_power_spectrum(x, smoothing):
    x_ =  (x - x.mean()) / x.std()
    fft_real = np.fft.rfft(x_)
    ps = np.abs(fft_real)**2 * 2 / len(x_)
    ps_smoothed = gaussian_filter1d(ps, smoothing)
    return ps_smoothed / ps_smoothed.sum()

def hellinger_distance(p, q):
    return np.sqrt(1-np.sum(np.sqrt(p*q)))

def power_spectrum_error(X, X_gen, smoothing):
    dists = []
    for i in range(X.shape[1]):
        ps = compute_and_smooth_power_spectrum(X[:, i], smoothing)
        ps_gen = compute_and_smooth_power_spectrum(X_gen[:, i], smoothing)
        dists.append(hellinger_distance(ps, ps_gen))
    return np.mean(dists)

# D_stsp
def calc_histogram(x, n_bins, min_, max_):
    dim_x = x.shape[1]  # number of dimensions

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = torch.ones(coordinates.shape[0], device=coordinates.device)
    if 'cuda' == coordinates.device.type:
        tens = torch.cuda.sparse.FloatTensor
    else:
        tens = torch.sparse.FloatTensor
    return tens(coordinates.t(), indices, size=size_).to_dense()

def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf

def kullback_leibler_divergence(p1, p2):
    """
    Calculate the Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = torch.tensor([float('nan')])
    else:
        kl = (p1 * torch.log(p1 / p2)).sum()
    return kl

def state_space_divergence_binning(x_gen, x_true, n_bins=30):
    x_true_ = torch.tensor(x_true)
    x_gen_ = torch.tensor(x_gen)
    min_, max_ = x_true_.min(0).values, x_true_.max(0).values
    hist_gen = calc_histogram(x_gen_, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true_, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return kullback_leibler_divergence(p_true, p_gen).item()

def clean_from_outliers(prior, posterior):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()
    return prior, posterior, outlier_ratio

def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = torch.diag_embed(precision).float()

    prec_vec = torch.einsum('zij,azj->azi', precision, vec)
    exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = torch.prod(std, dim=1)
    likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T

def save_and_plot_data(parameters, model, losses, X_test, orbit, name):
    if name is None:
        now = datetime.now()
        name = now.strftime("%Y%m%d_%H%M%S")
    folder_name = f"./results/{name}"
    os.makedirs(folder_name, exist_ok=True)

    with open(os.path.join(folder_name, "parameters.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    model_path = os.path.join(folder_name, "shallowPLRNN.pt")
    torch.save(model.state_dict(), model_path)

    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    plt.savefig(os.path.join(folder_name, "loss_curve.png"))
    plt.close()

    ntimepoints = parameters['sequence_length']
    nnodes = X_test.shape[1]
    fig, axes = plt.subplots(nnodes, 1, figsize=(10, 2 * nnodes), sharex=True)
    for node in range(nnodes):
        axes[node].plot(X_test[0:ntimepoints, node])
        axes[node].plot(orbit[0:ntimepoints, node], linestyle='--')
        axes[node].set_ylabel(f'Node {node + 1}')
        axes[node].grid(True)
    plt.show()
    fig.savefig(os.path.join(folder_name, "target_vs_generated.png"))
    plt.close(fig)

    functional_connectivity = np.corrcoef(orbit.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        functional_connectivity,
        cmap="coolwarm",
        center=0,
        square=True,
        annot=True,           # show correlation values in each cell
        fmt=".3f",            # format values (3 decimal places)
        annot_kws={"size": 8},
        cbar_kws={"label": "Connection Strength"},
        vmin=-1, vmax=1       # fixed color scale for correlations
    )
    plt.title("Functional Connectivity Matrix (Weights)")
    plt.xlabel("Node index")
    plt.ylabel("Node index")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(folder_name, "connectivity_matrix.png"))
    plt.close()

def state_space_divergence_gmm(X_gen, X_true, scaling=1.0, max_used=10000):
    time_steps = min(X_true.shape[0], max_used)
    mu_true = X_true[:time_steps, :]
    mu_gen = X_gen[:time_steps, :]

    cov_true = torch.ones(X_true.shape[-1]).repeat(time_steps, 1) * scaling
    cov_gen = torch.ones(X_gen.shape[-1]).repeat(time_steps, 1) * scaling

    mc_n = 1000
    t = torch.randint(0, mu_true.shape[0], (mc_n,))

    std_true = torch.sqrt(cov_true)
    std_gen = torch.sqrt(cov_gen)

    z_sample = (mu_true[t] + std_true[t] * torch.randn(mu_true[t].shape)).reshape((mc_n, 1, -1))
    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_true, std_true)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = torch.mean(torch.log(posterior + 1e-8) - torch.log(prior + 1e-8), dim=0)
    return kl_mc.item()

def standardize_timeseries(timeseries):
    """ Standardize timeseries to zero mean and unit variance.
    
    timeseries: numpy array of shape T x N"""
    mean_ts = np.mean(timeseries, axis=0, keepdims=True)
    std_ts = np.std(timeseries, axis=0, keepdims=True)
    return (timeseries - mean_ts) / std_ts


def preprocess_ts(data, sigma, smooth=True):
    if smooth:
        data = gaussian_filter1d(data, sigma=sigma, axis=0)
    return standardize_timeseries(data)


def stabilize_matrix(A, max_eigenvalue=0.95):
    """
    Rescales matrix A so its spectral radius is at most max_eigenvalue (default 0.95).
    A: torch.Tensor square matrix (M x M)
    Returns: rescaled torch.Tensor A
    """
    with torch.no_grad():
        # Move A to CPU for numpy eigenvalue computation if needed
        A_cpu = A.cpu().numpy()
        eigenvalues = np.linalg.eigvals(A_cpu)
        spectral_radius = np.max(np.abs(eigenvalues))
        if spectral_radius > max_eigenvalue:
            scaling_factor = max_eigenvalue / spectral_radius
            A.mul_(scaling_factor)
    return A

def evaluate(model, dataset, loss_fn, alpha):
    """Evaluate model loss on given dataset without updating parameters."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for _ in range(len(dataset) // dataset.batch_size):
            x, y, s = dataset.sample_batch()
            y_hat = predict_sequence_using_gtf(model, x, s, alpha)
            loss = loss_fn(y_hat, y)
            total_loss += loss.item()
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches > 0 else float('nan')

@torch.no_grad()
def generate_orbit(model, z1, T, S=None):
    """
    Generate an orbit of `model`, i.e. starting from initial condition `z1`, draw
    a trajectory of length `T` with optional external input matrix `S` of shape
    `T x K`.
    """
    if S is None:
        S = [None] * T

    z = z1
    orbit = [z]
    for t in range(T):
        z = model(z, S[t])
        orbit.append(z)
    return torch.stack(orbit)