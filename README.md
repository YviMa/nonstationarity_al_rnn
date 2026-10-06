# AL-RNN with temporal external input

Almost-Linear RNNs (AL-RNNs) trained on non-stationary time series, where the
model additionally receives **time as an external input**. The goal is to
understand how the model detects and represents regime changes (switches)
in the data.

Latent dynamics (see [alrnn/model.py](alrnn/model.py)):

```
z_{t+1} = A z_t + W phi(z_t) + h + (temporal input term)
```

`phi` applies a ReLU to the last `P` latent units (the piecewise-linear, PWL
units). Each sign pattern of these units (a *bitcode*) selects one of `2^P`
linear dynamical systems.

## Temporal input

The temporal feature `u_t` is never observed or predicted; it only drives the
latent dynamics. How it is provided is controlled by three config keys
(see the `CONFIG` dict in each run file):

```python
"use_temporal_input": True,
"temporal_input_mode": "linear",       # 'none' | 'linear' | 'fourier'
"temporal_input_target": "all",        # 'all' | 'pwl_only' | 'monotone_gate'
```

**What is fed in** (`temporal_input_mode`, [models/temporal_input.py](models/temporal_input.py)):

| mode      | `u_t`                                                        |
|-----------|--------------------------------------------------------------|
| `none`    | no temporal input                                            |
| `linear`  | one scalar, normalised time `t / (T-1)`                      |
| `fourier` | sin/cos pairs at `n_fourier_pairs` frequencies of `t/(T-1)`  |

**Where it enters** (`temporal_input_target`):

| target          | effect                                                                                                   |
|-----------------|----------------------------------------------------------------------------------------------------------|
| `all`           | `C u_t` is added to all latent units                                                                     |
| `pwl_only`      | projection only onto the (slow-switching) PWL units' pre-activations                                      |
| `monotone_gate` | like `pwl_only`, but scaled by a learned per-unit sigmoid gate `σ(α (t_norm − τ))`; biases the model towards a single clean transition. Requires `linear` mode |

## Experiments

Each script in [experiments/](experiments/) has its own `CONFIG` at the top;
edit it and run from the repository root, e.g.

```
python experiments/run_bifurcation.py
```

| script                | data                                                                  |
|-----------------------|-----------------------------------------------------------------------|
| `run_sinusoid.py`     | switching multi-frequency sinusoids                                   |
| `run_lorenz.py`       | switching Lorenz-63                                                   |
| `run_bifurcation.py`  | switching bifurcation ODEs (Hopf, Van der Pol, FitzHugh-Nagumo)       |
| `run_var.py`          | switching VAR(1) processes                                            |
| `run_ema.py`          | real EMA data                                                         |
| `run_eeg.py`          | CHB-MIT EEG (seizure detection)                                       |

Shared training logic lives in [experiments/runner.py](experiments/runner.py).
Outputs (models, plots) are written to `results/` (git-ignored).

## PWL switch analysis

The main analysis pipeline is [pwl_switch_analysis.ipynb](pwl_switch_analysis.ipynb),
backed by [analysis/pwl_switch_scan.py](analysis/pwl_switch_scan.py). It asks
how the switches of the PWL units relate to the generalized-teacher-forcing
(GTF) loss landscape of a trained model, i.e. *why* the model can sense a
regime change:

1. **Natural run**: bitcode trajectory and per-step MSE.
2. **Effective connectivity**: the effective `W` for each bitcode region.
3. **Bitcode-forced loss scan**: for each candidate switch time `t*`, force the
   transition `bc_pre → bc_post` (clamping the sign of the PWL pre-activations)
   and measure the loss. A minimum near the true switch time explains how the
   model encodes the regime change.
4. **Per-step loss breakdown** coloured by the active bitcode.

## Layout

```
alrnn/        AL-RNN model, training, estimator
models/       temporal input module
data/         data generators (switching ODE / sinusoid / VAR, EMA data)
experiments/  run_*.py entry points + shared runner
analysis/     regime logging, alignment, consistency, PWL switch scan
results/      outputs (git-ignored)
```
