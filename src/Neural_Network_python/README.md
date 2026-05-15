# Neural Network — Physics-Informed PDF Predictor

Python pipeline (TensorFlow 2 / Keras) that learns the probability-density function
(PDF) of particle positions from Langevin simulation data and predicts its future
evolution without additional simulation.

---

## Pipeline Overview

```
simulation.bin
      │
      ▼
data_preparation.py      ← converts binary → prepdata_from_binary.parquet
      │
      ├──► data_analytic.py  ← computes analytic Fokker–Planck Gaussian (analytic_data.parquet)
      │
      ▼
neural_network.NN.data_prep()
      │  histogram features  (n_bins=600) + normalised time → coarse analytic PDF
      │
      ├──► nn_model()          ← PDF predictor (histogram + t → PDF)
      │
      └──► build_propagator_dataset()
                │
                ▼
           nn_model_propagator()  ← Fokker–Planck propagator (FP physics loss)
                │
                ▼
           rollout()              ← autoregressive future prediction
                │
                ▼
           comparison_propagator() ← overlays predictions vs. analytic PDF
```

---

## Architecture

### PDF Predictor (`nn_model`)

| Layer | Units | Activation |
|-------|-------|-----------|
| Input | n\_bins + 1 | — |
| Dense | 512 | GELU |
| BatchNormalization | — | — |
| Dense | 256 | GELU |
| BatchNormalization | — | — |
| Dropout | — | 0.1 |
| Dense | 256 | GELU |
| BatchNormalization | — | — |
| Dense | 512 | GELU |
| BatchNormalization | — | — |
| Output | n\_bins | softplus (non-negative) |

Input: `[density_histogram (600 bins) | t_normalised]`
Target: analytic PDF interpolated onto the 600-bin grid.

### Fokker–Planck Propagator (`nn_model_propagator`)

Same hourglass architecture, trained with a composite loss:

$$\mathcal{L} = \underbrace{\text{MSE}(p_{\text{pred}},\, p_{\text{true}})}_{\text{data}} + \lambda \underbrace{\left\|\frac{\Delta p}{\Delta\tau} + a_{\text{eff}}\frac{\partial p}{\partial x} - D_{\text{eff}}\frac{\partial^2 p}{\partial x^2}\right\|^2}_{\text{Fokker–Planck residual}}$$

where spatial derivatives are computed with second-order central finite differences and:

$$a_{\text{eff}} = v_s \cdot \delta \cdot \texttt{timestep}, \qquad D_{\text{eff}} = D_t \cdot \delta \cdot \texttt{timestep}$$

Training pairs are built at gaps of 1 … `multi_step` saved snapshots to encourage
long-horizon consistency.

---

## Configuration — `settings.toml`

All hyper-parameters are centralised in [`code/settings.toml`](code/settings.toml).

```toml
[analytic]
x_min  = -25.0   # spatial domain left boundary
x_max  =  50.0   # spatial domain right boundary
x_step =   0.01  # analytic grid spacing (7500 points)

[network]
n_bins           = 600    # NN input/output resolution
train_test_split = 0.9    # fraction of snapshots used for training

[training]
epochs_nn             = 300
batch_size_nn         = 32
learning_rate_nn      = 3e-4
epochs_propagator     = 500
batch_size_propagator = 32
multi_step            = 15    # max prediction gap during propagator training

[physics]
delta    = 1e-3   # integration time-step (must match parameter.txt)
Dt       = 1.0    # diffusion coefficient
vs       = 1.0    # drift velocity
timestep = 10     # steps per saved snapshot
weight   = 0.1    # lambda — FP residual weight
```

**Important:** keep `[physics]` in sync with `src/Simulation_Cpp/code/parameter.txt`
whenever simulation parameters change.

---

## Training Callbacks

Both models use:

| Callback | Monitor | Setting |
|----------|---------|---------|
| `EarlyStopping` | val\_loss / val\_mse | patience=30, `restore_best_weights=True` |
| `ReduceLROnPlateau` | val\_loss / val\_mse | factor=0.5, patience=15, min\_lr=1e-6 |
| `ModelCheckpoint` / `_BackboneSaver` | val\_loss / val\_mse | saves best model to `keras_model/` |

---

## Running

```bash
# Install dependencies
uv sync

# Run full pipeline (binary data must exist — see Simulation_Cpp)
cd code
uv run python main.py

# Compare analytic vs. simulation at a single time-step
uv run python main.py --comparison
```

Trained models are saved to `code/keras_model/`.
Figures (HTML + PNG) are written to `code/figures/`.

---

## Dependencies

Managed via [uv](https://docs.astral.sh/uv/). Key packages:

| Package | Purpose |
|---------|---------|
| `tensorflow>=2.15` | Model training (CPU / Metal GPU) |
| `keras<3.0` | Keras 2 API |
| `numpy<2` | Numerical arrays |
| `pandas` / `pyarrow` | Parquet data storage |
| `plotly` + `kaleido` | Interactive HTML + static PNG figures |
