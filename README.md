# Physics-Informed Neural Networks (PINN)

A physics-informed machine learning pipeline for 1D Langevin dynamics.
The C++ simulator generates particle trajectory data; a Python neural network learns
the evolving probability-density function (PDF) and a physics-constrained propagator
predicts the PDF at future times without any further simulation.

---

## Physics Background

The position $x(t)$ of a particle follows the Langevin equation:

$$dx = v_s \, dt + \sqrt{2 D_t \, dt} \, \zeta(t)$$

where $v_s$ is the drift velocity, $D_t$ is the translational diffusion coefficient, and
$\zeta(t)$ is Gaussian white noise. The corresponding Fokker–Planck equation for the PDF
$p(x,t)$ is:

$$\frac{\partial p}{\partial t} + v_s \frac{\partial p}{\partial x} = D_t \frac{\partial^2 p}{\partial x^2}$$

which has the analytic solution:

$$p(x,t) = \frac{1}{\sqrt{4\pi D_t t}} \exp\!\left(-\frac{(x - v_s t)^2}{4 D_t t}\right)$$

---

## Repository Structure

```
PINN/
├── Makefile                        # Top-level build targets
├── src/
│   ├── Simulation_Cpp/             # Langevin dynamics simulator (C++17 / OpenMP)
│   │   ├── code/
│   │   │   ├── LE_1D_confine.cpp   # Main simulation entry point
│   │   │   ├── parameter.txt       # Simulation parameters
│   │   │   ├── Makefile            # C++ build rules
│   │   │   └── headers/            # C++ header files
│   │   ├── data/                   # Simulation output (simulation.bin / .csv)
│   │   └── Dockerfile              # Ubuntu 24.04 image for reproducible builds
│   └── Neural_Network_python/      # Python ML pipeline (TensorFlow / Keras)
│       ├── code/
│       │   ├── main.py             # Orchestration entry point
│       │   ├── settings.toml       # Central configuration (all tunable parameters)
│       │   └── modules/
│       │       ├── data_preparation.py  # Binary → Parquet conversion
│       │       ├── data_analytic.py     # Analytic Fokker–Planck solution
│       │       └── neural_network.py    # NN model, propagator, plotting
│       └── pyproject.toml          # Python project & dependency spec (uv)
```

---

## Quick Start

### Requirements

| Tool | Version |
|------|---------|
| clang++ (macOS) / g++ (Linux) | ≥ 17 |
| OpenMP | via libomp (Homebrew) or libgomp |
| Python | 3.11 |
| uv | ≥ 0.7 |

### 1 — Build and run the C++ simulation

```bash
# Compile (macOS with Homebrew libomp)
make sim-build

# Run — writes src/Simulation_Cpp/data/simulation.bin (~38 MB)
make sim-run
```

### 2 — Install Python dependencies

```bash
cd src/Neural_Network_python
uv sync
```

### 3 — Run the full ML pipeline

```bash
cd src/Neural_Network_python/code
uv run python main.py
```

Figures are written to `code/figures/` as both `.html` (interactive) and `.png` (static).

---

## Configuration

All tunable parameters live in [`src/Neural_Network_python/code/settings.toml`](src/Neural_Network_python/code/settings.toml):

| Section | Key | Description |
|---------|-----|-------------|
| `[analytic]` | `x_min`, `x_max`, `x_step` | Spatial domain for the analytic solution |
| `[network]` | `n_bins` | Histogram bins (coarse NN grid) |
| `[network]` | `train_test_split` | Fraction of time-steps used for training |
| `[training]` | `epochs_nn` | Max epochs for the PDF predictor |
| `[training]` | `epochs_propagator` | Max epochs for the Fokker–Planck propagator |
| `[training]` | `multi_step` | Max prediction-gap (steps) during propagator training |
| `[physics]` | `vs`, `Dt`, `delta`, `timestep` | Physical parameters — keep in sync with `parameter.txt` |
| `[physics]` | `weight` | Fokker–Planck residual weight $\lambda$ |

---

## Docker / Colima (optional)

A `Dockerfile` is provided for the C++ simulator (useful on Linux CI or when
the host compiler is unavailable):

```bash
# Start Colima (macOS) — only needed once per session
make colima-start

# Build Docker image
make docker-build

# Run simulation inside container → writes simulation.bin to the host's data/ dir
make docker-run
```

---

## Project Layout — Machine Learning

```
data_preparation  →  simulation.bin  →  prepdata_from_binary.parquet
data_analytic     →  analytic_data.parquet   (Fokker–Planck Gaussian)
neural_network    →  PDF predictor  (histogram + time → PDF)
                  →  FP propagator  (histogram_t + Δt → histogram_{t+Δt})
rollout           →  autoregressive future PDF prediction
```

