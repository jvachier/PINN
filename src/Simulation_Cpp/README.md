# Langevin Dynamics Simulator (C++)

A multi-threaded C++17 simulation of a 1D Langevin equation, discretised with the
Euler–Maruyama scheme. Particle positions are written to a compact binary file
read downstream by the Python ML pipeline.

---

## Physics

The equation of motion for particle $i$ is:

$$x_i(t + \delta) = x_i(t) + v_s \, \delta + \sqrt{2 D_t \, \delta} \, \zeta_i(t)$$

where:

| Symbol | Name | Value |
|--------|------|-------|
| $\delta$ | integration time-step | `1e-3` |
| $v_s$ | drift velocity | `1.0` |
| $D_t$ | translational diffusion | `1.0` |
| $N$ | number of particles | `10000` |
| $L$ | half-box size (confinement wall) | `300.0` |
| Total steps | simulation steps | `10000` |
| `timestep` | steps between saved snapshots | `10` |

Parameters are stored in [`code/parameter.txt`](code/parameter.txt) as a single
space-separated line:

```
delta  total_time  Dt  vs  Wall  total_steps  timestep
```

---

## Output Format

When `Type_Bin = true` (default) the simulator writes `data/simulation.bin`:

| Field | Type | Size |
|-------|------|------|
| Timestamp (integer step) | `int32` | 4 bytes |
| $N$ particle positions | `float32` × N | 4N bytes |

One record per saved snapshot → `(total_steps / timestep) + 1` records (including $t=0$).
Approximate file size: `(1 + N) × 4 × (total_steps / timestep + 1)` bytes ≈ **38 MB**
with the default parameters.

---

## Compilation

### macOS (Homebrew clang + libomp)

```bash
brew install llvm libomp
cd code
make          # uses CC=clang++ as set in the Makefile
```

### Linux (GCC)

```bash
cd code
make CC=g++ LDFLAGS="-fopenmp"
```

### Docker (Ubuntu 24.04)

```bash
# from the repository root
make docker-build
make docker-run
```

The container mounts `data/` so the output binary is available on the host after the run.

---

## Running

```bash
./LE_1D_confine.o
```

The binary is written to `../data/simulation.bin` relative to the `code/` directory.

---

## Source Files

| File | Purpose |
|------|---------|
| `LE_1D_confine.cpp` | Main loop — initialises particles, runs Euler–Maruyama, writes output |
| `initialization.cpp` | Place all particles at $x=0$ |
| `update_position.cpp` | One Euler–Maruyama step (OpenMP parallel, per-thread RNG) |
| `periodic_boundary_conditions.cpp` | Reflective confinement within $[-L, L]$ |
| `check_nooverlap.cpp` | Optional overlap check |
| `print_file.cpp` | CSV and binary writers |

