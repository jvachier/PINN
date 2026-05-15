# ── Build stage ───────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++-14 \
        libomp-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY code/ .

# Override compiler vars: the local Makefile targets macOS clang++/libomp,
# but inside Ubuntu we use g++-14 whose -fopenmp is self-contained.
RUN make all \
      CC=g++-14 \
      CFLAGS="-O3 -std=c++17 -march=native -Wall -g -fopenmp" \
      LDFLAGS="-fopenmp"

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libomp5-14 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /simulation

# Copy the compiled binary and the parameter file
COPY --from=builder /build/LE_1D_confine.out .
COPY code/parameter.txt .

# The simulation binary runs with CWD=/simulation and writes ../data/simulation.bin,
# which resolves to /data inside the container.
# Mount the host output directory to /data at runtime:
#   docker run -v /host/path/data:/data pinn-simulation
RUN mkdir -p /data

# Default entrypoint runs the simulation; override CMD to pass different flags.
ENTRYPOINT ["./LE_1D_confine.out"]
