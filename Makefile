NN_DIR  = src/Neural_Network_python
SIM_DIR = src/Simulation_Cpp

# ── C++ simulation ────────────────────────────────────────────────────────────
sim-build:
	make -C $(SIM_DIR)/code all

sim-run:
	cd $(SIM_DIR)/code && ./LE_1D_confine.out

sim-clean:
	make -C $(SIM_DIR)/code clean

# ── Docker / Colima (C++ simulation) ─────────────────────────────────────────
# Requires Colima: brew install colima docker
# Start Colima before building/running (ARM64 VM, 4 CPU, 8 GB RAM):
colima-start:
	colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta

# Build the image. Colima sets the Docker context automatically after start.
docker-build:
	docker build -t pinn-simulation $(SIM_DIR)

# Run the simulation. The binary writes ../data/simulation.bin relative to its
# CWD (/simulation), which resolves to /data inside the container — mount there.
docker-run:
	docker run --rm \
	  -v "$(PWD)/$(SIM_DIR)/data:/data" \
	  pinn-simulation

# ── Python environment ────────────────────────────────────────────────────────
install:
	uv sync --directory $(NN_DIR)

install-dev:
	uv sync --directory $(NN_DIR) --group dev

lock:
	uv lock --directory $(NN_DIR)

# ── Pre-commit ────────────────────────────────────────────────────────────────
pre-commit-install:
	uv run --directory $(NN_DIR) pre-commit install

pre-commit:
	uv run --directory $(NN_DIR) pre-commit run --all-files

# ── Linting / formatting (run individually without pre-commit) ────────────────
lint:
	uv run --directory $(NN_DIR) pylint --disable=R,C --exit-zero $(NN_DIR)/

ruff:
	uv run --directory $(NN_DIR) ruff check --fix $(NN_DIR)/
	uv run --directory $(NN_DIR) ruff format $(NN_DIR)/

black:
	uv run --directory $(NN_DIR) black $(NN_DIR)/

cpplint:
	cpplint src/Simulation_Cpp/code/*.cpp