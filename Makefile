NN_DIR  = src/Neural_Network_python
SIM_DIR = src/Simulation_Cpp
DOCKER  = /opt/homebrew/bin/docker --context colima

.PHONY: sim-build sim-run sim-clean sim-test \
        colima-start docker-build docker-run \
        install install-dev lock \
        pre-commit-install pre-commit \
        lint ruff black cpplint \
        gen-params test

# ── C++ simulation ────────────────────────────────────────────────────────────
sim-build:
	$(info [sim] Building C++ simulation...)
	$(MAKE) -C $(SIM_DIR)/code all

sim-run:
	$(info [sim] Running simulation → $(SIM_DIR)/data/)
	cd $(SIM_DIR)/code && ./LE_1D_confine.out

sim-clean:
	$(info [sim] Cleaning build artefacts...)
	$(MAKE) -C $(SIM_DIR)/code clean

sim-test:
	$(info [sim] Running C++ test suite...)
	$(MAKE) -C $(SIM_DIR)/code test

# ── Docker / Colima (C++ simulation) ─────────────────────────────────────────
# Requires: brew install colima docker
colima-start:
	$(info [docker] Starting Colima VM (ARM64, 4 CPU, 8 GB)...)
	colima start --cpu 4 --memory 8 --arch aarch64 --vm-type vz --vz-rosetta

docker-build:
	$(info [docker] Building image pinn-simulation...)
	$(DOCKER) build -t pinn-simulation $(SIM_DIR)

docker-run:
	$(info [docker] Running simulation container → $(SIM_DIR)/data/)
	$(DOCKER) run --rm \
	  -v "$(PWD)/$(SIM_DIR)/data:/data" \
	  pinn-simulation

# ── Python environment ────────────────────────────────────────────────────────
install:
	$(info [python] Installing dependencies...)
	uv sync --directory $(NN_DIR)

install-dev:
	$(info [python] Installing dev dependencies...)
	uv sync --directory $(NN_DIR) --group dev

lock:
	$(info [python] Updating lockfile...)
	uv lock --directory $(NN_DIR)

# ── Pre-commit ────────────────────────────────────────────────────────────────
pre-commit-install:
	$(info [pre-commit] Installing hooks...)
	uv run --directory $(NN_DIR) pre-commit install

pre-commit:
	$(info [pre-commit] Running all hooks...)
	uv run --directory $(NN_DIR) pre-commit run --all-files

# ── Linting / formatting ──────────────────────────────────────────────────────
lint:
	$(info [lint] Running pylint...)
	uv run --directory $(NN_DIR) pylint --disable=R,C --exit-zero $(NN_DIR)/

ruff:
	$(info [lint] Running ruff check + format...)
	uv run --directory $(NN_DIR) ruff check --fix $(NN_DIR)/
	uv run --directory $(NN_DIR) ruff format $(NN_DIR)/

black:
	$(info [lint] Running black...)
	uv run --directory $(NN_DIR) black $(NN_DIR)/

cpplint:
	$(info [lint] Running cpplint...)
	cpplint src/Simulation_Cpp/code/*.cpp

# ── Parameter sync (settings.toml → parameter.txt) ───────────────────────────
# Run after changing [physics] in settings.toml, then make sim-build sim-run.
gen-params:
	$(info [config] Syncing physics parameters to C++ parameter.txt...)
	uv run --directory $(NN_DIR) python code/gen_params.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(info [test] Running Python test suite...)
	uv run --directory $(NN_DIR) pytest code/tests/ -v

sim-test:
	$(info [sim] Running C++ test suite...)
	$(MAKE) -C $(SIM_DIR)/code test