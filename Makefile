NN_DIR = src/Neural_Network_python

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