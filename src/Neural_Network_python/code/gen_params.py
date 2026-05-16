"""Generate Simulation_Cpp/code/parameter.txt from settings.toml.

This script is the single point of synchronisation between the Python
configuration and the C++ simulation.  Run it whenever you change any
value in the [physics] section of settings.toml:

    make gen-params          # from the repo root
    uv run python gen_params.py  # from the code/ directory

The parameter.txt format expected by LE_1D_confine.cpp is:
    delta  Particles  Dt  vs  Wall  total_time  timestep
"""

import logging
import tomllib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_SETTINGS_PATH = Path(__file__).parent / "settings.toml"
_PARAM_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "src"
    / "Simulation_Cpp"
    / "code"
    / "parameter.txt"
)


def main() -> None:
    with open(_SETTINGS_PATH, "rb") as f:
        cfg = tomllib.load(f)

    p = cfg["physics"]
    line = (
        f"{p['delta']}\t"
        f"{p['particles']}\t"
        f"{p['Dt']}\t"
        f"{p['vs']}\t"
        f"{p['wall']}\t"
        f"{p['total_time']}\t"
        f"{p['timestep']}\n"
    )
    _PARAM_PATH.write_text(line)
    logging.info("Written: %s", _PARAM_PATH)
    logging.info("Content: %s", line.strip())


if __name__ == "__main__":
    main()
