import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_SETTINGS_PATH = Path(__file__).parent.parent / "settings.toml"
with open(_SETTINGS_PATH, "rb") as _f:
    _SETTINGS = tomllib.load(_f)


@dataclass(slots=True)
class Analytic:
    data: pd.DataFrame
    x: np.ndarray = field(init=False)
    scaling_time: float = field(init=False)
    drift: float = field(init=False)
    diffusion: float = field(init=False)

    def __post_init__(self):
        cfg = _SETTINGS["analytic"]
        self.x = np.arange(cfg["x_min"], cfg["x_max"], cfg["x_step"])
        cfg_p = _SETTINGS["physics"]
        self.scaling_time = float(cfg_p["delta"])
        self.drift = float(cfg_p["vs"])
        self.diffusion = float(cfg_p["Dt"])

    def analytic(self) -> pd.DataFrame:
        rows = [self._funct(i) for i in self.data["time"] if i > 0]
        list_array = np.array(rows)
        analytic_pd = pd.DataFrame(list_array, self.data["time"][1:])
        analytic_pd.columns = pd.Index(self.x)
        return analytic_pd

    def _funct(self, t: float):
        t = t * self.scaling_time
        factor = 1.0 / np.sqrt(4.0 * np.pi * self.diffusion * t)
        function = np.exp(
            -((self.x - self.drift * t) ** 2) / (4.0 * self.diffusion * t)
        )
        # To avoid e-239, precision of the exponetial, put it to 0 below 1e-9
        function[function < 1e-9] = 0.0
        return factor * function

    def comparison(self, time: int) -> None:
        new = self.data.loc[:, self.data.columns != "time"]
        positions = np.asarray(new.iloc[time].values, dtype=np.float64)
        counts, bin_edges = np.histogram(positions, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        a = self._funct(self.data["time"][time])
        t_phys = self.data["time"][time] * self.scaling_time
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=counts, name="Simulation", opacity=0.6))
        fig.add_trace(go.Scatter(x=self.x, y=a, name=f"Analytic, t={t_phys:.4f}"))
        fig.update_layout(xaxis_title="x", yaxis_title="Density", bargap=0)
        fig.show()

    def save_data(self, df: pd.DataFrame) -> None:
        df.to_parquet("./data/analytic_data.parquet")

    def read_data(self) -> pd.DataFrame:
        df = pd.read_parquet("./data/analytic_data.parquet", engine="pyarrow")
        return df
