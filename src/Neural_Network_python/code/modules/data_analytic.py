import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_SETTINGS_PATH = Path(__file__).parent.parent / "settings.toml"
with open(_SETTINGS_PATH, "rb") as _f:
    _SETTINGS = tomllib.load(_f)


@dataclass(slots=True)
class Analytic:
    parameter_path: str
    data: pd.DataFrame
    x: np.array = None
    scaling_time: float = None
    drift: float = None
    diffusion: float = None

    def __post_init__(self):
        cfg = _SETTINGS["analytic"]
        self.x = np.arange(cfg["x_min"], cfg["x_max"], cfg["x_step"])
        with open(
            self.parameter_path + "/Simulation_Cpp/code/parameter.txt"
        ) as file_data:
            for line in file_data:
                parameters = line.split()
        self.scaling_time = float(parameters[0])
        self.drift = float(parameters[3])
        self.diffusion = float(parameters[2])

    def analytic(self) -> pd.DataFrame:
        list = []
        for i in self.data["time"]:
            if i > 0:
                a = self._funct(i)
                list.append(a)
        list_array = np.array(list)
        analytic_pd = pd.DataFrame(list_array, self.data["time"][1:])
        analytic_pd.columns = self.x
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
        positions = new.iloc[time].values
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
