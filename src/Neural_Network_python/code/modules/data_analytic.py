from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(slots=True)
class Analytic:
    parameter_path: str
    data: pd.DataFrame
    x: np.array = None
    scaling_time: float = None
    drift: float = None
    diffusion: float = None

    def __post_init__(self):
        self.x = np.arange(-10.0, 10.0, 0.01)
        with open(
            self.parameter_path + "/Simulation_Cpp/code/parameter.txt", "r"
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
        plt.figure()
        new = self.data.loc[:, self.data.columns != "time"]
        new.iloc[time].hist(bins=100, density=True, label="Simulation")
        a = self._funct(self.data["time"][time])
        plt.plot(
            self.x,
            a,
            label="Analytic, $t=$" + str(self.data["time"][time] * self.scaling_time),
        )
        plt.legend()
        plt.show()

    def save_data(self, df: pd.DataFrame) -> None:
        df.to_parquet("./data/analytic_data.parquet")

    def read_data(self) -> pd.DataFrame:
        df = pd.read_parquet("./data/analytic_data.parquet", engine="pyarrow")
        return df
