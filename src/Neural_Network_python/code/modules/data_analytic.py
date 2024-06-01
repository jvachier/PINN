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
        self.x = np.arange(-5.0, 5.0, 0.01)
        with open(
            self.parameter_path + "/Simulation_Cpp/code/parameter.txt", "r"
        ) as file_data:
            for line in file_data:
                parameters = line.split()
        self.scaling_time = float(parameters[0])
        self.drift = float(parameters[3])
        self.diffusion = float(parameters[2])

    def analytic(self) -> None:
        for i in self.data["time"]:
            if i > 0:
                a = self._funct(i)
                # plt.plot(self.x, a)
        # plt.show()

    def _funct(self, t: float):
        t = t * self.scaling_time
        factor = 1.0 / np.sqrt(4.0 * np.pi * self.diffusion * t)
        function = np.exp(
            -((self.x - self.drift * t) ** 2) / (4.0 * self.diffusion * t)
        )
        return factor * function

    def comparison(self):
        plt.figure()
        new = self.data.loc[:, self.data.columns != "time"]
        new.iloc[500].hist(bins=100, density=True, label="Simulation")
        a = self._funct(self.data["time"][500])
        plt.plot(
            self.x,
            a,
            label="Analytic, $t=$" + str(self.data["time"][500] * self.scaling_time),
        )
        plt.legend()
        plt.show()


# @dataclass(slots=True)
# class Comparison:
