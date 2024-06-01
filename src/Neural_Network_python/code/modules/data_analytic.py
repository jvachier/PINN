from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class Analytic:
    parameter_path: str
    data: pd.DataFrame

    def analytic(self) -> None:
        with open(
            self.parameter_path + "/Simulation_Cpp/code/parameter.txt", "r"
        ) as file_data:
            for line in file_data:
                data = line.split()
                print(data)
        scaling_time = float(data[0])
        drift = float(data[3])
        diffusion = float(data[2])
        print(scaling_time, drift, diffusion)

    def _funct(x: np.array, t: int, drift: float, diffusion: float):
        t = t * 1e-4
        factor = 1.0 / np.sqrt(4.0 * np.pi * diffusion * t)
        function = np.exp(-((x - drift * t) ** 2) / (4.0 * diffusion * t))
        return factor * function


# @dataclass(slots=True)
# class Comparison:
