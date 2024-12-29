from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl


@dataclass(slots=True)
class PrepData:
    simulation_path: str
    file_name: str

    def preparation_slow(self) -> None:
        polardata = pl.read_csv(
            self.simulation_path + "/Simulation_Cpp/data/simulation.csv"
        )
        # Get all the different particles names
        tags = polardata["Particles"].arr.explode().unique().to_list()
        # Sort tags
        tags.sort()
        # Get time
        time0 = polardata.filter((pl.col("Particles") == tags[0]))
        time0.drop_in_place("Particles")
        time0.drop_in_place("x-position")
        for i in tags:
            particles = polardata.filter((pl.col("Particles") == i))
            particle = particles.rename({"x-position": i})
            particle.drop_in_place("Particles")
            particle.drop_in_place("time")
            time0 = time0.with_columns(particle[i].alias(i))
        # save file to parquet
        time0.write_parquet("./data/" + self.file_name + ".parquet")

    def preparation(self) -> None:
        polardata = pl.read_csv(
            self.simulation_path + "/Simulation_Cpp/data/simulation.csv"
        )
        polardata.drop_in_place("")
        polardata.write_parquet("./data/" + self.file_name + ".parquet")

    def preparation_binary(self) -> None:
        # Read binary file
        with open(
            self.simulation_path + "/Simulation_Cpp/data/simulation.bin", "rb"
        ) as f:
            file_binary = f.read()
        # Read parameter file to get the total number of particles
        with open(
            self.simulation_path + "/Simulation_Cpp/code/parameter.txt", "r"
        ) as file_data:
            for line in file_data:
                parameters = line.split()
        N_particles = int(parameters[1])
        # Preparing the type
        list_type = []
        list_type.append(("time", "int32"))
        for i in range(N_particles):
            list_type.append(("Particle" + str(i), "float64"))
        dt = np.dtype(list_type)
        np_data = np.frombuffer(file_binary, dt)
        df = pd.DataFrame(np_data)
        df.to_parquet(
            "./data/" + self.file_name + "_from_binary.parquet", engine="pyarrow"
        )

    def readdata(self) -> pd.DataFrame:
        df = pd.read_parquet("./data/" + self.file_name + ".parquet", engine="pyarrow")
        return df

    def readdata_binary(self) -> pd.DataFrame:
        df = pd.read_parquet(
            "./data/" + self.file_name + "_from_binary.parquet", engine="pyarrow"
        )
        return df
