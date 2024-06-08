from dataclasses import dataclass

import pandas as pd
import polars as pl


@dataclass(slots=True)
class PrepData:
    simulation_path: str
    file_name: str

    # def preparation(self) -> None:
    #     polardata = pl.read_csv(
    #         self.simulation_path + "/Simulation_Cpp/data/simulation.csv"
    #     )
    #     # Get all the different particles names
    #     tags = polardata["Particles"].arr.explode().unique().to_list()
    #     # Sort tags
    #     tags.sort()
    #     # Get time
    #     time0 = polardata.filter((pl.col("Particles") == tags[0]))
    #     time0.drop_in_place("Particles")
    #     time0.drop_in_place("x-position")
    #     for i in tags:
    #         particles = polardata.filter((pl.col("Particles") == i))
    #         particle = particles.rename({"x-position": i})
    #         particle.drop_in_place("Particles")
    #         particle.drop_in_place("time")
    #         time0 = time0.with_columns(particle[i].alias(i))
    #     # save file to parquet
    #     time0.write_parquet("./data/" + self.file_name + ".parquet")

    def preparation(self) -> None:
        polardata = pl.read_csv(
            self.simulation_path + "/Simulation_Cpp/data/simulation.csv"
        )
        polardata.drop_in_place("")
        polardata.write_parquet("./data/" + self.file_name + ".parquet")

    def readdata(self) -> pd.DataFrame:
        df = pd.read_parquet("./data/" + self.file_name + ".parquet", engine="pyarrow")
        return df
