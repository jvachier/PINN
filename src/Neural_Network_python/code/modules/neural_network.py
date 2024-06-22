import os.path
from dataclasses import dataclass
from typing import Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import (
    Dense,
)
from keras.models import Sequential
from keras.optimizers.legacy import (
    Adam,
)


@dataclass(slots=True)
class NN:
    df_ana: pd.DataFrame = None
    df_sim: pd.DataFrame = None

    df_ana_processing: pd.DataFrame = None
    df_sim_processing: pd.DataFrame = None

    df_ana_train: pd.DataFrame = None
    df_ana_test: pd.DataFrame = None
    df_sim_processing_train: pd.DataFrame = None
    df_sim_processing_test: pd.DataFrame = None

    n_xtrain: int = None
    m_xtrain: int = None
    n_ytrain: int = None
    m_ytrain: int = None

    def __post_init__(self):
        self.df_ana = pd.read_parquet("./data/analytic_data.parquet", engine="pyarrow")
        self.df_sim = pd.read_parquet("./data/prepdata.parquet", engine="pyarrow")

    def data_prep(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_ana_processing = self.df_ana.copy()
        df_ana_processing = df_ana_processing.rename_axis("time").reset_index()
        df_ana_processing = df_ana_processing.drop(columns="time")

        df_sim_processing = self.df_sim.copy()
        df_sim_processing = df_sim_processing.iloc[1:]

        (
            df_sim_processing_train,
            df_sim_processing_test,
            df_ana_train,
            df_ana_test,
        ) = self.train_test_data(df_ana_processing, df_sim_processing)

        test = df_ana_train.copy()
        df_ana_train = test._append([test] * 7, ignore_index=True)

        df_sim_processing_1 = df_sim_processing_train.iloc[:, :10000]
        df_sim_processing_2 = df_sim_processing_train.iloc[:, 10000:20000]
        df_sim_processing_3 = df_sim_processing_train.iloc[:, 20000:30000]
        df_sim_processing_4 = df_sim_processing_train.iloc[:, 30000:40000]
        df_sim_processing_5 = df_sim_processing_train.iloc[:, 40000:50000]
        df_sim_processing_6 = df_sim_processing_train.iloc[:, 50000:60000]
        df_sim_processing_7 = df_sim_processing_train.iloc[:, 60000:70000]
        df_sim_processing_8 = df_sim_processing_train.iloc[:, 70000:80000]

        df_sim_processing_2.columns = df_sim_processing_1.columns.values
        df_sim_processing_3.columns = df_sim_processing_1.columns.values
        df_sim_processing_4.columns = df_sim_processing_1.columns.values
        df_sim_processing_5.columns = df_sim_processing_1.columns.values
        df_sim_processing_6.columns = df_sim_processing_1.columns.values
        df_sim_processing_7.columns = df_sim_processing_1.columns.values
        df_sim_processing_8.columns = df_sim_processing_1.columns.values

        df_sim_processing_2["time"] = df_sim_processing_1.time
        df_sim_processing_3["time"] = df_sim_processing_1.time
        df_sim_processing_4["time"] = df_sim_processing_1.time
        df_sim_processing_5["time"] = df_sim_processing_1.time
        df_sim_processing_6["time"] = df_sim_processing_1.time
        df_sim_processing_7["time"] = df_sim_processing_1.time
        df_sim_processing_8["time"] = df_sim_processing_1.time

        concat_1 = pd.concat([df_sim_processing_1, df_sim_processing_2])
        concat_2 = pd.concat([concat_1, df_sim_processing_3])
        concat_3 = pd.concat([concat_2, df_sim_processing_4])
        concat_4 = pd.concat([concat_3, df_sim_processing_5])
        concat_5 = pd.concat([concat_4, df_sim_processing_6])
        concat_6 = pd.concat([concat_5, df_sim_processing_7])
        concat_7 = pd.concat([concat_6, df_sim_processing_8])

        df_sim_processing_train = concat_7

        return (
            df_ana_train,
            df_ana_test,
            df_sim_processing_train,
            df_sim_processing_test,
        )

    def train_test_data(
        self, df_ana_processing: pd.DataFrame, df_sim_processing: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_ana_train = df_ana_processing[0 : int(0.9 * len(df_ana_processing))]
        df_ana_test = df_ana_processing[int(0.9 * len(df_ana_processing)) :]

        df_sim_processing_train = df_sim_processing[
            0 : int(0.9 * len(df_sim_processing))
        ]
        df_sim_processing_test = df_sim_processing[int(0.9 * len(df_sim_processing)) :]
        return (
            df_sim_processing_train,
            df_sim_processing_test,
            df_ana_train,
            df_ana_test,
        )

    def shape_data(
        self, df_ana_train: pd.DataFrame, df_sim_processing_train: pd.DataFrame
    ) -> Tuple[int, int, int, int]:
        n_xtrain, m_xtrain = df_sim_processing_train.T.shape
        n_ytrain, m_ytrain = df_ana_train.T.shape
        return n_xtrain, m_xtrain, n_ytrain, m_ytrain

    def nn_model(
        self,
        df_ana_train: pd.DataFrame,
        df_sim_processing_train: pd.DataFrame,
        n_xtrain: int,
        n_ytrain: int,
        epoch: int,
    ) -> object:
        modell_nn = Sequential()
        modell_nn.add(Dense(units=1024, activation="gelu", input_shape=(n_xtrain,)))
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=256, activation="gelu"))
        modell_nn.add(Dense(units=128, activation="gelu"))
        modell_nn.add(Dense(units=64, activation="gelu"))
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=1024, activation="gelu"))
        modell_nn.add(Dense(n_ytrain, activation="tanh"))
        modell_nn.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        modell_nn.summary()
        modell_nn.fit(
            df_sim_processing_train.values,
            df_ana_train.values,
            epochs=epoch,
            verbose=2,
            validation_split=0.1,
        )
        evaluation = modell_nn.evaluate(
            df_sim_processing_train.values,
            df_ana_train.values,
            verbose=0,
        )
        print(evaluation)
        if os.path.isfile("./keras_model/modell_nn.keras") is False:
            modell_nn.save("./keras_model/modell_nn.keras")
        return modell_nn

    def predict_model_test(
        self, modell_nn: object, df_sim_processing_test: pd.DataFrame
    ) -> np.ndarray:
        return modell_nn.predict(df_sim_processing_test.iloc[:, :10000].values)

    def predict_model_train(
        self, modell_nn: object, df_sim_processing_train: pd.DataFrame
    ) -> np.ndarray:
        return modell_nn.predict(df_sim_processing_train.values)

    def comparison_nn_sim_ana_test(
        self,
        a: np.ndarray,
        time: int,
        df_sim_processing_test: pd.DataFrame,
        df_ana_test: pd.DataFrame,
    ) -> None:
        df_sim_test = df_sim_processing_test.drop(columns="time")
        plt.plot(df_ana_test.columns, df_ana_test.iloc[time], label="Ana")
        df_sim_test.iloc[time].hist(bins=100, density=True, label="Sim")
        plt.plot(df_ana_test.columns, a[time], "r", label="NN")
        plt.legend()
        plt.savefig("./figures/comparison_nn_sim_ana_without_loss_physics_test.png")
        plt.show()

    def comparison_nn_sim_ana_train(
        self,
        a: np.ndarray,
        time: int,
        df_sim_processing_train: pd.DataFrame,
        df_ana_train: pd.DataFrame,
    ) -> None:
        df_sim_train = df_sim_processing_train.drop(columns="time")
        plt.plot(df_ana_train.columns, df_ana_train.iloc[time], label="Ana")
        df_sim_train.iloc[time].hist(bins=100, density=True, label="Sim")
        plt.plot(df_ana_train.columns, a[time], "r", label="NN")
        plt.legend()
        plt.savefig("./figures/comparison_nn_sim_ana_without_loss_physics_train.png")
        plt.show()
