from dataclasses import dataclass

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

    def _data_prep(self):
        self.df_ana_processing = self.df_ana.copy()
        self.df_ana_processing = self.df_ana_processing.rename_axis(
            "time"
        ).reset_index()
        self.df_ana_processing = self.df_ana_processing.drop(columns="time")

        self.df_sim_processing = self.df_sim.copy()
        self.df_sim_processing = self.df_sim_processing.iloc[1:]

        self._train_test_data()

        test = self.df_ana_train.copy()
        self.df_ana_train = test._append([test] * 4, ignore_index=True)

        df_sim_processing_1 = self.df_sim_processing_train.iloc[:, :10000]
        df_sim_processing_2 = self.df_sim_processing_train.iloc[:, 10000:20000]
        df_sim_processing_3 = self.df_sim_processing_train.iloc[:, 20000:30000]
        df_sim_processing_4 = self.df_sim_processing_train.iloc[:, 30000:40000]
        df_sim_processing_5 = self.df_sim_processing_train.iloc[:, 40000:50000]

        df_sim_processing_2.columns = df_sim_processing_1.columns.values
        df_sim_processing_3.columns = df_sim_processing_1.columns.values
        df_sim_processing_4.columns = df_sim_processing_1.columns.values
        df_sim_processing_5.columns = df_sim_processing_1.columns.values

        df_sim_processing_2["time"] = df_sim_processing_1.time
        df_sim_processing_3["time"] = df_sim_processing_1.time
        df_sim_processing_4["time"] = df_sim_processing_1.time
        df_sim_processing_5["time"] = df_sim_processing_1.time

        concat_1 = pd.concat([df_sim_processing_1, df_sim_processing_2])
        concat_2 = pd.concat([concat_1, df_sim_processing_3])
        concat_3 = pd.concat([concat_2, df_sim_processing_4])
        concat_4 = pd.concat([concat_3, df_sim_processing_5])

        self.df_sim_processing_train = concat_4

    def _train_test_data(self):
        self.df_ana_train = self.df_ana_processing[
            0 : int(0.9 * len(self.df_ana_processing))
        ]
        self.df_ana_test = self.df_ana_processing[
            int(0.9 * len(self.df_ana_processing)) :
        ]

        self.df_sim_processing_train = self.df_sim_processing[
            0 : int(0.9 * len(self.df_sim_processing))
        ]
        self.df_sim_processing_test = self.df_sim_processing[
            int(0.9 * len(self.df_sim_processing)) :
        ]

    def _shape_data(self):
        self.n_xtrain, self.m_xtrain = self.df_sim_processing_train.T.shape
        self.n_ytrain, self.m_ytrain = self.df_ana_train.T.shape

    def nn_model(self) -> object:
        self._data_prep()
        self._shape_data()
        modell_nn = Sequential()
        modell_nn.add(
            Dense(units=1024, activation="gelu", input_shape=(self.n_xtrain,))
        )
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=256, activation="gelu"))
        modell_nn.add(Dense(units=128, activation="gelu"))
        modell_nn.add(Dense(units=64, activation="gelu"))
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=1024, activation="gelu"))
        modell_nn.add(Dense(self.n_ytrain, activation="tanh"))
        modell_nn.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        modell_nn.summary()
        return modell_nn

    def fit_evaluate(self, modell_nn: object, epoch: int) -> None:
        modell_nn.fit(
            self.df_sim_processing_train.values,
            self.df_ana_train.values,
            epochs=epoch,
            verbose=2,
            validation_split=0.1,
        )
        evaluation = modell_nn.evaluate(
            self.df_sim_processing_train.values,
            self.df_ana_train.values,
            verbose=0,
        )
        print(evaluation)

    def fit_model_test(self, modell_nn: object) -> np.ndarray:
        return modell_nn.predict(self.df_sim_processing_test.iloc[:, :10000].values)

    def fit_model_train(self, modell_nn: object) -> np.ndarray:
        return modell_nn.predict(self.df_sim_processing_train.values)

    def comparison_nn_sim_ana_test(self, a: np.ndarray, time: int) -> None:
        df_sim_test = self.df_sim_processing_test.drop(columns="time")
        plt.plot(self.df_ana_test.columns, self.df_ana_test.iloc[time], label="Ana")
        df_sim_test.iloc[time].hist(bins=100, density=True, label="Sim")
        plt.plot(self.df_ana_test.columns, a[time], "r", label="NN")
        plt.legend()
        plt.savefig("./figures/comparison_nn_sim_ana_without_loss_physics_test.png")
        plt.show()

    def comparison_nn_sim_ana_train(self, a: np.ndarray, time: int) -> None:
        df_sim_train = self.df_sim_processing_train.drop(columns="time")
        plt.plot(self.df_ana_train.columns, self.df_ana_train.iloc[time], label="Ana")
        df_sim_train.iloc[time].hist(bins=100, density=True, label="Sim")
        plt.plot(self.df_ana_train.columns, a[time], "r", label="NN")
        plt.legend()
        plt.savefig("./figures/comparison_nn_sim_ana_without_loss_physics_train.png")
        plt.show()
