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
        print(self.n_xtrain)

    def nn_model(self) -> object:
        self._data_prep()
        self._train_test_data()
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
            verbose=1,
            validation_split=0.1,
        )
        evaluation = modell_nn.evaluate(
            self.df_sim_processing_train.values,
            self.df_ana_train.values,
            verbose=0,
        )
        print(evaluation)

    def fit_model(self, modell_nn: object) -> np.ndarray:
        return modell_nn.predict(self.df_sim_processing_test.values)

    def comparison_nn_sim_ana(self, a: np.ndarray) -> None:
        df_sim_test = self.df_sim_processing_test.drop(columns="time")
        plt.plot(self.df_ana_test.columns, self.df_ana_test.iloc[40], label="Ana")
        df_sim_test.iloc[40].hist(bins=100, density=True, label="Sim")
        plt.plot(self.df_ana_test.columns, a[40], "r", label="NN")
        plt.show()
