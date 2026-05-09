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

    # Preprocessed arrays produced by data_prep()
    x_grid: np.ndarray = None
    X_train: np.ndarray = None
    X_test: np.ndarray = None
    y_train: np.ndarray = None
    y_test: np.ndarray = None

    # Propagator dataset produced by build_propagator_dataset()
    X_prop: np.ndarray = None   # (n_pairs, n_bins + 1)  — hist_t concat dt_norm
    y_prop: np.ndarray = None   # (n_pairs, n_bins)      — hist_{t+dt}
    dt_scale: float = None      # raw dt between saved steps (for denormalisation)

    def __post_init__(self):
        self.df_ana = pd.read_parquet("./data/analytic_data.parquet", engine="pyarrow")
        self.df_sim = pd.read_parquet("./data/prepdata.parquet", engine="pyarrow")

    def _compute_histogram_features(
        self,
        df_sim: pd.DataFrame,
        t_min: float,
        t_max: float,
    ) -> np.ndarray:
        """
        Convert raw particle positions into a density histogram + normalised time.

        Each row in df_sim becomes one sample:
          features[:len(x_grid)]  = density histogram aligned with self.x_grid
          features[-1]            = time normalised to [0, 1]

        Returns array of shape (n_samples, len(x_grid) + 1).
        """
        dx = self.x_grid[1] - self.x_grid[0]
        bin_edges = np.concatenate(
            [
                [self.x_grid[0] - dx / 2],
                (self.x_grid[:-1] + self.x_grid[1:]) / 2,
                [self.x_grid[-1] + dx / 2],
            ]
        )
        particle_cols = [c for c in df_sim.columns if c != "time"]
        positions = df_sim[particle_cols].values.astype(np.float64)  # (n, n_particles)
        times = df_sim["time"].values.astype(np.float64)
        t_range = t_max - t_min
        t_norm = (times - t_min) / t_range if t_range > 0 else np.zeros_like(times)

        n_samples = positions.shape[0]
        n_bins = len(self.x_grid)
        features = np.empty((n_samples, n_bins + 1), dtype=np.float32)
        for i in range(n_samples):
            hist, _ = np.histogram(positions[i], bins=bin_edges, density=True)
            features[i, :-1] = hist.astype(np.float32)
        features[:, -1] = t_norm.astype(np.float32)
        return features

    def data_prep(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_ana_processing = self.df_ana.copy()
        df_ana_processing = df_ana_processing.rename_axis("time").reset_index()
        df_ana_processing = df_ana_processing.drop(columns="time")

        # Derive x-grid from analytic data columns so histogram bins align exactly
        self.x_grid = df_ana_processing.columns.values.astype(np.float64)

        df_sim_processing = self.df_sim.copy()
        df_sim_processing = df_sim_processing.iloc[1:]

        # Compute time normalisation bounds from the full (pre-split) data
        t_min = float(df_sim_processing["time"].min())
        t_max = float(df_sim_processing["time"].max())

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

        # Build histogram-based input arrays; keep DataFrames for visualisation
        self.X_train = self._compute_histogram_features(
            df_sim_processing_train, t_min, t_max
        )
        self.X_test = self._compute_histogram_features(
            df_sim_processing_test, t_min, t_max
        )
        self.y_train = df_ana_train.values.astype(np.float32)
        self.y_test = df_ana_test.values.astype(np.float32)

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

    def nn_model(self, epoch: int) -> object:
        """
        Build and train the network.

        Input  (per sample): density histogram (len(x_grid) bins) + 1 normalised time
        Output (per sample): predicted PDF evaluated on x_grid

        Design rationale
        ----------------
        * Histogram input gives the model a spatially structured signal that already
          resembles the target PDF, so the network only needs to denoise/smooth it
          and apply the temporal correction.
        * Time is given as a dedicated, normalised scalar feature so its influence
          is not drowned out by the ~7500 spatial features.
        * The symmetric (no bottleneck) architecture preserves both spatial and
          temporal information through all layers.
        * softplus output guarantees non-negative predictions (required for a PDF).
        """
        n_input = self.X_train.shape[1]   # len(x_grid) + 1
        n_output = self.y_train.shape[1]  # len(x_grid)

        modell_nn = Sequential()
        modell_nn.add(Dense(units=1024, activation="gelu", input_shape=(n_input,)))
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=256, activation="gelu"))
        modell_nn.add(Dense(units=512, activation="gelu"))
        modell_nn.add(Dense(units=1024, activation="gelu"))
        modell_nn.add(Dense(n_output, activation="softplus"))
        modell_nn.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        modell_nn.summary()
        modell_nn.fit(
            self.X_train,
            self.y_train,
            epochs=epoch,
            batch_size=16,
            verbose=2,
            validation_split=0.1,
            shuffle=True,
        )
        evaluation = modell_nn.evaluate(self.X_train, self.y_train, verbose=0)
        print(evaluation)
        if os.path.isfile("./keras_model/modell_nn.keras") is False:
            modell_nn.save("./keras_model/modell_nn.keras")
        return modell_nn

    def predict_model_test(self, modell_nn: object) -> np.ndarray:
        return modell_nn.predict(self.X_test)

    def predict_model_train(self, modell_nn: object) -> np.ndarray:
        return modell_nn.predict(self.X_train)

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

    # ------------------------------------------------------------------
    # Propagator: learn p(x, t+dt) = f( p(x, t), dt )
    # ------------------------------------------------------------------

    def build_propagator_dataset(
        self, df_sim: pd.DataFrame, multi_step: int = 1
    ) -> None:
        """
        Build (histogram_t, dt_norm) -> histogram_{t+dt} pairs.

        Parameters
        ----------
        df_sim :
            Full simulation DataFrame (all rows, including test times).
            Using *all* available data maximises the number of training pairs.
        multi_step :
            Also include pairs separated by 2, 3, ... multi_step timesteps.
            This forces the propagator to be consistent over longer horizons
            and helps it generalise beyond the training window.
        """
        assert self.x_grid is not None, "Call data_prep() first to set x_grid."

        particle_cols = [c for c in df_sim.columns if c != "time"]
        times = df_sim["time"].values.astype(np.float64)

        # Raw timestep between consecutive saved rows
        self.dt_scale = float(np.median(np.diff(times)))

        dx = self.x_grid[1] - self.x_grid[0]
        bin_edges = np.concatenate(
            [
                [self.x_grid[0] - dx / 2],
                (self.x_grid[:-1] + self.x_grid[1:]) / 2,
                [self.x_grid[-1] + dx / 2],
            ]
        )
        positions = df_sim[particle_cols].values.astype(np.float64)
        n_rows, n_bins = len(times), len(self.x_grid)

        # Precompute histograms for every saved timestep
        hists = np.empty((n_rows, n_bins), dtype=np.float32)
        for i in range(n_rows):
            h, _ = np.histogram(positions[i], bins=bin_edges, density=True)
            hists[i] = h.astype(np.float32)

        pairs_x, pairs_y = [], []
        for gap in range(1, multi_step + 1):
            dt_norm = float(gap)  # normalised by dt_scale (1 unit = 1 saved step)
            for i in range(n_rows - gap):
                pairs_x.append(np.append(hists[i], dt_norm))
                pairs_y.append(hists[i + gap])

        self.X_prop = np.array(pairs_x, dtype=np.float32)
        self.y_prop = np.array(pairs_y, dtype=np.float32)

    def nn_model_propagator(self, epoch: int = 50) -> object:
        """
        Train a one-step / multi-step PDF propagator.

        Input  per sample : [hist(x, t) | dt_norm]  shape (n_bins + 1,)
        Output per sample : hist(x, t + dt)          shape (n_bins,)

        The network learns the Fokker-Planck time-evolution operator from data.
        At inference time you feed the last known histogram and step forward
        iteratively — no further C++ runs needed.
        """
        assert self.X_prop is not None, "Call build_propagator_dataset() first."

        n_input = self.X_prop.shape[1]   # n_bins + 1
        n_output = self.y_prop.shape[1]  # n_bins

        model = Sequential()
        model.add(Dense(1024, activation="gelu", input_shape=(n_input,)))
        model.add(Dense(512, activation="gelu"))
        model.add(Dense(256, activation="gelu"))
        model.add(Dense(512, activation="gelu"))
        model.add(Dense(1024, activation="gelu"))
        model.add(Dense(n_output, activation="softplus"))
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError()],
        )
        model.summary()
        model.fit(
            self.X_prop,
            self.y_prop,
            epochs=epoch,
            batch_size=32,
            verbose=2,
            validation_split=0.1,
            shuffle=True,
        )
        if not os.path.isfile("./keras_model/modell_propagator.keras"):
            model.save("./keras_model/modell_propagator.keras")
        return model

    def rollout(
        self,
        model: object,
        hist_start: np.ndarray,
        n_steps: int,
    ) -> np.ndarray:
        """
        Autoregressively propagate a starting histogram forward in time.

        Parameters
        ----------
        model :
            Trained propagator model.
        hist_start :
            Empirical histogram at the last known timestep,
            shape (n_bins,).  Compute with _compute_histogram_features()
            and take features[i, :-1] for row i.
        n_steps :
            Number of saved-step intervals to predict forward.

        Returns
        -------
        predictions : np.ndarray, shape (n_steps, n_bins)
            Predicted PDF at each future step.
        """
        predictions = []
        h = hist_start.copy().reshape(1, -1).astype(np.float32)
        dt_norm = np.array([[1.0]], dtype=np.float32)  # always 1 step at a time
        for _ in range(n_steps):
            x_in = np.concatenate([h, dt_norm], axis=1)
            h = model.predict(x_in, verbose=0).astype(np.float32)
            predictions.append(h[0])
        return np.array(predictions)

    def comparison_propagator(
        self,
        predictions: np.ndarray,
        df_ana: pd.DataFrame,
        start_step: int,
        n_steps: int,
    ) -> None:
        """
        Overlay rollout predictions against the analytic PDF for a range of steps.

        Parameters
        ----------
        predictions : output of rollout(), shape (n_steps, n_bins).
        df_ana      : full analytic DataFrame (index = raw time, columns = x).
        start_step  : row index in df_ana corresponding to the first *predicted* step.
        n_steps     : how many steps to plot (evenly spread across predictions).
        """
        indices = np.linspace(0, len(predictions) - 1, min(n_steps, len(predictions)), dtype=int)
        fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), sharey=True)
        if len(indices) == 1:
            axes = [axes]
        for ax, idx in zip(axes, indices):
            ana_row = df_ana.iloc[start_step + idx]
            ax.plot(self.x_grid, ana_row.values, label="Analytic")
            ax.plot(self.x_grid, predictions[idx], "r--", label="Propagator")
            ax.set_title(f"step +{idx + 1}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        plt.savefig("./figures/comparison_propagator.png")
        plt.show()
