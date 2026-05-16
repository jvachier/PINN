import logging
import tomllib
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model
from modules import data_analytic, data_preparation, neural_network

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

_SETTINGS_PATH = Path(__file__).parent / "settings.toml"
with open(_SETTINGS_PATH, "rb") as _f:
    _SETTINGS = tomllib.load(_f)

tf.config.set_soft_device_placement(True)

binary = _SETTINGS["data"]["use_binary"]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--comparison", action="store_true")

    args = parser.parse_args()

    # Getting the correct path to the data
    three_up = Path(__file__).parent.parent.parent.parent
    data = data_preparation.PrepData(str(three_up), "prepdata")

    if binary:
        if not Path("./data/prepdata_from_binary.parquet").is_file():
            data.preparation_binary()
        df_simulation = data.readdata_binary()
    else:
        if not Path("./data/prepdata.parquet").is_file():
            data.preparation()
        df_simulation = data.readdata()

    # Analytical results
    analytic = data_analytic.Analytic(df_simulation)

    if not Path("./data/analytic_data.parquet").is_file():
        df_analytic = analytic.analytic()
        analytic.save_data(df_analytic)
    else:
        df_analytic = analytic.read_data()
    if args.comparison:
        analytic.comparison(5)

    # Neural Network
    nn = neural_network.NN()
    (
        df_ana_train,
        df_ana_test,
        df_sim_processing_train,
        df_sim_processing_test,
    ) = nn.data_prep()
    with tf.device("/device:GPU:0"):
        train_pred_path = Path("./predictions/predict_model_train.npy")
        test_pred_path = Path("./predictions/predict_model_test.npy")
        if not Path("./keras_model/modell_nn.keras").is_file():
            model_nn = nn.nn_model()
            predict_model_test = nn.predict_model_test(modell_nn=model_nn)
            predict_model_train = nn.predict_model_train(modell_nn=model_nn)
            np.save(train_pred_path, predict_model_train)
            np.save(test_pred_path, predict_model_test)
        else:
            model_nn = load_model("./keras_model/modell_nn.keras", compile=False)
            predict_model_train = np.load(train_pred_path)
            predict_model_test = np.load(test_pred_path)

    mid_train = len(df_sim_processing_train) // 2
    mid_test = max(0, len(df_sim_processing_test) // 2)
    nn.comparison_nn_sim_ana_train(
        a=predict_model_train,
        time=mid_train,
        df_sim_processing_train=df_sim_processing_train,
        df_ana_train=df_ana_train,
    )
    nn.comparison_nn_sim_ana_test(
        a=predict_model_test,
        time=mid_test,
        df_sim_processing_test=df_sim_processing_test,
        df_ana_test=df_ana_test,
    )

    # ------------------------------------------------------------------
    # Propagator: train on (hist_t, dt) -> hist_{t+dt} pairs,
    # then roll out from the last *short-run* step without the C++ code.
    # ------------------------------------------------------------------
    # Propagator: train on consecutive analytic pairs so the temporal signal is
    # noise-free.  Simulation histograms have O(1/sqrt(N)) shot noise per bin
    # which can swamp the small step-to-step PDF change (especially at late
    # times when the Gaussian is broad and nearly static).
    # Use ALL available analytic rows so the model sees the full time range.
    df_sim_chrono = nn.df_sim.iloc[1:].copy()  # time axis; matches df_ana rows
    df_ana_chrono = nn.df_ana  # noise-free PDFs for all saved time steps

    nn.build_propagator_dataset(
        df_sim_chrono,
        multi_step=_SETTINGS["training"]["multi_step"],
        df_ana=df_ana_chrono,  # use analytic PDFs instead of simulation histograms
    )

    with tf.device("/device:GPU:0"):
        if not Path("./keras_model/modell_propagator.keras").is_file():
            model_prop = nn.nn_model_propagator()
        else:
            model_prop = load_model(
                "./keras_model/modell_propagator.keras", compile=False
            )

    # Seed the rollout from ~25% of the timeline so we can observe clear
    # temporal evolution (drift + broadening) over a long horizon.
    # e.g. at τ ≈ 2.5 the Gaussian is narrow at x≈2.5; by τ=10 it is broad at x≈10.
    seed_idx = int(len(df_ana_chrono) * _SETTINGS["training"]["rollout_seed_frac"])
    x_fine = df_ana_chrono.columns.values.astype(np.float64)
    hist_seed = np.interp(
        nn.x_grid, x_fine, df_ana_chrono.iloc[seed_idx].values
    ).astype(np.float32)

    # Roll forward from seed_idx+1 to the end of the analytic dataset.
    start_step = seed_idx + 1  # first predicted row index in df_analytic
    n_future_steps = len(df_analytic) - start_step
    predictions = nn.rollout(model_prop, hist_seed, n_steps=n_future_steps)

    nn.comparison_propagator(
        predictions=predictions,
        df_ana=df_analytic,
        start_step=start_step,
        n_steps=_SETTINGS["training"]["comparison_n_steps"],
    )


if __name__ == "__main__":
    main()
