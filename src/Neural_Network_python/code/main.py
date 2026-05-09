import os.path as path
import pickle
from argparse import ArgumentParser

import tensorflow as tf
from keras.models import load_model
from modules import data_analytic, data_preparation, neural_network

tf.config.set_soft_device_placement(True)

binary = False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--comparison", action="store_true")

    args = parser.parse_args()

    # Getting the correct path to the data
    three_up = path.abspath(path.join("__file__", "../../.."))
    data = data_preparation.PrepData(three_up, "prepdata")
    if path.isfile("./data/prepdata.parquet") is False:
        data.preparation()
        df_simulation = data.readdata()
    else:
        df_simulation = data.readdata()

    # Binary
    if binary is True:
        if path.isfile("./data/prepdata_from_binary.parquet") is False:
            data.preparation_binary()
            df_simulation = data.readdata_binary()
        else:
            df_simulation = data.readdata_binary()

    # Analytical results
    analytic = data_analytic.Analytic(three_up, df_simulation)

    if path.isfile("./data/analytic_data.parquet") is False:
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
        if path.isfile("./keras_model/modell_nn.keras") is False:
            model_nn = nn.nn_model(epoch=50)
            predict_model_test = nn.predict_model_test(modell_nn=model_nn)
            predict_model_train = nn.predict_model_train(modell_nn=model_nn)
            with open("./predictions/predict_model_train.pkl", "wb") as dbfile_train:
                pickle.dump(predict_model_train, dbfile_train)

            with open("./predictions/predict_model_test.pkl", "wb") as dbfile_test:
                pickle.dump(predict_model_test, dbfile_test)
        else:
            model_nn = load_model("./keras_model/modell_nn.keras")
            with open("./predictions/predict_model_train.pkl", "rb") as dbfile_train:
                predict_model_train = pickle.load(dbfile_train)

            with open("./predictions/predict_model_test.pkl", "rb") as dbfile_test:
                predict_model_test = pickle.load(dbfile_test)

    nn.comparison_nn_sim_ana_train(
        a=predict_model_train,
        time=700,
        df_sim_processing_train=df_sim_processing_train,
        df_ana_train=df_ana_train,
    )
    nn.comparison_nn_sim_ana_test(
        a=predict_model_test,
        time=0,
        df_sim_processing_test=df_sim_processing_test,
        df_ana_test=df_ana_test,
    )

    # ------------------------------------------------------------------
    # Propagator: train on (hist_t, dt) -> hist_{t+dt} pairs,
    # then roll out from the last *short-run* step without the C++ code.
    # ------------------------------------------------------------------
    # Use the full simulation (train + test rows) to maximise training pairs.
    df_sim_full = df_sim_processing_train._append(df_sim_processing_test)

    # multi_step=3 means we train on gaps of 1, 2, and 3 saved steps,
    # which encourages longer-horizon consistency.
    nn.build_propagator_dataset(df_sim_full, multi_step=3)

    with tf.device("/device:GPU:0"):
        if path.isfile("./keras_model/modell_propagator.keras") is False:
            model_prop = nn.nn_model_propagator(epoch=50)
            with open("./predictions/model_propagator.pkl", "wb") as f:
                pickle.dump(model_prop, f)
        else:
            from keras.models import load_model as lm

            model_prop = lm("./keras_model/modell_propagator.keras")

    # Seed the rollout with the empirical histogram at the last training step.
    # nn.X_train[:, :-1] holds the histogram part (all columns except dt_norm).
    hist_seed = nn.X_train[-1, :-1]  # last histogram seen during training

    # Predict the next 20 saved steps forward in time — no C++ needed.
    n_future_steps = 20
    predictions = nn.rollout(model_prop, hist_seed, n_steps=n_future_steps)

    # The analytic solution starts at start_step = number of training rows.
    start_step = len(df_sim_processing_train)
    nn.comparison_propagator(
        predictions=predictions,
        df_ana=df_analytic,
        start_step=start_step,
        n_steps=5,  # plot 5 evenly-spaced snapshots
    )


if __name__ == "__main__":
    main()
