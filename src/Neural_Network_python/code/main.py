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
    n_xtrain, m_xtrain, n_ytrain, m_ytrain = nn.shape_data(
        df_ana_train, df_sim_processing_train
    )
    with tf.device("/device:GPU:0"):
        if path.isfile("./keras_model/modell_nn.keras") is False:
            model_nn = nn.nn_model(
                df_ana_train=df_ana_train,
                df_sim_processing_train=df_sim_processing_train,
                n_xtrain=n_xtrain,
                n_ytrain=n_ytrain,
                epoch=50,
            )
            predict_model_test = nn.predict_model_test(
                modell_nn=model_nn, df_sim_processing_test=df_sim_processing_test
            )
            predict_model_train = nn.predict_model_train(
                modell_nn=model_nn, df_sim_processing_train=df_sim_processing_train
            )
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


if __name__ == "__main__":
    main()
