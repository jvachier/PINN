import os.path as path
from argparse import ArgumentParser

import tensorflow as tf
from modules import data_analytic, data_preparation, neural_network

tf.config.set_soft_device_placement(True)


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
    model_nn = nn.nn_model()
    with tf.device("/device:GPU:0"):
        nn.fit_evaluate(model_nn, 50)
        fit_model_test = nn.fit_model_test(model_nn)
        fit_model_train = nn.fit_model_train(model_nn)
    nn.comparison_nn_sim_ana_test(fit_model_test, 10)
    nn.comparison_nn_sim_ana_train(fit_model_train, 10)


if __name__ == "__main__":
    main()
