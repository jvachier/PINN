import os.path as path
from argparse import ArgumentParser

from modules import data_analytic, data_preparation


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--comparison", action="store_true")

    args = parser.parse_args()

    # Getting the correct path to the data
    three_up = path.abspath(path.join("__file__", "../../.."))
    data = data_preparation.PrepData(three_up, "prepdata")
    if path.isfile("./data/prepdata.parquet") is False:
        data.preparation()
        df = data.readdata()
    else:
        df = data.readdata()

    # Analytical results
    analytic = data_analytic.Analytic(three_up, df)

    analytic.analytic()
    if args.comparison:
        analytic.comparison(899)


if __name__ == "__main__":
    main()
