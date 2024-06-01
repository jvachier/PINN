import os.path as path

from modules import data_analytic, data_preparation


def main() -> None:
    # Getting the correct path to the data
    three_up = path.abspath(path.join("__file__", "../../.."))
    data = data_preparation.PrepData(three_up)
    if path.isfile("./data/prepdata.parquet") is False:
        data.preparation()
        df = data.readdata()
    else:
        df = data.readdata()

    print(df)

    analytic = data_analytic.Analytic(three_up, df)

    analytic.analytic()
    analytic.comparison()


if __name__ == "__main__":
    main()
