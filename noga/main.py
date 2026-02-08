import pandas as pd


def columns():
    noga = pd.read_csv("data/noga.csv")
    print(noga.columns)


if __name__ == "__main__":
    columns()
