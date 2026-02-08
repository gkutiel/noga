import pandas as pd


def describe():
    noga = pd.read_csv("data/noga.csv")
    print(noga.describe())


if __name__ == "__main__":
    describe()
