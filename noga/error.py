from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path("plots")


def error_csv():
    columns = [
        'year',
        'month',
        'day',
        'hour',
        'day-ahead-forecast',
        'actual-demand']

    data = pd.read_csv(
        "data/data.csv",
        usecols=columns)

    data['error'] = data['day-ahead-forecast'] - data['actual-demand']
    data['abs-error'] = data['error'].abs()
    data['error-percentage'] = data['abs-error'] / data['actual-demand'] * 100

    data.to_csv("data/error.csv", index=False)


def error_stat():
    data = pd.read_csv("data/error.csv")
    data = data[data['year'] == 2023]

    # BY MONTH
    by_month = data.groupby("month")
    err_by_month = by_month[["abs-error", "error-percentage"]].mean()

    plt.figure(figsize=(10, 6))
    plt.bar(
        err_by_month.index,
        err_by_month['error-percentage'],
        color="#3b82f6",
        alpha=0.85)

    plt.title("Error By Month (2023)")
    plt.xlabel("Month")
    plt.ylabel("Error (%)")
    plt.xticks(err_by_month.index)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "error_by_month.png", dpi=150)
    plt.close()

    err_by_month.to_csv("data/error_by_month.csv")


def cost(*, name: str, pred: pd.Series, actual: pd.Series, UNDER=5):
    costs = []
    for reserve in [0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2]:
        error = pred * reserve - actual

        cost = np.where(
            error > 0,
            error,
            -error * UNDER).sum()

        costs.append((reserve, cost))

    out = PLOTS_DIR / f"cost_vs_reserve_{name}.png"
    x, y = zip(*costs)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Reserve")
    plt.ylabel("Cost")
    plt.title(f"Cost vs Reserve {UNDER}:1 penalty ({name})")
    plt.grid(True)
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    print(out)


if __name__ == "__main__":
    # error_csv()
    # error_stat()
    data = pd.read_csv("data/pred.csv")
    cost(
        name='noga',
        pred=data['day-ahead-forecast'],
        actual=data['actual-demand'])

    cost(
        name='model',
        pred=data['y_hat'],
        actual=data['actual-demand'])

    data_new = pd.read_csv("data/pred-new.csv")
    cost(
        name='model-new',
        pred=data_new['y_hat'],
        actual=data_new['actual-demand'])
