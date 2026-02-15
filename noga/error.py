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


def cost():
    UNDER = 5
    OVER = 1
    data = pd.read_csv("data/data.csv")
    costs = []
    for reserve in [1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.2]:
        error = data['day-ahead-forecast'] * reserve - data['actual-demand']

        cost = np.where(
            error > 0,
            error * OVER,
            -error * UNDER).sum()

        costs.append((reserve, cost))

    x, y = zip(*costs)
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.xlabel("Reserve")
    plt.ylabel("Cost")
    plt.title(f"Cost vs Reserve ({UNDER}:{OVER})")
    plt.grid(True)
    plt.xticks(x)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cost_vs_reserve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # error_csv()
    # error_stat()
    cost()
