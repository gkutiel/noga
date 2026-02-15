from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path("plots")


def demand_vs_temp():
    data = pd.read_csv("data/data.csv")
    data = data[data['year'] == 2023]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data['temperature_c_Jerusalem'],
        data['actual-demand'],
        color="#3b82f6",
        alpha=0.85)

    plt.title("Actual Demand vs Temperature (Jerusalem, 2023)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Actual Demand (MW)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ec_vs_temperature.png", dpi=150)
    plt.close()


def demand_by_time():
    data = pd.read_csv("data/data.csv")
    data = data[data['year'] == 2023]
    data = data[data['hour'] == 8]
    data = data.sort_values(['month', 'day', 'hour'])

    plt.figure(figsize=(10, 6))
    plt.scatter(
        data.index,
        data['actual-demand'],
        s=2,
        color="#3b82f6",
        alpha=0.85)

    plt.title("Average Actual Demand by Time (2023)")
    plt.xlabel("Time")
    plt.ylabel("Demand (MW)")
    # plt.xticks(data.index)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "demand_by_time.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    demand_vs_temp()
    demand_by_time()
