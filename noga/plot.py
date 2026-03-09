from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path("plots")
CITY = Literal["Jerusalem", "Haifa", "Tel Aviv"]


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


def daily_demand_by_time():
    daily = pd.read_csv("data/daily.csv")
    daily["date"] = pd.to_datetime(daily["date"], format="%d-%m-%Y")
    daily = daily.sort_values("date")

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Scatter: total demand
    ax1.scatter(
        daily["date"],
        daily["total_demand"],
        s=8,
        color="#3b82f6",
        alpha=0.6,
        label="Total Demand",
        zorder=3,
    )

    # Average demand line
    avg = daily["total_demand"].mean()
    ax1.axhline(avg, color="#1d4ed8", linewidth=1.5,
                linestyle="--", label=f"Avg Demand ({avg:,.0f} MW)")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Total Demand (MW)", color="#3b82f6")
    ax1.tick_params(axis="y", labelcolor="#3b82f6")

    # Second y-axis: temperatures
    ax2 = ax1.twinx()
    temp_series = {
        "Tel Aviv":  ("temperature_Tel_Aviv",  "#a855f7"),
    }
    for city, (col, color) in temp_series.items():
        ax2.plot(
            daily["date"],
            daily[col],
            linewidth=1,
            color=color,
            alpha=0.8,
            label=f"Temp {city}",
        )

    ax2.set_ylabel("Temperature (°C)")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", fontsize=8)

    plt.title("Daily Total Demand and Temperature over Time")
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "daily_demand_by_time.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # demand_vs_temp()
    # demand_by_time()
    daily_demand_by_time()
