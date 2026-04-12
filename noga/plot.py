from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from noga.cost import loss_fns
from noga.date import DT_FRMT

PLOTS_DIR = Path("plots")
CITY = Literal["Jerusalem", "Haifa", "Tel Aviv"]


def daily_demand_by_time():
    daily = pd.read_csv("data/daily.csv")
    daily["date"] = pd.to_datetime(daily["date"], format=DT_FRMT)
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
    out = PLOTS_DIR / "daily_demand_by_time.png"

    print('Saving plot to:', out)
    plt.savefig(out, dpi=150)

    plt.close()


def demand_vs_temp():
    data = pd.read_csv("data/daily.csv")
    # data = pd.read_csv("data/data.csv")
    # data = data[data['year'] == 2023]

    for city in ["Jerusalem", "Haifa", "Tel_Aviv"]:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            data[f'temperature_{city}'],
            data['total_demand'],
            color="#3b82f6",
            s=3,
            alpha=0.85)

        plt.title(f"Actual Demand vs Temperature ({city})")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Actual Demand (MW)")
        plt.tight_layout()

        out = PLOTS_DIR / f"ec_vs_temperature_{city}.png"
        print('Saving plot to:', out)
        plt.savefig(out, dpi=150)

        plt.close()


CITY_TEMP_COL = {
    "Jerusalem": "temperature_Jerusalem",
    "Haifa":     "temperature_Haifa",
    "Tel Aviv":  "temperature_Tel_Aviv",
}

FIG_SIZE = (14, 10)


def daily_demand_vs_forecast():
    daily = pd.read_csv("data/daily.csv")
    daily["date"] = pd.to_datetime(daily["date"], format=DT_FRMT)
    daily = daily.sort_values("date")
    daily["total_demand"] = pd.to_numeric(
        daily["total_demand"], errors="coerce")
    daily["total_day_ahead_forecast"] = pd.to_numeric(
        daily["total_day_ahead_forecast"], errors="coerce")

    # a. Scatter: total_demand vs total_day_ahead_forecast
    plt.figure(figsize=FIG_SIZE)
    plt.scatter(
        daily["total_day_ahead_forecast"],
        daily["total_demand"],
        s=8,
        color="#3b82f6",
        alpha=0.6,
    )
    # y = x reference line
    lim_min = min(daily["total_day_ahead_forecast"].min(),
                  daily["total_demand"].min())
    lim_max = max(daily["total_day_ahead_forecast"].max(),
                  daily["total_demand"].max())
    plt.plot([lim_min, lim_max], [lim_min, lim_max],
             color="#ef4444", linewidth=1, linestyle="--", label="y = x")
    plt.title("Daily Total Demand vs Day-Ahead Forecast")
    plt.xlabel("Day-Ahead Forecast (MW)")
    plt.ylabel("Total Demand (MW)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "daily_demand_vs_forecast_scatter.png", dpi=150)
    plt.close()

    # b. Line plot: total_demand and total_day_ahead_forecast over time
    plt.figure(figsize=FIG_SIZE)
    plt.plot(
        daily["date"],
        daily["total_demand"],
        linewidth=1,
        color="#3b82f6",
        alpha=0.8,
        label="Total Demand",
    )
    plt.plot(
        daily["date"],
        daily["total_day_ahead_forecast"],
        linewidth=1,
        color="#ef4444",
        alpha=0.8,
        linestyle="--",
        label="Day-Ahead Forecast",
    )
    plt.title("Daily Total Demand and Day-Ahead Forecast over Time")
    plt.xlabel("Date")
    plt.ylabel("MW")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = PLOTS_DIR / "daily_demand_vs_forecast_time.png"
    print('Saving plot to:', out)
    plt.savefig(out, dpi=150)

    plt.close()


def demand_vs_forecast_kde_histogram():
    data = pd.read_csv("data/data.csv")
    data["actual-demand"] = pd.to_numeric(
        data["actual-demand"], errors="coerce")
    data["day-ahead-forecast"] = pd.to_numeric(
        data["day-ahead-forecast"], errors="coerce")
    data = data.dropna(subset=["actual-demand", "day-ahead-forecast"])

    errors = data["actual-demand"] - data["day-ahead-forecast"]
    errors = errors[(errors >= -1000) & (errors <= 1000)]

    kde = gaussian_kde(errors, bw_method="scott")
    x = np.linspace(errors.min(), errors.max(), 500)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        errors,
        bins=60,
        density=True,
        color="#3b82f6",
        alpha=0.5,
        label="Histogram",
    )
    ax.plot(
        x,
        kde(x),
        color="#1d4ed8",
        linewidth=2,
        label="KDE",
    )
    ax.axvline(0, color="#ef4444", linewidth=1.5,
               linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="#f97316", linewidth=1.5,
               linestyle="--", label=f"Mean ({errors.mean():,.0f} MW)")

    ax.set_title(
        "Distribution of Forecast Errors (Actual Demand − Day-Ahead Forecast)")
    ax.set_xlabel("Error (MW)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "demand_vs_forecast_kde_histogram.png", dpi=150)
    plt.close()


def day_ahead_forecast_abs_error():
    daily = pd.read_csv("data/daily.csv")
    daily["date"] = pd.to_datetime(daily["date"], format=DT_FRMT)
    daily = daily.sort_values("date")

    abs_error_pct = ((daily["total_demand"] - daily["total_day_ahead_forecast"]).abs()
                     / daily["total_demand"] * 100)

    plt.figure(figsize=(14, 5))
    plt.scatter(daily["date"], abs_error_pct, s=6, color="#3b82f6", alpha=0.6)
    plt.title("Absolute Error %: Day-Ahead Forecast vs Actual Demand by Date")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error (%)")
    plt.tight_layout()

    out = PLOTS_DIR / "daf_abs_error.png"
    print("Saving plot to:", out)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_loss_fns():
    import torch

    from noga.cost import loss_fns

    errors = torch.linspace(-1, 1, 500)
    zeros = torch.zeros(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, fn in loss_fns.items():
        costs = [fn(errors[i:i+1], zeros).item() for i in range(len(errors))]
        ax.plot(errors.numpy(), costs, label=name, linewidth=2)

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Loss functions (error = pred − y)")
    ax.set_xlabel("Error")
    ax.set_ylabel("Cost")
    ax.legend()
    fig.tight_layout()

    out = PLOTS_DIR / "loss_fns.png"
    print("Saving plot to:", out)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_error_kde_hist():
    # TODO: make a smooth nice histogram.

    csv_dir = Path("csv")
    names = list(loss_fns)
    fig, axes = plt.subplots(1, len(names), figsize=(
        5 * len(names), 5), sharey=True)

    for ax, name in zip(axes, names):
        df = pd.read_csv(csv_dir / f"pred_{name}.csv")
        errors = (df["pred"] - df["actual"])

        print(errors.describe())
        ax.hist(
            errors,
            bins=30,
            density=True,
            # range=(-XLIM, XLIM),
            color="#3b82f6",
            alpha=0.5,
            edgecolor="white")

        kde = gaussian_kde(errors, bw_method="scott")
        x = np.linspace(errors.min(), errors.max(), 500)
        ax.plot(x, kde(x), color="#1d4ed8", linewidth=2)

        ax.axvline(0, color="#ef4444", linewidth=1.5, linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("Error (pred − actual, MW)")
        ax.set_xlim(errors.min(), errors.max())
        # ax.xaxis.set_major_formatter(
        #     FuncFormatter(lambda v, _: f"{int(v/1000)}k" if v != 0 else "0"))

    axes[0].set_ylabel("Density")
    fig.suptitle("Prediction error distributions by model (test set)")
    fig.tight_layout()

    out = PLOTS_DIR / "error_histograms.png"
    print("Saving plot to:", out)
    plt.savefig(out, dpi=150)
    plt.close()


def plot_day_embeddings():
    from noga.cost import Name, loss_fns
    from noga.model import load_model

    DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    CITY_LABELS = ["Haifa", "Jerusalem", "Tel Aviv"]

    names: list[Name] = list(loss_fns)  # type: ignore[assignment]
    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 13))

    for col, name in enumerate(names):
        model = load_model(name)

        # --- day embeddings (7×2) ---
        day_w = model.day.weight.detach().numpy()
        ax = axes[0, col]
        ax.scatter(day_w[:, 0], day_w[:, 1], color="#3b82f6", s=60, zorder=3)
        for i, label in enumerate(DAY_LABELS):
            ax.annotate(label, (day_w[i, 0], day_w[i, 1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{name} — day")

        # --- month embeddings (12×2) ---
        month_w = model.month.weight.detach().numpy()
        ax = axes[1, col]
        ax.scatter(month_w[:, 0], month_w[:, 1],
                   color="#a855f7", s=60, zorder=3)
        for i, label in enumerate(MONTH_LABELS):
            ax.annotate(label, (month_w[i, 0], month_w[i, 1]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{name} — month")

        # --- balance temperatures ---
        balance = model.balance.detach().numpy()
        ax = axes[2, col]
        ax.bar(CITY_LABELS, balance, color="#f97316", alpha=0.8)
        ax.set_ylabel("°C")
        ax.set_title(f"{name} — balance temp")

    fig.suptitle("Learned embeddings and balance temperatures by model")
    fig.tight_layout()

    out = PLOTS_DIR / "embeddings.png"
    print("Saving plot to:", out)
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    # daily_demand_by_time()
    # demand_vs_temp()
    # demand_by_time()
    # daily_demand_vs_forecast()
    # day_ahead_forecast_abs_error()
    # demand_vs_forecast_kde_histogram()
    # plot_loss_fns()
    # plot_error_kde_hist()
    plot_day_embeddings()
