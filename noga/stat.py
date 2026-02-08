from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = Path("plots")


def mae_percent_bar_chart():
    data = pd.read_csv("data/data.csv")
    data = data[["day-ahead-forecast", "actual-demand"]].apply(
        pd.to_numeric, errors="coerce"
    )
    percent_error = (
        (data["day-ahead-forecast"] - data["actual-demand"]).abs()
        / data["actual-demand"].replace(0, pd.NA)
        * 100
    )
    percent_error = percent_error.dropna()

    bins = list(range(0, 11)) + [float("inf")]
    labels = ["1%", "2%", "3%", "4%", "5%",
              "6%", "7%", "8%", "9%", "10%", ">10%"]
    categories = pd.cut(
        percent_error,
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    counts = categories.value_counts().reindex(labels, fill_value=0)
    total = counts.sum()
    percentages = (counts / total * 100) if total else counts

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(percentages.index, percentages.values, color="#3b82f6", alpha=0.85)
    plt.title("Percentage Absolute Error (Binned)")
    plt.xlabel("Absolute Percentage Error")
    plt.ylabel("Percentage of Total (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mae_percent_bar_chart.png", dpi=150)
    plt.close()


def mae_bar_chart() -> None:
    data = pd.read_csv("data/data.csv")
    data = data[["day-ahead-forecast", "actual-demand"]].apply(
        pd.to_numeric, errors="coerce"
    )
    abs_error = (data["day-ahead-forecast"] - data["actual-demand"]).abs()

    bins = list(range(0, 11)) + [float("inf")]
    labels = ["1", "2", "3", "4", "5",
              "6", "7", "8", "9", "10", ">10"]
    categories = pd.cut(
        abs_error,
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    counts = categories.value_counts().reindex(labels, fill_value=0)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values, color="#10b981", alpha=0.85)
    plt.title("Absolute Error (Binned)")
    plt.xlabel("Absolute Error")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mae_bar_chart.png", dpi=150)
    plt.close()


def mae_by_day_of_week() -> None:
    data = pd.read_csv("data/data.csv")
    numeric = data[["day-ahead-forecast", "actual-demand"]].apply(
        pd.to_numeric, errors="coerce"
    )
    abs_error = (numeric["day-ahead-forecast"] -
                 numeric["actual-demand"]).abs()
    percent_error = (
        abs_error / numeric["actual-demand"].replace(0, pd.NA) * 100
    )
    dt = pd.to_datetime(
        data["date"].astype(str) + " " + data["time"].astype(str),
        errors="coerce",
    )
    data = data.assign(
        abs_error=abs_error,
        percent_error=percent_error,
        day=dt.dt.day_name(),
    )

    day_order = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ]
    day_means = (
        data.dropna(subset=["day", "abs_error"])
        .groupby("day")["abs_error"]
        .mean()
        .reindex(day_order)
    )
    day_percent_means = (
        data.dropna(subset=["day", "percent_error"])
        .groupby("day")["percent_error"]
        .mean()
        .reindex(day_order)
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(day_means.index, day_means.values, color="#f59e0b", alpha=0.85)
    ax.set_title("Average Absolute Error by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Absolute Error")
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(
        day_percent_means.index,
        day_percent_means.values,
        color="#2563eb",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Average Absolute Percentage Error (%)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mae_by_day_of_week.png", dpi=150)
    plt.close(fig)


def mae_by_month() -> None:
    data = pd.read_csv("data/data.csv")
    numeric = data[["day-ahead-forecast", "actual-demand"]].apply(
        pd.to_numeric, errors="coerce"
    )
    abs_error = (numeric["day-ahead-forecast"] -
                 numeric["actual-demand"]).abs()
    percent_error = (
        abs_error / numeric["actual-demand"].replace(0, pd.NA) * 100
    )
    dt = pd.to_datetime(
        data["date"].astype(str) + " " + data["time"].astype(str),
        errors="coerce",
    )
    data = data.assign(
        abs_error=abs_error,
        percent_error=percent_error,
        month=dt.dt.month_name(),
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_means = (
        data.dropna(subset=["month", "abs_error"])
        .groupby("month")["abs_error"]
        .mean()
        .reindex(month_order)
    )
    month_percent_means = (
        data.dropna(subset=["month", "percent_error"])
        .groupby("month")["percent_error"]
        .mean()
        .reindex(month_order)
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(month_means.index, month_means.values, color="#a855f7", alpha=0.85)
    ax.set_title("Average Absolute Error by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Absolute Error")
    ax.tick_params(axis="x", rotation=45)

    ax2 = ax.twinx()
    ax2.plot(
        month_percent_means.index,
        month_percent_means.values,
        color="#2563eb",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Average Absolute Percentage Error (%)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mae_by_month.png", dpi=150)
    plt.close(fig)


def mae_by_hour() -> None:
    data = pd.read_csv("data/data.csv")
    numeric = data[["day-ahead-forecast", "actual-demand"]].apply(
        pd.to_numeric, errors="coerce"
    )
    abs_error = (numeric["day-ahead-forecast"] -
                 numeric["actual-demand"]).abs()
    percent_error = (
        abs_error / numeric["actual-demand"].replace(0, pd.NA) * 100
    )
    dt = pd.to_datetime(
        data["date"].astype(str) + " " + data["time"].astype(str),
        errors="coerce",
    )
    data = data.assign(
        abs_error=abs_error,
        percent_error=percent_error,
        hour=dt.dt.hour,
    )

    hour_means = (
        data.dropna(subset=["hour", "abs_error"])
        .groupby("hour")["abs_error"]
        .mean()
        .reindex(range(24))
    )
    hour_percent_means = (
        data.dropna(subset=["hour", "percent_error"])
        .groupby("hour")["percent_error"]
        .mean()
        .reindex(range(24))
    )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(hour_means.index, hour_means.values, color="#ef4444", alpha=0.85)
    ax.set_title("Average Absolute Error by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Absolute Error")
    ax.set_xticks(range(0, 24, 1))

    ax2 = ax.twinx()
    ax2.plot(
        hour_percent_means.index,
        hour_percent_means.values,
        color="#2563eb",
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Average Absolute Percentage Error (%)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mae_by_hour.png", dpi=150)
    plt.close(fig)


def main() -> None:
    mae_percent_bar_chart()
    mae_bar_chart()
    mae_by_day_of_week()
    mae_by_month()
    mae_by_hour()


if __name__ == "__main__":
    main()
