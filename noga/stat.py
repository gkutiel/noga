from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path("plots")


def mae_by_day(df: pd.DataFrame, year: int = 2024):
    df = df[df['year'] == year]

    df = df.copy()
    df['mae'] = (df['forecast'] - df['actual']).abs()
    df['mae_percent'] = df['mae'] / df['actual'] * 100
    agg = df.groupby('day')[['mae', 'mae_percent']].mean()

    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    x = np.arange(7)
    width = 0.4

    fig, ax1 = plt.subplots()
    color1 = 'steelblue'
    ax1.bar(
        x - width / 2, agg['mae'],
        width, color=color1, alpha=0.8, label='MAE')

    ax1.set_xlabel('Day of the week')
    ax1.set_ylabel('Mean Absolute Error (MAE)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)

    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.bar(
        x + width / 2, agg['mae_percent'],
        width, color=color2, alpha=0.8, label='MAE %')

    ax2.set_ylabel('MAE (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title(f'MAE by Day of the Week ({year})')
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / f"mae_by_day_{year}.png")


def mae():
    data = pd.read_csv("data/data.csv")
    data_2024 = data[data['year'] == 2024]
    mae_2024 = (data_2024['forecast'] - data_2024['actual']).abs().mean()

    data_2025 = data[data['year'] == 2025]
    mae_2025 = (data_2025['forecast'] - data_2025['actual']).abs().mean()

    print(f"MAE for 2024: {mae_2024:.2f}")
    print(f"MAE for 2025: {mae_2025:.2f}")


if __name__ == "__main__":
    # data = pd.read_csv("data/data.csv")
    # mae_by_day(data, year=2024)
    # mae_by_day(data, year=2025)
    mae()
    pass
