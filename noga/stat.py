from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS_DIR = Path("plots")


def mae_by_day(df: pd.DataFrame):
    df['mae'] = (df['forecast'] - df['actual']).abs()
    df['mae_percent'] = df['mae'] / df['actual'] * 100
    return df.groupby(['day'])[['mae', 'mae_percent']].mean().reset_index()


if __name__ == "__main__":
    data = pd.read_csv("data/data.csv")
    data_2024 = data[data['year'] == 2024]
    mae_by_day_2024 = mae_by_day(data_2024)

    print(mae_by_day_2024)
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

    x = np.arange(7)
    width = 0.4

    fig, ax1 = plt.subplots()
    color1 = 'steelblue'
    ax1.bar(
        x - width / 2, mae_by_day_2024['mae'],
        width, color=color1, alpha=0.8, label='MAE')

    ax1.set_xlabel('Day of the week')
    ax1.set_ylabel('Mean Absolute Error (MAE)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(days)

    ax2 = ax1.twinx()
    color2 = 'tomato'
    ax2.bar(x + width / 2, mae_by_day_2024['mae_percent'],
            width, color=color2, alpha=0.8, label='MAE %')
    ax2.set_ylabel('MAE (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('MAE by Day of the Week (2024)')
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.savefig(PLOTS_DIR / "mae_by_day_2024.png")
